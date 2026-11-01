"""Thermal module (v2). Works with a radiometric array (e.g. MLX90640 24x32 degC, FLIR/Seek exports)
or a thermal dataset (.npy/.csv/.json of temperatures). Registered to the rectified wall map.

Pipeline: register -> smooth -> robust local background -> cold/hot anomaly masks (hysteresis) ->
regions (outline polygons, area cm2, perimeter, skeleton PATH LENGTH cm, delta degC, position) -> heat-map image.
Thermal anomalies have many non-leak causes (ambient, insulation, airflow, sun, heaters): always verify on site."""
import json
import cv2
import numpy as np
from .evidence import Evidence

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ------------------------------------------------------------------ io / registration
def load_temp(path):
    p = str(path).lower()
    if p.endswith(".npy"):
        return np.load(path).astype(np.float32)
    if p.endswith(".json"):
        with open(path) as f:
            d = json.load(f)
        return np.array(d["temps"] if isinstance(d, dict) else d, np.float32)
    return np.loadtxt(path, delimiter=",").astype(np.float32)


def register(temp, corners_in_temp_px, map_size):
    """Warp a temperature array onto the rectified wall map. corners order TL,TR,BR,BL."""
    W, H = map_size
    dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    M = cv2.getPerspectiveTransform(np.float32(corners_in_temp_px), dst)
    return cv2.warpPerspective(np.ascontiguousarray(temp, np.float32), M, (W, H), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


# ------------------------------------------------------------------ background / cue
def _background(t):
    """Robust smooth wall background (normalised convolution that ignores outliers = the anomalies)."""
    s = 64.0 / max(t.shape)
    small = cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else t.copy()
    sig = max(2.0, min(small.shape) / 6.0)
    w = np.ones_like(small)
    bg = small
    for _ in range(4):
        den = cv2.GaussianBlur(w, (0, 0), sig) + 1e-6
        bg = cv2.GaussianBlur(small * w, (0, 0), sig) / den
        r = small - bg
        mad = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-3
        w = (np.abs(r) < 2.5 * mad).astype(np.float32)
        if w.sum() < 10:
            w = np.ones_like(small)
    return cv2.resize(bg, (t.shape[1], t.shape[0]), interpolation=cv2.INTER_CUBIC) if s < 1 else bg


def thermal_cue(temp_map, delta_c=1.2, sigma=3.0):
    """Cool anomaly (evaporation / cold leaking water) vs local wall background -> 0..1 (used by the fusion model)."""
    t = cv2.GaussianBlur(np.ascontiguousarray(temp_map, np.float32), (0, 0), float(sigma))
    return np.clip((_background(t) - t) / max(float(delta_c), 1e-3), 0, 1).astype(np.float32)


# ------------------------------------------------------------------ skeleton / path length
def _thin(mask):
    """Vectorised Zhang-Suen thinning -> 1-px skeleton (bool)."""
    img = np.pad((mask > 0).astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p2, p3, p4, p5 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:], img[2:, 2:]
            p6, p7, p8, p9 = img[2:, 1:-1], img[2:, :-2], img[1:-1, :-2], img[:-2, :-2]
            P = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(p.astype(np.int16) for p in P)
            A = sum(((P[i] == 0) & (P[(i + 1) % 8] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                c = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                c = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rm = (img[1:-1, 1:-1] == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if rm.any():
                img[1:-1, 1:-1][rm] = 0
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1] > 0


def _edge_len(sk):
    s = sk.astype(np.uint8)
    if s.sum() == 0:
        return 0.0
    L = int((s[:, 1:] & s[:, :-1]).sum()) + int((s[1:, :] & s[:-1, :]).sum()) + \
        1.4142 * (int((s[1:, 1:] & s[:-1, :-1]).sum()) + int((s[1:, :-1] & s[:-1, 1:]).sum()))
    return float(L) + 1.0


def _path_length(comp):
    """Length (px) of the medial-axis path of a region + the skeleton mask at full crop size."""
    h, w = comp.shape
    f = min(1.0, 160.0 / max(h, w))
    small = comp.astype(np.uint8)
    if f < 1.0:
        small = (cv2.resize(small * 255, (max(3, int(w * f)), max(3, int(h * f))), interpolation=cv2.INTER_AREA) > 127).astype(np.uint8)
    sk = _thin(small)
    length = _edge_len(sk) / f
    full = cv2.resize(sk.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0 if f < 1.0 else sk
    return length, full & comp


# ------------------------------------------------------------------ analysis
def analyze_thermal(temp_map, px_per_cm, delta_c=1.2, min_area_cm2=30.0, sigma=3.0, max_regions=8):
    """Detect cold (possible moisture/evaporation) and hot anomalies on the wall-map temperature array."""
    raw = np.ascontiguousarray(temp_map, np.float32)
    H, W = raw.shape
    ppc = max(float(px_per_cm), 1e-6)
    t = cv2.GaussianBlur(raw, (0, 0), float(sigma))
    bg = _background(t)
    d = t - bg
    mad = float(1.4826 * np.median(np.abs(d - np.median(d))))
    thr = max(float(delta_c), 3.0 * mad)
    min_px = max(9.0, float(min_area_cm2) * ppc * ppc)
    k3 = np.ones((3, 3), np.uint8)
    found = []
    masks = {}
    skel = np.zeros((H, W), bool)
    for kind, sgn in (("cold", -1.0), ("hot", 1.0)):
        s = d * sgn
        n, lab = cv2.connectedComponents((s >= 0.5 * thr).astype(np.uint8))      # hysteresis: grow at 0.5*thr ...
        keep = np.zeros(n, bool)
        keep[np.unique(lab[s >= thr])] = True                                        # ... only where seeded at thr
        keep[0] = False
        m = cv2.morphologyEx(keep[lab].astype(np.uint8), cv2.MORPH_CLOSE, k3)
        n2, lab2, st, cen = cv2.connectedComponentsWithStats(m)
        full = np.zeros((H, W), bool)
        for i in range(1, n2):
            area_px = float(st[i, cv2.CC_STAT_AREA])
            if area_px < min_px:
                continue
            x, y, w, h = (int(v) for v in st[i, :4])
            comp = lab2[y:y + h, x:x + w] == i
            full[y:y + h, x:x + w] |= comp
            ds, ts, ss = d[y:y + h, x:x + w][comp], t[y:y + h, x:x + w][comp], s[y:y + h, x:x + w][comp]
            mean_d, peak_d = float(ds.mean()), float(sgn * ss.max())
            peak_c = float(ts.min() if kind == "cold" else ts.max())
            cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(x, y))
            polys, perim = [], 0.0
            for c in cnts:
                al = cv2.arcLength(c, True)
                perim += al
                a = cv2.approxPolyDP(c, max(1.0, 0.01 * al), True).reshape(-1, 2)
                if len(a) >= 3:
                    polys.append(np.round(a / [W, H], 4).tolist())
            plen, sk = _path_length(comp)
            skel[y:y + h, x:x + w] |= sk
            lvl = "strong" if abs(peak_d) >= 4 else "moderate" if abs(peak_d) >= 2 else "mild"
            found.append({
                "kind": kind, "level": lvl,
                "label": "cool anomaly (possible moisture / evaporation)" if kind == "cold" else "warm anomaly (heat source; not a leak indicator)",
                "delta_c": round(mean_d, 2), "peak_delta_c": round(peak_d, 2), "peak_c": round(peak_c, 2), "mean_c": round(float(ts.mean()), 2),
                "area_cm2": round(area_px / ppc ** 2, 1), "perimeter_cm": round(perim / ppc, 1), "path_length_cm": round(plen / ppc, 1),
                "width_cm": round(w / ppc, 1), "height_cm": round(h / ppc, 1),
                "x_cm": round(float(cen[i][0]) / ppc, 1), "y_cm": round(float(cen[i][1]) / ppc, 1),
                "left_cm": round(x / ppc, 1), "right_cm": round((W - x - w) / ppc, 1), "top_cm": round(y / ppc, 1), "bottom_cm": round((H - y - h) / ppc, 1),
                "centroid_norm": [round(float(cen[i][0]) / W, 4), round(float(cen[i][1]) / H, 4)], "polygons": polys,
                "confidence": round(float(np.clip(abs(peak_d) / (3.0 * thr), 0, 1)), 2),
                "note": "relative evidence strength, not a probability; thermal anomalies have many non-leak causes"})
        masks[kind] = full
    found.sort(key=lambda r: -r["confidence"])
    cnt = {"cold": 0, "hot": 0}
    regions = []
    for r in found:
        if len(regions) >= max_regions:
            break
        cnt[r["kind"]] += 1
        r["id"] = f"{'C' if r['kind'] == 'cold' else 'H'}{cnt[r['kind']]}"
        regions.append(r)
    cold = [r for r in regions if r["kind"] == "cold"]
    hot = [r for r in regions if r["kind"] == "hot"]
    return {
        "stats": {"min": float(raw.min()), "max": float(raw.max()), "mean": float(raw.mean()), "median": float(np.median(raw)),
                  "bg_median": float(np.median(bg)), "noise_c": round(mad, 3), "threshold_c": round(thr, 2), "map_px": [W, H], "px_per_cm": ppc},
        "regions": regions, "n_cold": len(cold), "n_hot": len(hot),
        "cold_area_cm2": float(sum(r["area_cm2"] for r in cold)), "hot_area_cm2": float(sum(r["area_cm2"] for r in hot)),
        "total_path_cm": float(sum(r["path_length_cm"] for r in cold)),
        "max_delta_c": float(max([abs(r["peak_delta_c"]) for r in regions], default=0.0)),
        "coverage_pct": float(100.0 * (masks["cold"] | masks["hot"]).sum() / (H * W)),
        "_cold_mask": masks["cold"], "_hot_mask": masks["hot"], "_skel": skel}


# ------------------------------------------------------------------ rendering
def _label(img, txt, org, scale=0.45):
    cv2.putText(img, txt, org, FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, org, FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)


def render_heatmap(temp_map, an, px_per_cm=None, units="cm", max_w=640):
    """BGR image: JET heat-map, outlined cold (cyan) / hot (white, dashed look) regions, yellow path skeleton, colour-bar."""
    t = np.asarray(temp_map, np.float32)
    H, W = t.shape
    lo, hi = (float(v) for v in np.percentile(t, (1, 99)))
    if hi - lo < 2.0:
        mid = (hi + lo) / 2
        lo, hi = mid - 1.0, mid + 1.0
    s = max_w / float(W)
    w2, h2 = int(W * s), int(H * s)
    img = cv2.applyColorMap((np.clip((t - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img = cv2.resize(img, (w2, h2), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    sk = cv2.resize(an["_skel"].astype(np.uint8), (w2, h2), interpolation=cv2.INTER_NEAREST)
    sk = cv2.dilate(sk, np.ones((2, 2), np.uint8))
    img[sk > 0] = (0, 255, 255)
    U = "%" if units == "relative" or units == "%" else "cm"
    for r in an["regions"]:
        col = (255, 255, 0) if r["kind"] == "cold" else (255, 255, 255)
        for poly in r["polygons"]:
            cv2.polylines(img, [(np.array(poly) * [w2, h2]).astype(np.int32)], True, col, 2, cv2.LINE_AA)
        cx, cy = r["centroid_norm"]
        txt = f"{r['id']} {r['delta_c']:+.1f}C {r['area_cm2']:.0f}{U}2"
        tw = cv2.getTextSize(txt, FONT, 0.45, 1)[0][0]
        _label(img, txt, (int(min(max(2, cx * w2 + 4), w2 - tw - 2)), int(min(max(14, cy * h2), h2 - 4))))
    top, side = 26, 74
    canvas = np.zeros((h2 + top, w2 + side, 3), np.uint8)
    canvas[top:, :w2] = img
    bar = cv2.applyColorMap(np.linspace(255, 0, h2).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET)
    canvas[top:, w2 + 8:w2 + 24] = np.repeat(bar, 16, axis=1)
    _label(canvas, f"{hi:.1f}C", (w2 + 28, top + 12), 0.4)
    _label(canvas, f"{lo:.1f}C", (w2 + 28, top + h2 - 4), 0.4)
    _label(canvas, f"Thermal | cold {an['n_cold']} hot {an['n_hot']} | thr +/-{an['stats']['threshold_c']:.1f}C | path {an['total_path_cm']:.0f}{U}", (6, 17), 0.48)
    return canvas


# ------------------------------------------------------------------ demo data (clearly simulated)
def demo_thermal(shape=(24, 32), seed=7):
    """MLX90640-like 24x32 frame: 24 C wall + a cold damp-like streak + a warm spot. SIMULATED, for UI testing only."""
    rng = np.random.default_rng(seed)
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = x / (w - 1), y / (h - 1)
    t = 24.0 + 0.6 * u - 0.4 * v + rng.normal(0, 0.08, shape)
    p0, p1 = np.array([0.30, 0.12]), np.array([0.48, 0.88])
    dv = p1 - p0
    sp = np.clip(((u - p0[0]) * dv[0] + (v - p0[1]) * dv[1]) / float(dv @ dv), 0, 1)
    dist = np.hypot(u - (p0[0] + sp * dv[0]), v - (p0[1] + sp * dv[1]))
    t -= 3.0 * np.exp(-(dist / 0.055) ** 2) * (0.6 + 0.4 * np.sin(np.pi * sp))
    t += 3.5 * np.exp(-((u - 0.82) ** 2 + (v - 0.28) ** 2) / 0.012)
    return t.astype(np.float32)


# ------------------------------------------------------------------ legacy evidence (kept for the fusion pipeline)
def thermal_evidence(wall, temp_map, px_per_cm, t, thr=0.5, min_area_cm2=30):
    cue = thermal_cue(temp_map)
    m = (cue > thr).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA] / px_per_cm ** 2
        if a >= min_area_cm2:
            conf = float(min(1.0, cue[lab == i].mean()))
            out.append(Evidence("thermal", wall, cen[i][0] / px_per_cm, cen[i][1] / px_per_cm, conf, t,
                                {"area_cm2": a, "note": "thermal anomalies have many non-leak causes"}))
    return out