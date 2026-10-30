"""Wetting-front back-tracing, pipe prior, assets-at-risk, relative severity index."""
import json
import numpy as np


def estimate_source(tr, gravity_prior=0.35):
    """Source = where the patch centre was when radius->0 (regress centroid on radius).
    Gravity: water spreads sideways and sinks, so the centre drifts down as the patch grows;
    extrapolating to r=0 removes that bias. Returns source (cm) + 1-sigma uncertainty (cm)."""
    ph = tr.present_hist
    if not ph:
        return None
    r = np.array([x["r_eq"] for x in ph]); cx = np.array([x["cx"] for x in ph]); cy = np.array([x["cy"] for x in ph])
    r0 = float(r[0])
    if len(ph) >= 3 and np.ptp(r) > 3.0:
        A = np.c_[np.ones_like(r), r]
        (ax, bx), *_ = np.linalg.lstsq(A, cx, rcond=None)
        (ay, by), *_ = np.linalg.lstsq(A, cy, rcond=None)
        res_x, res_y = cx - A @ [ax, bx], cy - A @ [ay, by]
        sx = float(np.sqrt(np.mean(res_x ** 2) + (0.15 * r0) ** 2))
        sy = float(np.sqrt(np.mean(res_y ** 2) + (0.15 * r0) ** 2))
        # keep the extrapolation inside the observed first patch
        sxp = float(np.clip(ax, cx[0] - r0, cx[0] + r0)); syp = float(np.clip(ay, cy[0] - r0, cy[0] + r0))
        method = "regression to r=0"
    else:
        sxp, syp = float(cx[0]), float(cy[0] - gravity_prior * r0)
        sx = sy = 0.5 * r0 + 3.0
        method = "first-seen centre with gravity prior"
    sx, sy = max(sx, 3.0) + 0.1 * r0, max(sy, 3.0) + 0.1 * r0
    conf = float(np.clip(len(ph) / 6.0, 0.2, 1.0))
    return {"x_cm": sxp, "y_cm": syp, "sx_cm": sx, "sy_cm": sy, "method": method,
            "confidence": conf, "n_obs": len(ph)}


def load_layout(path):
    if not path:
        return {"pipes": [], "assets": []}
    return json.load(open(path))


def _closest_on_polyline(p, pts):
    best = (1e18, None, 0.0)
    acc = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        a, b = np.array(a, float), np.array(b, float)
        ab = b - a; L = float(np.linalg.norm(ab))
        u = np.clip(np.dot(p - a, ab) / (L * L + 1e-9), 0, 1)
        q = a + u * ab
        d = float(np.linalg.norm(p - q))
        if d < best[0]:
            best = (d, q, acc + u * L)
        acc += L
    return best


def pipe_prior(src, layout, tol_cm=4.0):
    """Probability that each drawn pipe segment is the source (a probability, not a guarantee)."""
    if not src or not layout.get("pipes"):
        return []
    p = np.array([src["x_cm"], src["y_cm"]])
    sig = np.sqrt(0.5 * (src["sx_cm"] ** 2 + src["sy_cm"] ** 2)) + tol_cm
    rows = []
    for pipe in layout["pipes"]:
        d, q, arc = _closest_on_polyline(p, pipe["points"])
        rows.append({"pipe": pipe["name"], "dist_cm": d, "inspect_x_cm": float(q[0]), "inspect_y_cm": float(q[1]),
                     "arc_length_cm": arc + 100.0 * pipe.get("start_m", 0.0), "lik": float(np.exp(-0.5 * (d / sig) ** 2))})
    z = sum(r["lik"] for r in rows) + 0.05          # 0.05 = "none of the drawn pipes"
    for r in rows:
        r["prob"] = r["lik"] / z
    return sorted(rows, key=lambda r: -r["prob"])


def growth_rate(tr, k=5):
    ph = tr.present_hist[-k:]
    if len(ph) < 3 or np.ptp([x["t"] for x in ph]) <= 0:
        return 0.0, 0.0
    t = np.array([x["t"] for x in ph]); r = np.array([x["r_eq"] for x in ph])
    A = np.c_[t - t.mean(), np.ones_like(t)]
    coef, res, *_ = np.linalg.lstsq(A, r, rcond=None)
    resid = r - A @ coef
    se = float(np.sqrt(np.sum(resid ** 2) / max(len(t) - 2, 1)) / (np.sqrt(np.sum((t - t.mean()) ** 2)) + 1e-9))
    return float(coef[0]), se             # cm/h radius growth, std-error


def assets_at_risk(tr, layout):
    out = []
    ph = tr.present_hist
    if not ph or not layout.get("assets"):
        return out
    c = np.array([ph[-1]["cx"], ph[-1]["cy"]]); r = ph[-1]["r_eq"]
    rate, se = growth_rate(tr)
    for a in layout["assets"]:
        d = float(np.linalg.norm(np.array([a["x"], a["y"]]) - c) - r)
        electrical = bool(a.get("electrical", a.get("type") in ("socket", "switchboard", "switch")))
        if d <= 0:
            out.append({"name": a["name"], "electrical": electrical, "status": "reached", "t_min_h": 0.0, "t_max_h": 0.0, "dist_cm": 0.0})
        elif rate > 0.5:
            out.append({"name": a["name"], "electrical": electrical, "status": "approaching", "dist_cm": d,
                        "t_min_h": d / (rate + 2 * se), "t_max_h": d / max(rate - 2 * se, 0.2)})
        else:
            out.append({"name": a["name"], "electrical": electrical, "status": "not growing toward it", "dist_cm": d,
                        "t_min_h": None, "t_max_h": None})
    return out


def severity_index(tr, wall_area_cm2):
    """Relative severity 0..100 (NOT litres/hour - needs calibration against measured volumes)."""
    ph = tr.present_hist
    if not ph:
        return 0.0
    area = ph[-1]["area"]; rate, _ = growth_rate(tr)
    r = ph[-1]["r_eq"]
    area_rate = 2 * np.pi * r * max(rate, 0.0)             # cm2 / h
    g0, a0 = 0.02 * wall_area_cm2, 0.05 * wall_area_cm2
    return float(100 * (1 - np.exp(-(area_rate / g0 + area / a0))))


def forecast_growth(tr, min_span_h=10 / 60.0, horizons_h=(1 / 6, 1.0)):
    """Linear area trend extrapolation 'if the current trend continues' (+/-2 standard errors).
    Returns None unless the region was observed for >= min_span_h (live-scan seconds are too short)."""
    ph = tr.present_hist[-8:]
    if len(ph) < 4:
        return None
    t = np.array([x["t"] for x in ph]); a = np.array([x["area"] for x in ph])
    if np.ptp(t) < min_span_h:
        return None
    A = np.c_[t - t.mean(), np.ones_like(t)]
    coef, *_ = np.linalg.lstsq(A, a, rcond=None)
    res = a - A @ coef
    s2 = float(np.sum(res ** 2) / max(len(t) - 2, 1))
    out = []
    for h in horizons_h:
        dt = t[-1] + h - t.mean()
        pred = float(coef[0] * dt + coef[1])
        se = float(np.sqrt(s2 * (1.0 / len(t) + dt ** 2 / (np.sum((t - t.mean()) ** 2) + 1e-9))))
        out.append({"horizon_h": float(h), "area_cm2": max(pred, 0.0), "lo": max(pred - 2 * se, 0.0), "hi": max(pred + 2 * se, 0.0)})
    return {"trend_cm2_per_h": float(coef[0]), "points": out,
            "note": "linear extrapolation if the current trend continues; not a prediction of real leak behaviour"}
