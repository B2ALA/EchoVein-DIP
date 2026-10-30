"""Multi-cue moisture evidence on the rectified wall map.

Cues (all vs. a dry baseline): darkening, saturation gain, texture/edge blur,
specular-sheen loss, (optional) thermal cooling. Illumination is normalised with a
smooth lighting field estimated from the non-wet part of the wall."""
import cv2
import numpy as np


def _L(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)


def _S(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32) / 255.0


def light_field(ratio, iters=3, sigma_frac=0.12, global_only=False):
    """Smooth multiplicative lighting change, estimated robustly (wet = darker outliers excluded)."""
    if global_only:
        return np.full_like(ratio, np.median(ratio))
    sig = max(8.0, sigma_frac * ratio.shape[1])
    w = np.ones_like(ratio)
    light = np.full_like(ratio, np.median(ratio))
    for _ in range(iters):
        num = cv2.GaussianBlur(ratio * w, (0, 0), sig)
        den = cv2.GaussianBlur(w, (0, 0), sig) + 1e-6
        light = num / den
        w = (ratio / np.maximum(light, 1e-3) > 0.96).astype(np.float32)
        w = cv2.erode(w, np.ones((5, 5), np.uint8))
        if w.mean() < 0.3:
            break
    return light


def _tex(Ls):
    g = cv2.GaussianBlur(Ls, (0, 0), 1.0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    e = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 3.5)
    return e / (cv2.GaussianBlur(Ls, (0, 0), 3.5) + 1.0)          # relative contrast (light invariant)


def _spec(Ls):
    hp = cv2.GaussianBlur(Ls, (0, 0), 1.0) - cv2.GaussianBlur(Ls, (0, 0), 9.0)
    return cv2.GaussianBlur(np.maximum(hp - 3.0, 0), (0, 0), 3.0)


def cue_maps(ref, cur, use_illum=True, thermal=None):
    """Returns dict with raw cues + normalised 0..1 cues + wall-level descriptors."""
    Lr, Lc = _L(ref), _L(cur)
    Lrb, Lcb = cv2.GaussianBlur(Lr, (0, 0), 2.5), cv2.GaussianBlur(Lc, (0, 0), 2.5)
    ratio = (Lcb + 1.0) / (Lrb + 1.0)
    light = light_field(ratio, global_only=not use_illum)
    dark_raw = 1.0 - ratio / np.maximum(light, 1e-3)
    dark = np.clip(dark_raw, 0, 1)

    Sr, Sc = cv2.GaussianBlur(_S(ref), (0, 0), 3), cv2.GaussianBlur(_S(cur), (0, 0), 3)
    sat_raw = Sc / (Sr + 0.03) - 1.0
    tex_raw = 1.0 - _tex(Lc) / (_tex(Lr) + 1e-3)
    tex_raw = cv2.GaussianBlur(tex_raw, (0, 0), 2.0)
    spec_raw = (_spec(Lr) - _spec(Lc)) / 6.0

    cues = {
        "dark": dark, "sat": np.clip(sat_raw / 0.25, 0, 1),
        "tex": np.clip(tex_raw / 0.5, 0, 1), "spec": np.clip(spec_raw, 0, 1),
        "raw": {"dark": dark_raw, "sat": sat_raw, "tex": tex_raw, "spec": spec_raw},
        "light_level": float(np.median(ratio)),          # global exposure proxy
        "wall_tex": float(np.median(np.clip(tex_raw / 0.5, 0, 1))),
        "wall_sat": float(np.median(np.clip(sat_raw / 0.25, 0, 1))),
    }
    if thermal is not None:
        cues["thermal"] = thermal
    return cues


def fuse_weighted(c, multicue=True):
    dark_n = np.clip(c["dark"] / 0.25, 0, 1)
    if not multicue:
        return dark_n
    ev = 0.40 * dark_n + 0.20 * c["sat"] + 0.30 * c["tex"] + 0.10 * c["spec"]
    if "thermal" in c:
        ev = 0.8 * ev + 0.2 * c["thermal"]
    return np.clip(ev * np.clip(dark_n * 4, 0, 1), 0, 1)     # no darkening -> not wet


def feature_stack(c):
    """Per-pixel features for the ML pixel classifier (H,W,F)."""
    r = c["raw"]
    bl = lambda a, s: cv2.GaussianBlur(a, (0, 0), s)
    f = [r["dark"], r["sat"], r["tex"], r["spec"],
         bl(r["dark"], 6), bl(r["sat"], 6), bl(r["tex"], 6),
         np.full_like(r["dark"], np.median(r["tex"])), np.full_like(r["dark"], np.median(r["sat"]))]
    return np.dstack(f).astype(np.float32)
