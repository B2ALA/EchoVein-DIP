
"""Multi-cue moisture evidence on the rectified wall map.

Cues (all vs. a dry baseline):
- Darkening
- Saturation gain
- Texture / edge blur
- Specular-sheen loss
- Optional thermal cooling

Lighting changes are normalised using a smooth lighting field.
"""

import cv2
import numpy as np


def _validate_image(img, name="image"):
    """Validate an input BGR image."""
    if not isinstance(img, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array.")

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"{name} must have 3 BGR colour channels.")

    if img.shape[0] < 16 or img.shape[1] < 16:
        raise ValueError(f"{name} must be at least 16x16 pixels.")

    if not np.isfinite(img).all():
        raise ValueError(f"{name} contains invalid numeric values.")

    return img.astype(np.uint8) if img.dtype != np.uint8 else img


def _L(img):
    """Convert BGR image to LAB lightness."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)


def _S(img):
    """Extract normalised HSV saturation."""
    return (
        cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
        / 255.0
    )


def light_field(ratio, iters=3, sigma_frac=0.12, global_only=False):
    """Estimate smooth multiplicative lighting change."""

    if not isinstance(ratio, np.ndarray) or ratio.ndim != 2:
        raise ValueError("ratio must be a 2D NumPy array.")

    if ratio.size == 0 or not np.isfinite(ratio).all():
        raise ValueError("ratio must contain finite values.")

    if iters < 1:
        raise ValueError("iters must be at least 1.")

    if sigma_frac <= 0:
        raise ValueError("sigma_frac must be positive.")

    ratio = np.clip(ratio.astype(np.float32), 1e-3, 100.0)

    if global_only:
        return np.full_like(ratio, np.median(ratio))

    sig = max(8.0, sigma_frac * ratio.shape[1])

    weights = np.ones_like(ratio)
    light = np.full_like(ratio, np.median(ratio))

    for _ in range(iters):
        numerator = cv2.GaussianBlur(
            ratio * weights, (0, 0), sig
        )

        denominator = cv2.GaussianBlur(
            weights, (0, 0), sig
        ) + 1e-6

        light = numerator / denominator

        weights = (
            ratio / np.maximum(light, 1e-3) > 0.96
        ).astype(np.float32)

        weights = cv2.erode(
            weights, np.ones((5, 5), np.uint8)
        )

        if weights.mean() < 0.3:
            break

    return np.maximum(light, 1e-3)


def _tex(Ls):
    """Calculate relative wall texture / edge contrast."""

    blurred = cv2.GaussianBlur(Ls, (0, 0), 1.0)

    gx = cv2.Sobel(
        blurred, cv2.CV_32F, 1, 0, ksize=3
    )

    gy = cv2.Sobel(
        blurred, cv2.CV_32F, 0, 1, ksize=3
    )

    edges = cv2.GaussianBlur(
        np.sqrt(gx * gx + gy * gy),
        (0, 0),
        3.5
    )

    brightness = cv2.GaussianBlur(
        Ls, (0, 0), 3.5
    )

    return edges / (brightness + 1.0)


def _spec(Ls):
    """Estimate local specular-sheen evidence."""

    high_pass = (
        cv2.GaussianBlur(Ls, (0, 0), 1.0)
        - cv2.GaussianBlur(Ls, (0, 0), 9.0)
    )

    return cv2.GaussianBlur(
        np.maximum(high_pass - 3.0, 0),
        (0, 0),
        3.0
    )


def cue_maps(ref, cur, use_illum=True, thermal=None):
    """Return raw cues, normalised cues and wall-level descriptors."""

    ref = _validate_image(ref, "ref")
    cur = _validate_image(cur, "cur")

    if ref.shape != cur.shape:
        raise ValueError(
            "Reference and current images must have identical dimensions."
        )

    if thermal is not None:
        thermal = np.asarray(thermal, dtype=np.float32)

        if thermal.shape != ref.shape[:2]:
            raise ValueError(
                "Thermal map must match the image height and width."
            )

        if not np.isfinite(thermal).all():
            raise ValueError("Thermal map contains invalid values.")

        thermal = np.clip(thermal, 0.0, 1.0)

    # Lightness comparison
    Lr = _L(ref)
    Lc = _L(cur)

    Lrb = cv2.GaussianBlur(Lr, (0, 0), 2.5)
    Lcb = cv2.GaussianBlur(Lc, (0, 0), 2.5)

    ratio = (Lcb + 1.0) / (Lrb + 1.0)

    light = light_field(
        ratio,
        global_only=not use_illum
    )

    dark_raw = 1.0 - ratio / np.maximum(light, 1e-3)
    dark = np.clip(dark_raw, 0.0, 1.0)

    # Saturation comparison
    Sr = cv2.GaussianBlur(_S(ref), (0, 0), 3)
    Sc = cv2.GaussianBlur(_S(cur), (0, 0), 3)

    sat_raw = Sc / (Sr + 0.03) - 1.0

    # Texture comparison
    tex_raw = 1.0 - _tex(Lc) / (_tex(Lr) + 1e-3)
    tex_raw = cv2.GaussianBlur(tex_raw, (0, 0), 2.0)

    # Specular-sheen comparison
    spec_raw = (_spec(Lr) - _spec(Lc)) / 6.0

    sat = np.clip(sat_raw / 0.25, 0.0, 1.0)
    tex = np.clip(tex_raw / 0.5, 0.0, 1.0)
    spec = np.clip(spec_raw, 0.0, 1.0)

    cues = {
        "dark": dark,
        "sat": sat,
        "tex": tex,
        "spec": spec,

        "raw": {
            "dark": dark_raw,
            "sat": sat_raw,
            "tex": tex_raw,
            "spec": spec_raw,
        },

        "light_level": float(np.median(ratio)),
        "wall_tex": float(np.median(tex)),
        "wall_sat": float(np.median(sat)),
    }

    if thermal is not None:
        cues["thermal"] = thermal

    return cues


def fuse_weighted(c, multicue=True):
    """Combine moisture evidence into a normalised 0-1 map."""

    dark_n = np.clip(
        c["dark"] / 0.25,
        0.0,
        1.0
    )

    if not multicue:
        return dark_n

    ev = (
        0.40 * dark_n
        + 0.20 * c["sat"]
        + 0.30 * c["tex"]
        + 0.10 * c["spec"]
    )

    if "thermal" in c:
        ev = 0.8 * ev + 0.2 * c["thermal"]

    return np.clip(
        ev * np.clip(dark_n * 4.0, 0.0, 1.0),
        0.0,
        1.0
    )


def feature_stack(c):
    """Build the nine per-pixel features expected by the ML classifier."""

    if "raw" not in c:
        raise ValueError("Cue dictionary is missing raw features.")

    r = c["raw"]

    required = ("dark", "sat", "tex", "spec")

    for key in required:
        if key not in r:
            raise ValueError(f"Missing raw feature: {key}")

        if not isinstance(r[key], np.ndarray) or r[key].ndim != 2:
            raise ValueError(f"Raw feature '{key}' must be a 2D array.")

        if not np.isfinite(r[key]).all():
            raise ValueError(f"Raw feature '{key}' contains invalid values.")

    if any(r[key].shape != r["dark"].shape for key in required):
        raise ValueError("All raw feature maps must have identical dimensions.")

    blur = lambda array, sigma: cv2.GaussianBlur(
        array, (0, 0), sigma
    )

    # IMPORTANT: Preserve the original nine-feature order.
    features = [
        r["dark"],
        r["sat"],
        r["tex"],
        r["spec"],
        blur(r["dark"], 6),
        blur(r["sat"], 6),
        blur(r["tex"], 6),
        np.full_like(r["dark"], np.median(r["tex"])),
        np.full_like(r["dark"], np.median(r["sat"])),
    ]

    return np.dstack(features).astype(np.float32)
