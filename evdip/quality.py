"""Image-quality assessment (blur, exposure, glare, resolution)."""
import cv2
import numpy as np


def assess_quality(img, min_w=320):
    h, w = img.shape[:2]
    s = 640.0 / max(w, 1)
    g = cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s) if w > 640 else img, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(g, cv2.CV_64F).var())
    mean, contrast = float(g.mean()), float(g.std())
    glare_m = cv2.morphologyEx((g >= 245).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    glare = float(glare_m.mean()); dark_clip = float((g <= 5).mean())
    flags, severe = [], False
    if w < min_w:
        flags.append(f"resolution too low ({w}px wide)"); severe = True
    if blur < 8:
        flags.append("image very blurry / out of focus (or featureless)"); severe = True
    elif blur < 20:
        flags.append("low detail: possible blur or very smooth wall")
    if mean < 35:
        flags.append("underexposed (too dark)"); severe = True
    elif mean > 225:
        flags.append("overexposed"); severe = True
    if glare > 0.08:
        flags.append(f"glare/reflection covers {glare*100:.0f}% of the image"); severe = True
    elif glare > 0.02:
        flags.append("some glare present")
    if dark_clip > 0.3:
        flags.append("large black areas (lens blocked?)"); severe = True
    score = 1.0
    score -= 0.5 * (blur < 8) + 0.2 * (blur < 20) + 0.4 * (mean < 35 or mean > 225) + min(0.4, glare * 4) + 0.4 * (w < min_w)
    return {"score": float(np.clip(score, 0, 1)), "ok": not severe, "flags": flags,
            "metrics": {"sharpness": blur, "mean_brightness": mean, "contrast": contrast, "glare_fraction": glare, "width_px": w, "height_px": h}}
