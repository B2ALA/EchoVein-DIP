"""Appearance statistics for references that are NOT pixel-aligned (other walls, known-damp samples).
Never used for localisation - only to report how similar the current wall's colour/texture statistics are."""
import cv2
import numpy as np


def _lbp(g):
    g = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA)
    c = g[1:-1, 1:-1]; code = np.zeros_like(c, np.uint8)
    for i, (dy, dx) in enumerate([(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]):
        code |= ((g[1 + dy:g.shape[0] - 1 + dy, 1 + dx:g.shape[1] - 1 + dx] >= c).astype(np.uint8) << i)
    h = np.bincount(code.ravel(), minlength=256).astype(np.float32)
    return h / (h.sum() + 1e-9)


def descriptor(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hist = np.concatenate([cv2.calcHist([lab], [i], None, [16], [0, 256]).ravel() for i in range(3)]).astype(np.float32)
    hist /= hist.sum() + 1e-9
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {"hist": hist, "lbp": _lbp(g), "mean_L": float(lab[:, :, 0].mean()), "tex": float(cv2.Laplacian(g, cv2.CV_32F).var())}


def compare(a, b):
    col = float(cv2.compareHist(a["hist"], b["hist"], cv2.HISTCMP_BHATTACHARYYA))
    chi = float(0.5 * np.sum((a["lbp"] - b["lbp"]) ** 2 / (a["lbp"] + b["lbp"] + 1e-9)))
    return {"colour_distance": col, "texture_distance": chi, "brightness_ratio": a["mean_L"] / (b["mean_L"] + 1e-6),
            "texture_ratio": a["tex"] / (b["tex"] + 1e-6)}
