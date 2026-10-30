"""Wall quadrilateral detection, calibration quality, scale check, coordinate maths."""
import cv2
import numpy as np
from .common import order_corners


def rect_to_cm(px, size_px, wall_cm):
    """Rectified-image pixel -> physical cm:  x = px / size * wall   (e.g. 120/600*300 = 60 cm)."""
    return float(px) / float(size_px) * float(wall_cm)


def _angle_dev(q):
    dev = 0.0
    for i in range(4):
        a, b, c = q[i - 1], q[i], q[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cosv = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        dev = max(dev, abs(np.degrees(np.arccos(np.clip(cosv, -1, 1))) - 90))
    return dev


def detect_wall_quad(img):
    """Auto-detect a wall-like quadrilateral (Canny + Otsu contours, 4-vertex convex polygons).
    Returns dict(found, corners_norm TL,TR,BR,BL, score, method). Manual correction is always available."""
    h, w = img.shape[:2]
    s = 640.0 / max(h, w)
    sm = cv2.resize(img, None, fx=s, fy=s) if s < 1 else img.copy()
    H, W = sm.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    med = float(np.median(g))
    edges = cv2.dilate(cv2.Canny(g, 0.66 * med, 1.33 * med), np.ones((3, 3), np.uint8), iterations=2)
    _, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    best = None
    for method, m in (("canny", edges), ("otsu", otsu), ("otsu_inv", 255 - otsu)):
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL if method != "canny" else cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:12]:
            ap = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(ap) != 4 or not cv2.isContourConvex(ap):
                continue
            area = cv2.contourArea(ap) / float(W * H)
            if not 0.12 < area < 0.97:
                continue
            q = order_corners(ap.reshape(4, 2))
            score = area * max(0.0, 1 - _angle_dev(q) / 60.0)
            if best is None or score > best["score"]:
                best = {"found": True, "corners_norm": (q / [W, H]).tolist(), "score": float(score), "method": method}
    return best or {"found": False, "corners_norm": None, "score": 0.0, "method": None}


def calibration_quality(corners_px, size, wall_w_cm, wall_h_cm, scale_err_pct=None):
    q = order_corners(corners_px); pw, ph = size
    warn, pen = [], 0.0
    cr = [float((q[(i + 1) % 4] - q[i])[0] * (q[(i + 2) % 4] - q[(i + 1) % 4])[1] - (q[(i + 1) % 4] - q[i])[1] * (q[(i + 2) % 4] - q[(i + 1) % 4])[0]) for i in range(4)]
    if not (all(c > 0 for c in cr) or all(c < 0 for c in cr)):
        return {"score": 0.0, "level": "invalid", "warnings": ["corners do not form a convex quadrilateral"], "corner_err_px": 99.0}
    area = cv2.contourArea(q.astype(np.float32)) / float(pw * ph)
    if area < 0.15:
        warn.append("wall covers <15% of the image: low resolution on the wall"); pen += 0.25
    if (q[:, 0].min() < 0.01 * pw or q[:, 1].min() < 0.01 * ph or q[:, 0].max() > 0.99 * pw or q[:, 1].max() > 0.99 * ph):
        warn.append("a corner is at the image edge: the entire wall may not be visible"); pen += 0.3
    L = lambda a, b: float(np.linalg.norm(q[a] - q[b]))
    top, bot, lef, rig = L(0, 1), L(3, 2), L(0, 3), L(1, 2)
    fore = max(top / bot, bot / top, lef / rig, rig / lef)
    if fore > 1.8:
        warn.append(f"strong perspective (opposite sides differ {fore:.1f}x): accuracy drops at the far side"); pen += 0.25
    elif fore > 1.3:
        warn.append("moderate perspective: move the camera more square-on if possible"); pen += 0.08
    apparent = 0.5 * (top + bot) / (0.5 * (lef + rig) + 1e-9); entered = wall_w_cm / wall_h_cm
    if abs(np.log(apparent / entered)) > 0.45:
        warn.append("apparent shape differs strongly from the entered width/height (perspective or wrong dimensions)"); pen += 0.2
    if scale_err_pct is not None:
        if abs(scale_err_pct) > 5:
            warn.append(f"reference-object check off by {scale_err_pct:+.1f}%: recalibrate"); pen += 0.3
        elif abs(scale_err_pct) > 2:
            pen += 0.08
    score = float(np.clip(1 - pen, 0, 1))
    return {"score": score, "level": "good" if score >= 0.75 else "fair" if score >= 0.5 else "poor", "warnings": warn,
            "corner_err_px": 3.0 + 4.0 * pen, "scale_err_pct": scale_err_pct, "perspective_ratio": float(fore)}


def scale_check(cal, p1_px, p2_px, known_cm):
    """User taps two points a known distance apart; compare with the homography-measured distance."""
    P = cal.H @ np.array([[p1_px[0], p2_px[0]], [p1_px[1], p2_px[1]], [1, 1]], float)
    P = P[:2] / P[2]
    measured = float(np.linalg.norm(P[:, 0] - P[:, 1]) / cal.px_per_cm)
    return {"measured_cm": measured, "known_cm": float(known_cm), "error_pct": (measured - known_cm) / known_cm * 100.0}


def cm_per_cam_px(cal, mx, my, T=None):
    """Local physical size (cm) of one camera pixel near map point (mx,my) - for uncertainty propagation."""
    p = cal.map_to_image([[mx, my], [mx + 10, my], [mx, my + 10]], T)
    cam_per_map = 0.5 * (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[0])) / 10.0
    return float(1.0 / (cam_per_map * cal.px_per_cm + 1e-9))
