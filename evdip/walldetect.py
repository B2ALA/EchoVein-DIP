
"""Wall quadrilateral detection, calibration quality, scale check and coordinate maths."""

import cv2
import numpy as np
from .common import order_corners


def _valid_number(value, name):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _validate_image(img):
    if not isinstance(img, np.ndarray):
        raise TypeError("Image must be a NumPy array.")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Image must be a BGR colour image.")
    if img.shape[0] < 16 or img.shape[1] < 16:
        raise ValueError("Image must be at least 16x16 pixels.")
    if not np.isfinite(img).all():
        raise ValueError("Image contains invalid values.")
    return img.astype(np.uint8) if img.dtype != np.uint8 else img


def rect_to_cm(px, size_px, wall_cm):
    """Convert rectified-image pixels into physical centimetres."""
    px = _valid_number(px, "px")
    size_px = _valid_number(size_px, "size_px")
    wall_cm = _valid_number(wall_cm, "wall_cm")

    if size_px <= 0 or wall_cm <= 0:
        raise ValueError("Pixel size and wall dimension must be positive.")

    return px / size_px * wall_cm


def _angle_dev(q):
    """Maximum deviation of quadrilateral corners from 90 degrees."""
    dev = 0.0

    for i in range(4):
        a = q[i - 1]
        b = q[i]
        c = q[(i + 1) % 4]

        v1 = a - b
        v2 = c - b

        denom = np.linalg.norm(v1) * np.linalg.norm(v2)

        if denom < 1e-9:
            return 90.0

        cosv = np.dot(v1, v2) / denom
        angle = np.degrees(np.arccos(np.clip(cosv, -1, 1)))

        dev = max(dev, abs(angle - 90.0))

    return float(dev)


def detect_wall_quad(img):
    """Detect a wall-like quadrilateral.

    Returns:
        found, corners_norm, score, method

    Corner order: top-left, top-right, bottom-right, bottom-left.
    Manual correction should remain available.
    """
    img = _validate_image(img)

    h, w = img.shape[:2]

    scale = min(1.0, 640.0 / max(h, w))

    sm = (
        cv2.resize(img, None, fx=scale, fy=scale)
        if scale < 1.0 else img.copy()
    )

    H, W = sm.shape[:2]

    gray = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    median = float(np.median(gray))

    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))

    edges = cv2.Canny(gray, lower, upper)

    edges = cv2.dilate(
        edges,
        np.ones((3, 3), np.uint8),
        iterations=2
    )

    _, otsu = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((7, 7), np.uint8)

    otsu = cv2.morphologyEx(
        otsu, cv2.MORPH_OPEN, kernel
    )

    candidates = (
        ("canny", edges),
        ("otsu", otsu),
        ("otsu_inv", 255 - otsu),
    )

    best = None

    for method, mask in candidates:

        retrieval = (
            cv2.RETR_LIST
            if method == "canny"
            else cv2.RETR_EXTERNAL
        )

        contours, _ = cv2.findContours(
            mask, retrieval, cv2.CHAIN_APPROX_SIMPLE
        )

        contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True
        )[:12]

        for contour in contours:

            perimeter = cv2.arcLength(contour, True)

            if perimeter <= 0:
                continue

            approx = cv2.approxPolyDP(
                contour,
                0.02 * perimeter,
                True
            )

            if len(approx) != 4:
                continue

            if not cv2.isContourConvex(approx):
                continue

            area = cv2.contourArea(approx) / float(W * H)

            if not 0.12 < area < 0.97:
                continue

            corners = order_corners(
                approx.reshape(4, 2)
            ).astype(np.float32)

            angle_error = _angle_dev(corners)

            score = area * max(
                0.0,
                1.0 - angle_error / 60.0
            )

            if best is None or score > best["score"]:
                best = {
                    "found": True,
                    "corners_norm": (
                        corners / np.array([W, H], dtype=np.float32)
                    ).tolist(),
                    "score": float(score),
                    "method": method,
                }

    return best or {
        "found": False,
        "corners_norm": None,
        "score": 0.0,
        "method": None,
    }


def calibration_quality(
    corners_px,
    size,
    wall_w_cm,
    wall_h_cm,
    scale_err_pct=None
):
    """Evaluate wall calibration and return quality warnings."""

    pw, ph = size

    if pw <= 0 or ph <= 0:
        raise ValueError("Image dimensions must be positive.")

    wall_w_cm = _valid_number(wall_w_cm, "wall_w_cm")
    wall_h_cm = _valid_number(wall_h_cm, "wall_h_cm")

    if wall_w_cm <= 0 or wall_h_cm <= 0:
        raise ValueError("Wall dimensions must be positive.")

    q = np.asarray(corners_px, dtype=np.float32)

    if q.shape != (4, 2) or not np.isfinite(q).all():
        return {
            "score": 0.0,
            "level": "invalid",
            "warnings": ["four valid wall corners are required"],
            "corner_err_px": 99.0,
        }

    q = order_corners(q)

    # Check convexity and non-zero area.
    contour = q.reshape((-1, 1, 2))

    if not cv2.isContourConvex(contour):
        return {
            "score": 0.0,
            "level": "invalid",
            "warnings": [
                "corners do not form a convex quadrilateral"
            ],
            "corner_err_px": 99.0,
        }

    area_px = cv2.contourArea(contour)

    if area_px < 1.0:
        return {
            "score": 0.0,
            "level": "invalid",
            "warnings": ["wall area is too small"],
            "corner_err_px": 99.0,
        }

    area = area_px / float(pw * ph)

    warnings = []
    penalty = 0.0

    if area < 0.15:
        warnings.append(
            "wall covers <15% of the image: low resolution on the wall"
        )
        penalty += 0.25

    if (
        q[:, 0].min() < 0.01 * pw
        or q[:, 1].min() < 0.01 * ph
        or q[:, 0].max() > 0.99 * pw
        or q[:, 1].max() > 0.99 * ph
    ):
        warnings.append(
            "a corner is at the image edge: the entire wall may not be visible"
        )
        penalty += 0.3

    def length(a, b):
        return float(np.linalg.norm(q[a] - q[b]))

    top = length(0, 1)
    bottom = length(3, 2)
    left = length(0, 3)
    right = length(1, 2)

    if min(top, bottom, left, right) < 1e-6:
        return {
            "score": 0.0,
            "level": "invalid",
            "warnings": ["one or more wall edges have zero length"],
            "corner_err_px": 99.0,
        }

    perspective = max(
        top / bottom,
        bottom / top,
        left / right,
        right / left
    )

    if perspective > 1.8:
        warnings.append(
            f"strong perspective (opposite sides differ {perspective:.1f}x): accuracy drops at the far side"
        )
        penalty += 0.25

    elif perspective > 1.3:
        warnings.append(
            "moderate perspective: move the camera more square-on if possible"
        )
        penalty += 0.08

    apparent = (
        0.5 * (top + bottom)
        / (0.5 * (left + right) + 1e-9)
    )

    entered = wall_w_cm / wall_h_cm

    if abs(np.log(apparent / entered)) > 0.45:
        warnings.append(
            "apparent shape differs strongly from the entered width/height (perspective or wrong dimensions)"
        )
        penalty += 0.2

    if scale_err_pct is not None:
        scale_err_pct = _valid_number(
            scale_err_pct, "scale_err_pct"
        )

        if abs(scale_err_pct) > 5:
            warnings.append(
                f"reference-object check off by {scale_err_pct:+.1f}%: recalibrate"
            )
            penalty += 0.3

        elif abs(scale_err_pct) > 2:
            penalty += 0.08

    score = float(np.clip(1.0 - penalty, 0.0, 1.0))

    level = (
        "good" if score >= 0.75
        else "fair" if score >= 0.5
        else "poor"
    )

    return {
        "score": score,
        "level": level,
        "warnings": warnings,
        "corner_err_px": 3.0 + 4.0 * penalty,
        "scale_err_pct": scale_err_pct,
        "perspective_ratio": float(perspective),
    }


def scale_check(cal, p1_px, p2_px, known_cm):
    """Compare a known physical distance with its calibrated distance."""

    known_cm = _valid_number(known_cm, "known_cm")

    if known_cm <= 0:
        raise ValueError("Known distance must be positive.")

    p1 = np.asarray(p1_px, dtype=np.float64).reshape(2)
    p2 = np.asarray(p2_px, dtype=np.float64).reshape(2)

    if not np.isfinite(p1).all() or not np.isfinite(p2).all():
        raise ValueError("Scale-check points must be finite.")

    if not np.isfinite(cal.H).all():
        raise ValueError("Calibration homography is invalid.")

    if cal.px_per_cm <= 0:
        raise ValueError("Calibration scale must be positive.")

    points = np.array([
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        [1.0, 1.0],
    ], dtype=np.float64)

    transformed = cal.H @ points

    if np.any(np.abs(transformed[2]) < 1e-9):
        raise ValueError("Scale-check points cannot be projected.")

    transformed = transformed[:2] / transformed[2]

    measured = float(
        np.linalg.norm(
            transformed[:, 0] - transformed[:, 1]
        ) / cal.px_per_cm
    )

    return {
        "measured_cm": measured,
        "known_cm": known_cm,
        "error_pct": (
            (measured - known_cm) / known_cm * 100.0
        ),
    }


def cm_per_cam_px(cal, mx, my, T=None):
    """Estimate local physical size of one camera pixel."""

    mx = _valid_number(mx, "mx")
    my = _valid_number(my, "my")

    points = cal.map_to_image(
        [
            [mx, my],
            [mx + 10, my],
            [mx, my + 10],
        ],
        T
    )

    points = np.asarray(points, dtype=np.float64)

    if points.shape != (3, 2) or not np.isfinite(points).all():
        raise ValueError("Invalid camera-coordinate mapping.")

    cam_per_map = (
        np.linalg.norm(points[1] - points[0])
        + np.linalg.norm(points[2] - points[0])
    ) / 20.0

    scale = cam_per_map * cal.px_per_cm

    if not np.isfinite(scale) or scale <= 1e-9:
        raise ValueError("Invalid local calibration scale.")

    return float(1.0 / scale)
