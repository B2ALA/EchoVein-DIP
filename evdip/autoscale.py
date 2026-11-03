
"""Reliable marker-based automatic wall calibration for EchoVein-DIP.

A known-size ArUco marker must be flat on the same plane as the wall.
Without a known physical reference, absolute dimensions cannot be calculated.

Public API retained:
    detect_markers(img)
    measure_quad(...)
    estimate_aspect(...)
"""

import cv2
import numpy as np

from .common import order_corners

ARUCO = cv2.aruco

MARKER_DICTIONARY = ARUCO.DICT_4X4_50
MIN_MARKER_SIDE_PX = 25.0
MIN_QUAD_AREA_RATIO = 0.01
MAX_CONSISTENCY_RESIDUAL = 0.15
MAX_ASPECT_DISAGREEMENT = 0.18


def _detector():
    """Create an ArUco detector with subpixel corner refinement."""
    params = ARUCO.DetectorParameters()
    params.cornerRefinementMethod = ARUCO.CORNER_REFINE_SUBPIX

    dictionary = ARUCO.getPredefinedDictionary(MARKER_DICTIONARY)
    return ARUCO.ArucoDetector(dictionary, params)


def _valid_image(img):
    """Check whether an image is suitable for marker detection."""
    return (
        isinstance(img, np.ndarray)
        and img.ndim in (2, 3)
        and img.size > 0
        and img.shape[0] >= 20
        and img.shape[1] >= 20
    )


def _valid_quad(points):
    """Check four finite, distinct, non-degenerate quadrilateral points."""
    try:
        q = np.asarray(points, dtype=np.float64).reshape(4, 2)
    except (TypeError, ValueError):
        return False

    if not np.isfinite(q).all():
        return False

    if len(np.unique(np.round(q, 8), axis=0)) != 4:
        return False

    area = abs(float(cv2.contourArea(q.astype(np.float32))))
    return area > 1e-8


def detect_markers(img):
    """Return detected markers sorted by apparent pixel side length.

    Each result contains:
        id: ArUco marker ID
        corners: four image points
        side_px: mean marker side length in pixels
    """
    if not _valid_image(img):
        return []

    if img.ndim == 2:
        gray = img
    elif img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        return []

    try:
        corners, ids, _ = _detector().detectMarkers(gray)
    except cv2.error:
        return []

    if ids is None:
        return []

    output = []

    for corner_data, marker_id in zip(corners, ids.ravel()):
        points = np.asarray(corner_data, dtype=np.float64).reshape(4, 2)

        if not np.isfinite(points).all():
            continue

        side_lengths = [
            float(np.linalg.norm(points[(i + 1) % 4] - points[i]))
            for i in range(4)
        ]

        if not np.isfinite(side_lengths).all():
            continue

        side_px = float(np.mean(side_lengths))

        if side_px <= 0:
            continue

        output.append({
            "id": int(marker_id),
            "corners": points,
            "side_px": side_px,
        })

    return sorted(output, key=lambda item: -item["side_px"])


def _solve(q_norm, marker_norm, marker_cm):
    """Calculate wall-plane width and height from a marker of known size.

    Returns:
        (width_cm, height_cm, consistency_residual)
        or None if the geometry is invalid.
    """
    if not _valid_quad(q_norm) or not _valid_quad(marker_norm):
        return None

    if not np.isfinite(marker_cm) or marker_cm <= 0:
        return None

    q = np.asarray(q_norm, dtype=np.float64).reshape(4, 2)
    marker = np.asarray(marker_norm, dtype=np.float64).reshape(4, 2)

    target = np.array(
        [[0, 0], [1, 0], [1, 1], [0, 1]],
        dtype=np.float32,
    )

    try:
        homography = cv2.getPerspectiveTransform(
            q.astype(np.float32),
            target,
        ).astype(np.float64)
    except cv2.error:
        return None

    if not np.isfinite(homography).all():
        return None

    if abs(float(np.linalg.det(homography))) < 1e-12:
        return None

    try:
        mapped = cv2.perspectiveTransform(
            marker.reshape(-1, 1, 2),
            homography,
        ).reshape(4, 2)
    except cv2.error:
        return None

    if not np.isfinite(mapped).all():
        return None

    # Average opposite edges to reduce corner-noise sensitivity.
    e1 = ((mapped[1] - mapped[0]) + (mapped[2] - mapped[3])) / 2.0
    e2 = ((mapped[3] - mapped[0]) + (mapped[2] - mapped[1])) / 2.0

    matrix = np.array([
        [e1[0] ** 2, e1[1] ** 2],
        [e2[0] ** 2, e2[1] ** 2],
        [e1[0] * e2[0], e1[1] * e2[1]],
    ], dtype=np.float64)

    target_values = np.array([
        marker_cm ** 2,
        marker_cm ** 2,
        0.0,
    ], dtype=np.float64)

    try:
        solution, _, _, _ = np.linalg.lstsq(
            matrix,
            target_values,
            rcond=None,
        )
    except np.linalg.LinAlgError:
        return None

    if not np.isfinite(solution).all() or np.any(solution <= 0):
        return None

    width_cm = float(np.sqrt(solution[0]))
    height_cm = float(np.sqrt(solution[1]))

    if not np.isfinite([width_cm, height_cm]).all():
        return None

    if width_cm <= 0 or height_cm <= 0:
        return None

    residual = float(
        np.linalg.norm(matrix @ solution - target_values)
        / (marker_cm ** 2)
    )

    if not np.isfinite(residual):
        return None

    return width_cm, height_cm, residual


def measure_quad(
    quad_norm,
    marker_corners_norm,
    marker_cm,
    img_w=640,
    img_h=480,
    sigma_marker_px=0.3,
    sigma_quad_px=3.0,
    n=200,
    seed=0,
):
    """Calculate physical wall dimensions and estimated uncertainty.

    Coordinates must be normalized to the image dimensions.
    Marker and wall must lie on the same plane.

    Uncertainty is a Monte-Carlo estimate based on assumed pixel errors.
    It is not a complete physical measurement uncertainty.
    """
    try:
        img_w = int(img_w)
        img_h = int(img_h)
        marker_cm = float(marker_cm)
        sigma_marker_px = float(sigma_marker_px)
        sigma_quad_px = float(sigma_quad_px)
        n = int(n)
    except (TypeError, ValueError, OverflowError):
        return None

    if img_w < 20 or img_h < 20:
        return None

    if not np.isfinite(marker_cm) or marker_cm <= 0:
        return None

    if (
        not np.isfinite(sigma_marker_px)
        or not np.isfinite(sigma_quad_px)
        or sigma_marker_px < 0
        or sigma_quad_px < 0
        or n < 0
    ):
        return None

    try:
        q_input = np.asarray(quad_norm, dtype=np.float64).reshape(4, 2)
        marker_input = np.asarray(
            marker_corners_norm,
            dtype=np.float64,
        ).reshape(4, 2)
    except (TypeError, ValueError):
        return None

    if not _valid_quad(q_input) or not _valid_quad(marker_input):
        return None

    if (
        np.any(q_input < -0.01)
        or np.any(q_input > 1.01)
        or np.any(marker_input < -0.01)
        or np.any(marker_input > 1.01)
    ):
        return None

    # Preserve the project's existing corner-ordering convention.
    try:
        q_px = order_corners(
            q_input * np.array([img_w, img_h], dtype=np.float64)
        )
        q = q_px / np.array([img_w, img_h], dtype=np.float64)
    except (ValueError, cv2.error):
        return None

    if not _valid_quad(q):
        return None

    area_ratio = abs(
        float(cv2.contourArea(q_px.astype(np.float32)))
    ) / float(img_w * img_h)

    if area_ratio < MIN_QUAD_AREA_RATIO:
        return None

    result = _solve(q, marker_input, marker_cm)

    if result is None:
        return None

    width_cm, height_cm, residual = result

    if not (0.1 <= width_cm <= 100000):
        return None

    if not (0.1 <= height_cm <= 100000):
        return None

    # Monte-Carlo uncertainty estimate.
    uncertainty = None

    if n >= 21 and (sigma_marker_px > 0 or sigma_quad_px > 0):
        rng = np.random.default_rng(seed)
        widths = []
        heights = []

        image_scale = np.array([img_w, img_h], dtype=np.float64)

        for _ in range(n):
            noisy_q = q + (
                rng.normal(0, sigma_quad_px, q.shape) / image_scale
            )

            noisy_marker = marker_input + (
                rng.normal(0, sigma_marker_px, marker_input.shape)
                / image_scale
            )

            sample = _solve(noisy_q, noisy_marker, marker_cm)

            if sample is not None:
                sw, sh, _ = sample

                if (
                    np.isfinite([sw, sh]).all()
                    and 0 < sw < 100000
                    and 0 < sh < 100000
                ):
                    widths.append(sw)
                    heights.append(sh)

        # Do not report uncertainty from too few successful samples.
        if len(widths) >= max(21, int(0.8 * n)):
            uncertainty = (
                2.0 * float(np.std(widths, ddof=1)),
                2.0 * float(np.std(heights, ddof=1)),
            )

    warnings = []

    if residual > MAX_CONSISTENCY_RESIDUAL:
        warnings.append(
            "Marker and wall outline are inconsistent. "
            "Check marker placement, wall corners and coplanarity."
        )

    if min(
        np.linalg.norm(marker_input[(i + 1) % 4] - marker_input[i])
        * min(img_w, img_h)
        for i in range(4)
    ) < MIN_MARKER_SIDE_PX:
        warnings.append(
            "Marker is small in the image; improve camera distance "
            "or use a larger printed marker."
        )

    # Approximate aspect-ratio cross-check only.
    aspect_estimate = None

    try:
        aspect_estimate = estimate_aspect(
            q * np.array([img_w, img_h]),
            img_w,
            img_h,
        )

        if (
            np.isfinite(aspect_estimate)
            and aspect_estimate > 0
            and abs(np.log((width_cm / height_cm) / aspect_estimate))
            > MAX_ASPECT_DISAGREEMENT
        ):
            warnings.append(
                "Approximate aspect-ratio cross-check disagrees. "
                "Review wall corners and camera assumptions."
            )
    except (ValueError, ArithmeticError, np.linalg.LinAlgError, cv2.error):
        warnings.append(
            "Aspect-ratio cross-check unavailable; review wall corners."
        )

    return {
        "width_cm": width_cm,
        "height_cm": height_cm,
        "uncertainty_cm": uncertainty,
        "consistency_residual": residual,
        "aspect_crosscheck": (
            float(aspect_estimate)
            if aspect_estimate is not None
            and np.isfinite(aspect_estimate)
            else None
        ),
        "warnings": warnings,
    }


def estimate_aspect(quad_px, img_w, img_h, f_rel=0.85):
    """Estimate rectangle aspect ratio using an assumed focal length.

    This is an approximate cross-check only. It does not provide
    physical scale and must not be treated as marker calibration.
    """
    try:
        img_w = float(img_w)
        img_h = float(img_h)
        f_rel = float(f_rel)
        q_input = np.asarray(quad_px, dtype=np.float64).reshape(4, 2)
    except (TypeError, ValueError):
        return float("nan")

    if (
        img_w <= 0
        or img_h <= 0
        or f_rel <= 0
        or not np.isfinite(q_input).all()
        or not _valid_quad(q_input)
    ):
        return float("nan")

    try:
        q = order_corners(q_input).astype(np.float64)

        m1, m2, m4, m3 = [
            np.array([p[0], p[1], 1.0], dtype=np.float64)
            for p in q
        ]

        denominator_2 = float(np.dot(np.cross(m2, m4), m3))
        denominator_3 = float(np.dot(np.cross(m3, m4), m2))

        if abs(denominator_2) < 1e-12 or abs(denominator_3) < 1e-12:
            return float("nan")

        k2 = float(
            np.dot(np.cross(m1, m4), m3) / denominator_2
        )
        k3 = float(
            np.dot(np.cross(m1, m4), m2) / denominator_3
        )

        focal_length = f_rel * img_w

        intrinsic_inverse = np.linalg.inv(
            np.array([
                [focal_length, 0, img_w / 2],
                [0, focal_length, img_h / 2],
                [0, 0, 1.0],
            ], dtype=np.float64)
        )

        n2 = k2 * m2 - m1
        n3 = k3 * m3 - m1

        def quadratic_value(vector):
            transformed = intrinsic_inverse @ vector
            return float(transformed @ transformed)

        a2 = quadratic_value(n2)
        a3 = quadratic_value(n3)

        if a3 <= 1e-12 or not np.isfinite([a2, a3]).all():
            return float("nan")

        ratio = a2 / a3

        if ratio <= 0 or not np.isfinite(ratio):
            return float("nan")

        return float(np.sqrt(ratio))

    except (
        ValueError,
        ArithmeticError,
        np.linalg.LinAlgError,
        cv2.error,
    ):
        return float("nan")
