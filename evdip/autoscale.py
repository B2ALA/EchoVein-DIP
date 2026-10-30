"""Automatic wall width/height from the camera.

A single camera cannot know absolute size by itself, so a scale reference of KNOWN size is detected automatically:
an ArUco marker (print make_marker.py output at 100 %, stick it flat on the wall). Its 4 corners give a metric
(cm) homography of the wall plane; the wall corners (auto-detected, or the area the user selects) are mapped
into cm -> width/height are CALCULATED, with a Monte-Carlo uncertainty. No marker -> relative units only."""
import cv2
import numpy as np
from .common import order_corners

ARUCO = cv2.aruco


def _detector():
    p = ARUCO.DetectorParameters(); p.cornerRefinementMethod = ARUCO.CORNER_REFINE_SUBPIX
    return ARUCO.ArucoDetector(ARUCO.getPredefinedDictionary(ARUCO.DICT_4X4_50), p)


def detect_markers(img):
    """All 4x4_50 ArUco markers: list of dict(id, corners TL,TR,BR,BL in pixels, side_px)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    corners, ids, _ = _detector().detectMarkers(g)
    out = []
    for c, i in zip(corners or [], [] if ids is None else ids.ravel()):
        c = c.reshape(4, 2).astype(np.float64)
        side = float(np.mean([np.linalg.norm(c[(k + 1) % 4] - c[k]) for k in range(4)]))
        out.append({"id": int(i), "corners": c, "side_px": side})
    return sorted(out, key=lambda m: -m["side_px"])


def _solve(q_norm, m_norm, L):
    """Wall outline (TL,TR,BR,BL) gives the perspective (-> unit square, well conditioned); the marker (a known LxL
    square on the same plane) gives scale along each wall axis: |e1|^2 = Wu^2 ex^2 + Hu^2 ey^2 = L^2 (same for e2, e1 _|_ e2)."""
    Hu = cv2.getPerspectiveTransform(np.float32(q_norm), np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])).astype(np.float64)
    mu = cv2.perspectiveTransform(np.asarray(m_norm, np.float64).reshape(-1, 1, 2), Hu).reshape(4, 2)
    e1 = ((mu[1] - mu[0]) + (mu[2] - mu[3])) / 2; e2 = ((mu[3] - mu[0]) + (mu[2] - mu[1])) / 2
    A = np.array([[e1[0] ** 2, e1[1] ** 2], [e2[0] ** 2, e2[1] ** 2], [e1[0] * e2[0], e1[1] * e2[1]]])
    bb = np.array([L * L, L * L, 0.0])
    sol, *_ = np.linalg.lstsq(A, bb, rcond=None)
    if (sol <= 0).any():
        return None
    resid = float(np.linalg.norm(A @ sol - bb) / (L * L))
    return float(np.sqrt(sol[0])), float(np.sqrt(sol[1])), resid


def measure_quad(quad_norm, marker_corners_norm, marker_cm, img_w=640, img_h=480, sigma_marker_px=0.3, sigma_quad_px=3.0, n=200, seed=0):
    """Width/height (cm) of the wall or user-selected AREA from its 4 corners + the marker; uncertainty = 2 sigma Monte-Carlo."""
    q = order_corners(np.asarray(quad_norm, np.float64) * [img_w, img_h]) / [img_w, img_h]
    mc = np.asarray(marker_corners_norm, np.float64)
    r = _solve(q, mc, marker_cm)
    if r is None:
        return None
    w, h, resid = r
    rng = np.random.default_rng(seed); ws, hs = [], []
    for _ in range(n):
        o = _solve(q + rng.normal(0, sigma_quad_px, q.shape) / [img_w, img_h], mc + rng.normal(0, sigma_marker_px, mc.shape) / [img_w, img_h], marker_cm)
        if o:
            ws.append(o[0]); hs.append(o[1])
    unc = (2 * float(np.std(ws)), 2 * float(np.std(hs))) if len(ws) > 20 else None
    asp_est = estimate_aspect(q * [img_w, img_h], img_w, img_h)
    warn = []
    if resid > 0.15:
        warn.append("marker and wall outline are not consistent (marker not flat on the wall plane, or selected area is not a rectangle)")
    if abs(np.log((w / h) / asp_est)) > 0.18:
        warn.append("aspect cross-check (assumed camera focal length) disagrees by >18%: check the corners and marker")
    return {"width_cm": w, "height_cm": h, "uncertainty_cm": unc, "consistency_residual": resid, "aspect_crosscheck": float(asp_est), "warnings": warn}


def estimate_aspect(quad_px, img_w, img_h, f_rel=0.85):
    """Width/height ratio of a physical rectangle seen as quad_px (Zhang & He whiteboard method) with an
    ASSUMED focal length f = f_rel * image width (phones ~0.7-0.9, webcams ~0.8-1.0). Approximate."""
    q = order_corners(quad_px).astype(np.float64)
    m1, m2, m4, m3 = [np.array([*p, 1.0]) for p in q]              # TL, TR, BR, BL  (paper naming: m3=BL, m4=BR)
    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)
    f = f_rel * img_w
    Ai = np.linalg.inv(np.array([[f, 0, img_w / 2], [0, f, img_h / 2], [0, 0, 1.0]]))
    n2, n3 = k2 * m2 - m1, k3 * m3 - m1
    a = lambda n: float(n @ Ai.T @ Ai @ n)
    r = a(n2) / a(n3)
    return float(np.sqrt(abs(r)))
