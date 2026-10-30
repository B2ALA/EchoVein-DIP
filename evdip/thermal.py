"""Optional thermal module. Works with a radiometric array (e.g. MLX90640 24x32 degC) or a
thermal dataset (.npy/.csv of temperatures). Registered to the wall map with its own corners."""
import cv2
import numpy as np
from .evidence import Evidence


def load_temp(path):
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    return np.loadtxt(path, delimiter=",").astype(np.float32)


def register(temp, corners_in_temp_px, map_size):
    """Warp a temperature array onto the rectified wall map. corners order TL,TR,BR,BL."""
    W, H = map_size
    dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    M = cv2.getPerspectiveTransform(np.float32(corners_in_temp_px), dst)
    return cv2.warpPerspective(temp, M, (W, H), flags=cv2.INTER_LINEAR)


def thermal_cue(temp_map, delta_c=1.2):
    """Cool anomaly (evaporation / cold leaking water) vs local wall median -> 0..1."""
    k = max(15, (min(temp_map.shape) // 3) | 1)
    bg = cv2.medianBlur(cv2.GaussianBlur(temp_map, (0, 0), 3).astype(np.float32), 5)
    bg = cv2.GaussianBlur(bg, (0, 0), k)
    cue = np.clip((bg - cv2.GaussianBlur(temp_map, (0, 0), 3)) / delta_c, 0, 1)
    return cue


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
