"""Sources, calibration (homography), config."""
import json, os
from dataclasses import dataclass, asdict
import cv2
import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class Source:
    """Phone/webcam/video/image source.
    "0" webcam index (DroidCam/Iriun/Camo), "http://IP:8080/video" (Android IP Webcam),
    "clip.mp4", or "photo.jpg"."""

    def __init__(self, src):
        self.image, self.cap = None, None
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        if isinstance(src, str) and os.path.isfile(src) and src.lower().endswith(IMG_EXT):
            self.image = cv2.imread(src)
            if self.image is None:
                raise IOError(f"Cannot read image {src}")
        else:
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                raise IOError(f"Cannot open source {src!r}")

    def read(self):
        if self.image is not None:
            return self.image.copy()
        for _ in range(2):          # drop stale buffered frames
            self.cap.grab()
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        if self.cap is not None:
            self.cap.release()


def order_corners(pts):
    pts = np.array(pts, np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)


def build_calib(corners, wall_w_m, wall_h_m, px_per_m=200, image_size=None):
    corners = order_corners(corners)
    W, H = int(round(wall_w_m * px_per_m)), int(round(wall_h_m * px_per_m))
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
    Hm = cv2.getPerspectiveTransform(corners, dst)
    return {"corners": corners.tolist(), "wall_w_m": wall_w_m, "wall_h_m": wall_h_m,
            "px_per_m": px_per_m, "map_size": [W, H], "H": Hm.tolist(), "image_size": image_size}


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


class Calib:
    """Wall homography. H maps camera pixels -> rectified wall map (1 px = 100/px_per_m cm)."""

    def __init__(self, c):
        if isinstance(c, str):
            c = json.load(open(c))
        self.c = c
        self.H = np.array(c["H"], np.float64)
        self.corners = np.array(c["corners"], np.float32)
        self.W, self.Hpx = c["map_size"]
        self.ppm = c["px_per_m"]
        self.px_per_cm = self.ppm / 100.0
        self.wall_w_cm = c["wall_w_m"] * 100
        self.wall_h_cm = c["wall_h_m"] * 100
        self.wall_area_cm2 = self.wall_w_cm * self.wall_h_cm
        self.Heff = self.H.copy()          # may be updated by self-recalibration

    def rectify(self, frame, T=None):
        Hm = self.H if T is None else self.H @ T
        return cv2.warpPerspective(frame, Hm, (self.W, self.Hpx), flags=cv2.INTER_LINEAR)

    def map_to_image(self, pts, T=None):
        Hm = self.H if T is None else self.H @ T
        p = np.array(pts, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, np.linalg.inv(Hm)).reshape(-1, 2)

    def wall_mask_cam(self, shape):
        m = np.zeros(shape[:2], np.uint8)
        cv2.fillPoly(m, [self.corners.astype(np.int32)], 255)
        return m


@dataclass
class Config:
    fusion: str = "weighted"        # weighted | ml | dark_only
    thr: float = 0.55               # score threshold (weighted/ml scale 0..1)
    min_area_cm2: float = 40.0
    persist: int = 2                # analyses before WATCH -> SUSPECTED
    use_library: bool = True
    use_illum: bool = True
    use_multicue: bool = True
    use_temporal: bool = True
    use_occlusion: bool = True
    use_selfcal: bool = True
    use_classifier: bool = True
    classifier: str = "ml"          # ml | rules
    privacy: bool = True
    auto_learn_baseline: bool = False
    thr_scale: float = 1.0          # tuned by human feedback
    max_state: str = "VERIFIED"     # live scanner uses SUSPECTED (hours-scale behaviour logic needs long observation)

    def to_dict(self):
        return asdict(self)
