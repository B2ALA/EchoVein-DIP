"""Physical-ish wall simulator used for (a) demo without hardware, (b) ML training labels,
(c) validation with ground truth. Times are in hours."""
import cv2
import numpy as np

KINDS = ["leak", "splash", "condensation", "shadow", "light_switch", "person", "camera_nudge", "dry_drift"]
CAM = (640, 480)


class WallSim:
    def __init__(self, seed=0, w_m=2.0, h_m=1.5, ppm=200, quad=None):
        r = np.random.default_rng(seed)
        self.rng, self.ppm = r, ppm
        self.W, self.H = int(w_m * ppm), int(h_m * ppm)
        self.w_cm, self.h_cm = w_m * 100, h_m * 100
        tint = np.array([140, 178, 215], np.float32) + r.normal(0, 6, 3)        # BGR beige
        n1 = cv2.GaussianBlur(r.normal(0, 1, (self.H, self.W)).astype(np.float32), (0, 0), 1.0) * 55
        n2 = cv2.GaussianBlur(r.normal(0, 1, (self.H, self.W)).astype(np.float32), (0, 0), 7) * 70
        base = tint[None, None, :] + (n1 + n2)[..., None]
        sheen = np.zeros((self.H, self.W), np.float32)
        for _ in range(35):
            cv2.circle(sheen, (int(r.integers(0, self.W)), int(r.integers(0, self.H))), int(r.integers(2, 5)), 1.0, -1)
        sheen = cv2.GaussianBlur(sheen, (0, 0), 2.0) * 160
        self.base, self.sheen = base.astype(np.float32), sheen
        s = CAM[0] / 640.0
        q = np.array(quad if quad is not None else [[90, 70], [560, 50], [590, 430], [60, 410]], np.float32) * s
        q += r.normal(0, 6, q.shape).astype(np.float32)
        self.quad = q
        bg = np.full((CAM[1], CAM[0], 3), (70, 95, 80), np.float32)
        bg += cv2.GaussianBlur(r.normal(0, 1, (CAM[1], CAM[0], 3)).astype(np.float32), (0, 0), 1.2) * 25
        for _ in range(40):          # static clutter (features for ORB self-calibration)
            p1 = (int(r.integers(0, CAM[0])), int(r.integers(0, CAM[1])))
            p2 = (int(r.integers(0, CAM[0])), int(r.integers(0, CAM[1])))
            cv2.rectangle(bg, p1, p2, [float(v) for v in r.integers(30, 220, 3)], int(r.integers(1, 3)))
        self.bg = bg
        src = np.float32([[0, 0], [self.W, 0], [self.W, self.H], [0, self.H]])
        self.M = cv2.getPerspectiveTransform(src, self.quad)
        self.mask_cam = cv2.warpPerspective(np.full((self.H, self.W), 255, np.uint8), self.M, CAM)
        yy, xx = np.mgrid[0:self.H, 0:self.W]
        self.xx, self.yy = xx.astype(np.float32), yy.astype(np.float32)

    # ---- appearance model -------------------------------------------------
    def render_map(self, wet=None, shadow=None, gain=1.0, grad=0.0, noise=2.5, marker=None):
        m = np.zeros((self.H, self.W), np.float32) if wet is None else wet
        base = self.base + self.sheen[..., None] * (1 - m[..., None])        # sheen lost when wet
        if marker is not None:                      # ArUco scale marker stuck on the wall: (cx_cm, cy_cm, size_cm, id)
            cxm, cym, szm, mid = marker
            n = max(int(szm * self.ppm / 100), 12)
            mk = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), mid, n, borderBits=1)
            pad = max(n // 6, 3); mk = cv2.copyMakeBorder(mk, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=235)
            x0, y0 = int(cxm * self.ppm / 100 - mk.shape[1] / 2), int(cym * self.ppm / 100 - mk.shape[0] / 2)
            base = base.copy(); base[y0:y0 + mk.shape[0], x0:x0 + mk.shape[1]] = mk[..., None].astype(np.float32)
        smooth = cv2.GaussianBlur(base, (0, 0), 2.2)
        img = base * (1 - 0.75 * m[..., None]) + smooth * 0.75 * m[..., None]   # texture blur
        gray = img.mean(2, keepdims=True)
        img = gray + (img - gray) * (1 + 0.35 * m[..., None])                 # colour deepens
        img = img * (1 - 0.26 * m[..., None])                                 # darkening
        if shadow is not None:
            img = img * (1 - 0.32 * shadow[..., None])
        light = gain * (1 + grad * (self.xx / self.W - 0.5))
        img = img * light[..., None]
        img = img + self.rng.normal(0, noise, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    def to_camera(self, img_map, person=None, nudge=None, bg_gain=1.0):
        w = cv2.warpPerspective(img_map, self.M, CAM)
        frame = np.where(self.mask_cam[..., None] > 0, w, np.clip(self.bg * bg_gain, 0, 255).astype(np.uint8))
        if person is not None:
            x, y = int(person[0]), int(person[1])
            cv2.ellipse(frame, (x, y - 95), (22, 28), 0, 0, 360, (90, 110, 160), -1)
            cv2.rectangle(frame, (x - 35, y - 65), (x + 35, y + 120), (60, 50, 40), -1)
        if nudge is not None:
            dx, dy, ang = nudge
            A = cv2.getRotationMatrix2D((CAM[0] / 2, CAM[1] / 2), ang, 1.0)
            A[:, 2] += (dx, dy)
            frame = cv2.warpAffine(frame, A, CAM, borderMode=cv2.BORDER_REPLICATE)
        return frame

    def calib_dict(self, px_per_m=None):
        from .common import build_calib
        return build_calib(self.quad, self.w_cm / 100, self.h_cm / 100, px_per_m or self.ppm, list(CAM))

    # ---- event fields (map space, intensity 0..1) ---------------------------
    def blob(self, cx_cm, cy_cm, r_cm, aspect=1.0, drift_cm=0.0, edge=0.25):
        cx, cy = cx_cm * self.ppm / 100, (cy_cm + drift_cm) * self.ppm / 100
        r = max(r_cm * self.ppm / 100, 1.0)
        d = np.hypot((self.xx - cx), (self.yy - cy) / aspect) / r
        return np.clip((1.0 - d) / edge, 0, 1).astype(np.float32)


def dry_baselines(sim):
    """Two dry references: lights normal, and dimmer with a lighting gradient."""
    return [("day", sim.to_camera(sim.render_map(gain=1.0, grad=0.0, noise=1.0))),
            ("dim", sim.to_camera(sim.render_map(gain=0.72, grad=0.35, noise=1.0)))]


def simulate(kind, seed=0, hours=8.0, dt_min=15, n_walls=1):
    """Returns dict: times, walls=[{sim, frames, gt_masks, gt_source, baselines, label}]"""
    r = np.random.default_rng(1000 + seed)
    times = np.arange(0.0, hours + 1e-6, dt_min / 60.0)
    walls = []
    t0 = float(r.uniform(1.0, 2.0))
    for w in range(n_walls):
        sim = WallSim(seed=seed * 7 + w)
        sx, sy = float(r.uniform(35, sim.w_cm - 35)), float(r.uniform(25, sim.h_cm - 55))
        rate = float(r.uniform(4.0, 7.0))
        frames, masks = [], []
        label = {"leak": "leak", "splash": "splash", "condensation": "condensation"}.get(kind, "none")
        affected = (kind == "condensation") or w == 0
        sh_dir = r.uniform(0, 2 * np.pi)
        for t in times:
            wet = None; shadow = None; gain = 1.0; grad = 0.0; person = None; nudge = None
            if kind == "leak" and affected:
                a = np.clip((t - t0) / 0.4, 0, 1)
                rad = 5 + rate * max(t - t0, 0)
                wet = a * sim.blob(sx, sy, rad, aspect=1.25, drift_cm=0.35 * (rad - 5))
            elif kind == "splash" and affected:
                a = np.clip((t - t0) / 0.15, 0, 1) * (np.exp(-(t - t0 - 0.15) / 1.1) if t > t0 + 0.15 else 1.0)
                wet = a * sim.blob(sx, sy, 11.0, aspect=1.2) if t >= t0 else None
            elif kind == "condensation":
                f = np.clip((t - t0) / 2.5, 0, 1) * (1 - np.clip((t - (hours - 2.5)) / 2.0, 0, 1))
                fld = cv2.GaussianBlur(sim.rng.normal(0, 1, (sim.H, sim.W)).astype(np.float32), (0, 0), 25)
                fld = np.clip(0.75 + 7 * fld / (fld.std() + 1e-6) * 0.12, 0, 1)
                wet = f * 0.85 * fld
            elif kind == "shadow":
                p = (t / hours)
                shadow = sim.blob(40 + p * (sim.w_cm - 80), sim.h_cm * 0.45, 28, aspect=1.6, edge=0.9)
            elif kind == "light_switch":
                if 2.0 <= t < 5.0:
                    gain, grad = 0.72, 0.35
            elif kind == "dry_drift":
                gain = 1.0 + 0.12 * np.sin(2 * np.pi * t / hours)
            elif kind == "person":
                if 0.4 < (t % 1.5) < 0.8:
                    person = (80 + 500 * ((t % 1.5) - 0.4) / 0.4, 330)
            elif kind == "camera_nudge":
                if t >= 2.5:
                    nudge = (6.0, 4.0, 0.7)
            img = sim.render_map(wet=wet, shadow=shadow, gain=gain, grad=grad)
            frames.append(sim.to_camera(img, person=person, nudge=nudge))
            masks.append((wet > 0.3) if wet is not None else np.zeros((sim.H, sim.W), bool))
        walls.append({"sim": sim, "frames": frames, "gt_masks": masks, "baselines": dry_baselines(sim),
                      "gt_source": (sx, sy) if kind in ("leak", "splash") and affected else None,
                      "label": label if affected else "none", "t0": t0})
    return {"kind": kind, "times": times, "walls": walls}


def physical_quad(w_cm, h_cm, size=(640, 480), f_rel=0.85, yaw=18.0, pitch=6.0, dist_cm=None, off=(0.0, 0.0)):
    """Image corners (TL,TR,BR,BL) of a flat wall seen by a real pinhole camera (focal f_rel*width)."""
    W, H = size; f = f_rel * W
    dist = dist_cm or (f * max(w_cm / W, h_cm / H) * 1.18)
    obj = np.float64([[-w_cm / 2, -h_cm / 2, 0], [w_cm / 2, -h_cm / 2, 0], [w_cm / 2, h_cm / 2, 0], [-w_cm / 2, h_cm / 2, 0]])
    rv, _ = cv2.Rodrigues(cv2.Rodrigues(np.float64([np.radians(pitch), 0, 0]))[0] @ cv2.Rodrigues(np.float64([0, np.radians(yaw), 0]))[0])
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    pts, _ = cv2.projectPoints(obj, rv, np.float64([off[0], off[1], dist]), K, None)
    return pts.reshape(4, 2).astype(np.float32)
