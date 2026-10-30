"""Occlusion (people/objects) + privacy masking. Runs on-device only."""
import cv2
import numpy as np


class OcclusionDetector:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.prev = None

    def update(self, frame, wall_mask):
        """Returns uint8 mask (255 = occluded/unsafe) in camera coordinates."""
        small = cv2.resize(frame, None, fx=0.5, fy=0.5)
        occ = np.zeros(frame.shape[:2], np.uint8)
        try:
            rects, _ = self.hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.1)
            for (x, y, w, h) in rects:
                cv2.rectangle(occ, (2 * x, 2 * y), (2 * (x + w), 2 * (y + h)), 255, -1)
        except cv2.error:
            pass
        g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 3)
        if self.prev is not None and self.prev.shape == g.shape:
            d = np.abs(g - self.prev)
            d = np.abs(d - np.median(d))                  # remove global exposure change
            mot = (d > 28).astype(np.uint8) * 255
            mot = cv2.morphologyEx(mot, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            mot = cv2.dilate(mot, np.ones((15, 15), np.uint8))
            # only treat as occluder if the moving blob is big (a person/object, not a growing leak)
            n, lab, st, _ = cv2.connectedComponentsWithStats(mot)
            for i in range(1, n):
                if st[i, cv2.CC_STAT_AREA] > 0.012 * frame.shape[0] * frame.shape[1]:
                    occ[lab == i] = 255
        self.prev = g
        return cv2.bitwise_and(occ, wall_mask) if wall_mask is not None else occ


def privacy_blur(frame, wall_mask, occ_mask=None):
    """Blank everything outside the wall and blur occluders before any image is stored/shown."""
    out = frame.copy()
    out[wall_mask == 0] = (out[wall_mask == 0] * 0.15).astype(np.uint8)
    if occ_mask is not None and occ_mask.any():
        bl = cv2.GaussianBlur(frame, (0, 0), 18)
        out[occ_mask > 0] = bl[occ_mask > 0]
    return out
