"""Illumination-robust dry-baseline library."""
import cv2
import numpy as np


def _small_L(rect):
    L = cv2.cvtColor(rect, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    return cv2.resize(L, (48, 36), interpolation=cv2.INTER_AREA) + 1.0


class BaselineLibrary:
    """Several dry reference views of the same wall (day, evening, lights on/off...).
    select() returns the one whose lighting best explains the current view; the match
    score is a robust spread (MAD) of log-ratio, so a small wet patch does not matter."""

    def __init__(self):
        self.items = []          # dict(name, rect, small)

    def add(self, name, rect):
        self.items.append({"name": name, "rect": rect, "small": _small_L(rect)})

    def __len__(self):
        return len(self.items)

    def mismatch(self, i, cur_small):
        lr = np.log(cur_small / self.items[i]["small"])
        # allow smooth lighting change: remove low-order trend (plane fit) before the MAD
        h, w = lr.shape
        yy, xx = np.mgrid[0:h, 0:w]
        A = np.c_[np.ones(h * w), xx.ravel() / w, yy.ravel() / h]
        coef, *_ = np.linalg.lstsq(A, lr.ravel(), rcond=None)
        res = lr.ravel() - A @ coef
        return float(np.median(np.abs(res - np.median(res))))

    def select(self, cur_rect):
        cs = _small_L(cur_rect)
        scores = [self.mismatch(i, cs) for i in range(len(self.items))]
        i = int(np.argmin(scores))
        return i, scores[i]
