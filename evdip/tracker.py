"""Region tracking across analyses (time in hours)."""
import cv2
import numpy as np


class Track:
    _next = 1

    def __init__(self, t):
        self.id = Track._next; Track._next += 1
        self.t_first = t
        self.hist = []             # dict(t, area, cx, cy, r_eq, dark, tex, sat, score, light, present)
        self.mask = None           # last mask (map res, uint8)
        self.misses = 0
        self.active = True
        self.label = "unknown"; self.probs = {}
        self.state = "WATCH"; self.reason = ""
        self.verdict = None        # 'verified' | 'dismissed' (human)
        self.peak_state = "WATCH"
        self.extra = {}

    @property
    def present_hist(self):
        return [h for h in self.hist if h["present"]]

    def last(self):
        return self.hist[-1] if self.hist else None


class RegionTracker:
    def __init__(self, px_per_cm, max_misses=2):
        self.tracks, self.ppc, self.max_misses = [], px_per_cm, max_misses

    def update(self, comps, t, light):
        """comps: list of dict(mask, area, cx, cy, dark, tex, sat, score). Returns matched tracks."""
        used = set()
        for tr in self.tracks:
            if not tr.active:
                continue
            best, bi = 0.0, None
            for i, c in enumerate(comps):
                if i in used:
                    continue
                inter = np.count_nonzero(cv2.bitwise_and(tr.mask, c["mask"]))
                if inter == 0:      # near-miss: dilated overlap (growing/drifting patch)
                    inter = np.count_nonzero(cv2.bitwise_and(cv2.dilate(tr.mask, np.ones((15, 15), np.uint8)), c["mask"])) * 0.5
                if inter > best:
                    best, bi = inter, i
            if bi is not None:
                c = comps[bi]; used.add(bi)
                tr.mask, tr.misses = c["mask"], 0
                tr.hist.append(dict(t=t, present=True, light=light, r_eq=float(np.sqrt(c["area"] / np.pi)), **{
                    k: c[k] for k in ("area", "cx", "cy", "dark", "tex", "sat", "score")}))
            else:
                tr.misses += 1
                tr.hist.append(dict(t=t, present=False, light=light, area=0.0, r_eq=0.0, cx=tr.hist[-1]["cx"],
                                    cy=tr.hist[-1]["cy"], dark=0, tex=0, sat=0, score=0))
                if tr.misses >= self.max_misses:
                    tr.active = False
        for i, c in enumerate(comps):
            if i in used:
                continue
            tr = Track(t); tr.mask = c["mask"]
            tr.hist.append(dict(t=t, present=True, light=light, r_eq=float(np.sqrt(c["area"] / np.pi)), **{
                k: c[k] for k in ("area", "cx", "cy", "dark", "tex", "sat", "score")}))
            self.tracks.append(tr)
        return [t_ for t_ in self.tracks if t_.active]
