"""Event classification by drying/growth behaviour (rules + small random forest)."""
import os
import numpy as np

CLASSES = ["leak", "splash", "condensation", "artifact"]
FEATURES = ["dur", "n_obs", "area_max", "area_last_ratio", "peak_pos", "slope_norm", "mono_frac",
            "drift_y", "drift_x", "tex_support", "sat_mean", "tex_mean", "dark_mean",
            "wall_frac", "cross", "light_corr", "wall_tex", "wall_sat", "present_now"]


def track_features(tr, wall_area_cm2, cross_frac=0.0, wall_tex=0.0, wall_sat=0.0):
    h = tr.hist
    ph = tr.present_hist
    t = np.array([x["t"] for x in h]); a = np.array([x["area"] for x in h], float)
    dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    amax = float(a.max()) if len(a) else 0.0
    i_pk = int(a.argmax()) if len(a) else 0
    k = min(4, len(a))
    slope = float(np.polyfit(t[-k:], a[-k:], 1)[0]) if k >= 2 and np.ptp(t[-k:]) > 0 else 0.0
    d = np.diff(a) if len(a) > 1 else np.array([0.0])
    mono = float(np.mean(d >= -0.05 * max(amax, 1.0)))
    dy = (ph[-1]["cy"] - ph[0]["cy"]) if len(ph) > 1 else 0.0
    dx = (ph[-1]["cx"] - ph[0]["cx"]) if len(ph) > 1 else 0.0
    dark = float(np.mean([x["dark"] for x in ph])) if ph else 0.0
    tex = float(np.mean([x["tex"] for x in ph])) if ph else 0.0
    sat = float(np.mean([x["sat"] for x in ph])) if ph else 0.0
    lc = 0.0
    if len(ph) >= 4:
        s = np.array([x["score"] for x in ph]); l = np.array([x["light"] for x in ph])
        if s.std() > 1e-6 and l.std() > 1e-6:
            lc = float(abs(np.corrcoef(s, l)[0, 1]))
    f = dict(dur=dur, n_obs=len(h), area_max=amax, area_last_ratio=float(a[-1] / amax) if amax > 0 else 0.0,
             peak_pos=float((t[i_pk] - t[0]) / (dur + 1e-6)) if dur > 0 else 0.0,
             slope_norm=slope / max(amax, 1.0), mono_frac=mono, drift_y=dy, drift_x=dx,
             tex_support=(tex + sat) / (dark + 1e-3), sat_mean=sat, tex_mean=tex, dark_mean=dark,
             wall_frac=amax / wall_area_cm2, cross=cross_frac, light_corr=lc,
             wall_tex=wall_tex, wall_sat=wall_sat, present_now=float(h[-1]["present"]))
    return f


def rules(f):
    if f["n_obs"] < 2:
        return "unknown", {}
    if f["cross"] >= 0.5 and f["wall_frac"] > 0.12:
        return "condensation", {"condensation": 0.8}
    if f["wall_frac"] > 0.35 and f["tex_mean"] > 0.1:
        return "condensation", {"condensation": 0.6}
    if f["tex_support"] < 0.35:
        return "artifact", {"artifact": 0.7}
    if f["present_now"] == 0 or (f["area_last_ratio"] < 0.6 and f["dur"] > 0.3):
        return "splash", {"splash": 0.7}
    if f["slope_norm"] > 0 and f["mono_frac"] > 0.7:
        return "leak", {"leak": 0.6}
    return "unknown", {}


class EventClassifier:
    def __init__(self, mode="ml", path="models/event_clf.joblib"):
        self.mode, self.model = mode, None
        if mode == "ml" and os.path.exists(path):
            import joblib
            self.model = joblib.load(path)

    def predict(self, f):
        if f["n_obs"] < 2:
            return "unknown", {}
        if self.model is not None:
            x = np.array([[f[k] for k in FEATURES]])
            p = self.model.predict_proba(x)[0]
            probs = {c: float(v) for c, v in zip(self.model.classes_, p)}
            lab = str(max(probs, key=probs.get))
            return (lab if probs[lab] >= 0.45 else "unknown"), probs
        return rules(f)
