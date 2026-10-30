"""Train the two ML models on simulated data (+ optional real feedback).
  python train_ml.py --pixel-runs 40 --event-runs 60
  python train_ml.py --feedback out/feedback.csv        # add human-verified/dismissed events
Models: models/pixel_clf.joblib (per-pixel wet/dry from multi-cue features, HistGradientBoosting)
        models/event_clf.joblib (leak/splash/condensation/artifact from track behaviour, RandomForest)"""
import argparse, os, sys, time
import cv2, joblib, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from evdip.sim import simulate, KINDS
from evdip.common import Calib, Config
from evdip.cues import feature_stack
from evdip.pipeline import WallMonitor, Fleet
from evdip.eventclass import FEATURES

ap = argparse.ArgumentParser()
ap.add_argument("--pixel-runs", type=int, default=40)
ap.add_argument("--event-runs", type=int, default=60)
ap.add_argument("--feedback", default=None)
ap.add_argument("--skip-pixel", action="store_true")
ap.add_argument("--seed0", type=int, default=500)
a = ap.parse_args()
os.makedirs("models", exist_ok=True)
rng = np.random.default_rng(0)

# ---------------- 1. pixel model --------------------------------------------------------
if not a.skip_pixel:
    X, y = [], []
    t0 = time.time()
    for r in range(a.pixel_runs):
        kind = KINDS[r % len(KINDS)]
        s = simulate(kind, seed=a.seed0 + r, hours=5, dt_min=30)
        w = s["walls"][0]
        m = WallMonitor("t", Calib(w["sim"].calib_dict()), w["baselines"], cfg=Config(fusion="weighted"), pixel_model=None)
        for f, g in zip(w["frames"], w["gt_masks"]):
            info = m.analyze(f, 0.0)
            if info["skipped"] or m._pending is None:
                continue
            F = feature_stack(m._pending["cues"])
            Fs = np.dstack([cv2.resize(F[:, :, i], (F.shape[1] // 2, F.shape[0] // 2), interpolation=cv2.INTER_AREA) for i in range(F.shape[2])])
            gs = cv2.resize(g.astype(np.uint8), (Fs.shape[1], Fs.shape[0]), interpolation=cv2.INTER_NEAREST).ravel() > 0
            Fs = Fs.reshape(-1, Fs.shape[2])
            pos, neg = np.where(gs)[0], np.where(~gs)[0]
            ip = rng.choice(pos, min(len(pos), 600), replace=False) if len(pos) else []
            ineg = rng.choice(neg, 900, replace=False)
            X.append(Fs[ip] if len(ip) else np.zeros((0, Fs.shape[1]))); y += [1] * len(ip)
            X.append(Fs[ineg]); y += [0] * len(ineg)
    X = np.vstack(X); y = np.array(y)
    print(f"pixel samples {len(y)} (pos {y.sum()}) in {time.time()-t0:.0f}s")
    clf = HistGradientBoostingClassifier(max_iter=80, max_depth=5, learning_rate=0.1, random_state=0)
    from sklearn.model_selection import train_test_split
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    clf.fit(Xa, ya)
    from sklearn.metrics import f1_score
    print("pixel holdout accuracy:", round(clf.score(Xb, yb), 3), " F1(wet):", round(f1_score(yb, clf.predict(Xb)), 3))
    clf.fit(X, y)
    joblib.dump(clf, "models/pixel_clf.joblib"); print("saved models/pixel_clf.joblib")

# ---------------- 2. event model -------------------------------------------------------
LAB = {"leak": "leak", "splash": "splash", "condensation": "condensation"}
rows, labels, groups = [], [], []
t0 = time.time()
for r in range(a.event_runs):
    kind = KINDS[r % len(KINDS)]
    nw = 2 if kind == "condensation" else int(rng.integers(1, 3))
    s = simulate(kind, seed=a.seed0 + 300 + r, hours=7, dt_min=20, n_walls=nw)
    # for artifact scenarios use a weaker front-end (darkening-only/weighted) so shadow-like tracks
    # actually appear and the classifier can learn to reject them
    fus = "ml" if kind in LAB else str(rng.choice(["dark_only", "weighted", "dark_only"]))
    mons = [WallMonitor(f"w{i}", Calib(w["sim"].calib_dict()), w["baselines"],
                        cfg=Config(fusion=fus, classifier="rules", use_multicue=(fus != "dark_only"),
                                   thr=0.5 if fus == "dark_only" else 0.55))
            for i, w in enumerate(s["walls"])]
    fleet = Fleet(mons)
    for k, t in enumerate(s["times"]):
        fleet.step({f"w{i}": w["frames"][k] for i, w in enumerate(s["walls"])}, t)
        for i, m in enumerate(mons):
            lab_w = s["walls"][i]["label"]
            for tr in m.tracker.tracks:
                if not hasattr(tr, "feat") or tr.feat["n_obs"] < 2 or tr.hist[-1]["t"] != t:
                    continue
                if lab_w in LAB and kind in LAB:
                    lab = LAB[lab_w]
                elif kind in LAB:
                    continue                  # spurious track on an unaffected wall of an event scenario
                else:
                    lab = "artifact"
                rows.append([tr.feat[f] for f in FEATURES]); labels.append(lab); groups.append(r)
print(f"event samples {len(labels)} in {time.time()-t0:.0f}s:", {c: labels.count(c) for c in set(labels)})
if a.feedback and os.path.exists(a.feedback):
    import csv
    n = 0
    for row in csv.DictReader(open(a.feedback)):
        lab = "artifact" if row["verdict"] == "dismissed" else row["label_pred"]
        if lab in ("leak", "splash", "condensation", "artifact"):
            rows.append([float(row[f]) for f in FEATURES]); labels.append(lab); groups.append(10_000 + n); n += 1
    print("feedback rows added:", n)
X, y = np.array(rows), np.array(labels)
rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=3, class_weight="balanced", random_state=0)
from sklearn.model_selection import GroupKFold
print("event CV accuracy (grouped by simulation run):", cross_val_score(rf, X, y, cv=GroupKFold(4), groups=np.array(groups)).mean().round(3))
rf.fit(X, y)
joblib.dump(rf, "models/event_clf.joblib"); print("saved models/event_clf.joblib")
print("feature importance:", sorted(zip(rf.feature_importances_.round(3), FEATURES), reverse=True)[:6])
