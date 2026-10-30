"""Validation + ablation on simulated ground truth (no hardware).
  python validate.py --seeds 3 [--quick]  ->  validation_report.md, validation_runs.csv
Metrics: localisation error (cm), IoU, detection delay (h), false alarms per simulated day,
event-classification accuracy.  Every number comes from running this script - report setup too."""
import argparse, csv, time
import numpy as np
from evdip.sim import simulate
from evdip.common import Calib, Config
from evdip.pipeline import WallMonitor, Fleet, ORDER

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--hours", type=float, default=8)
a = ap.parse_args()

POS = ["leak", "splash"]
NEG = ["shadow", "light_switch", "person", "camera_nudge", "dry_drift", "condensation"]
CONFIGS = {
    "full (ML pixel + ML events)": dict(fusion="ml", classifier="ml"),
    "full (weighted fusion, rule events)": dict(fusion="weighted", classifier="rules"),
    "- illumination library": dict(fusion="ml", use_library=False),
    "- illumination field normalisation": dict(fusion="ml", use_illum=False),
    "- multi-cue fusion (darkening only)": dict(fusion="dark_only", use_multicue=False),
    "- temporal logic (persist=1, no tracking memory)": dict(fusion="ml", use_temporal=False, persist=1),
    "- occlusion handling": dict(fusion="ml", use_occlusion=False),
    "- camera self-recalibration": dict(fusion="ml", use_selfcal=False),
    "- event classifier (no cross-wall/drying logic)": dict(fusion="ml", use_classifier=False, classifier="rules"),
}
if a.quick:
    CONFIGS = {k: CONFIGS[k] for k in list(CONFIGS)[:3]}


def run(kind, seed, cfgd):
    n_w = 2 if kind in ("condensation", "leak") else 1
    s = simulate(kind, seed=seed, hours=a.hours, dt_min=20, n_walls=n_w)
    cfg = Config(**{"fusion": "weighted", **cfgd})
    mons = [WallMonitor(f"w{i}", Calib(w["sim"].calib_dict()), w["baselines"], cfg=Config(**cfg.to_dict())) for i, w in enumerate(s["walls"])]
    fleet = Fleet(mons)
    first_alert, alerts, last = None, 0, None
    for k, t in enumerate(s["times"]):
        snaps = fleet.step({f"w{i}": w["frames"][k] for i, w in enumerate(s["walls"])}, float(t))
        for n, sn in snaps.items():
            bad = [tr for tr in sn["tracks"] if tr["active"] and tr["state"] in ("SUSPECTED", "ESCALATED")]
            if bad:
                if n == "w0" and first_alert is None:
                    first_alert = float(t)
                alerts += 1
        last = snaps
    res = {"kind": kind, "seed": seed, "alert_frames": alerts, "delay": None, "loc_err": None, "iou": None, "cls_ok": None,
           "escalated": False, "false_alarm": False, "hours": a.hours}
    w0 = s["walls"][0]; m0 = mons[0]
    if kind in POS:
        res["delay"] = None if first_alert is None else first_alert - w0["t0"]
        tracks = [tr for tr in m0.tracker.tracks if tr.peak_state in ("SUSPECTED", "ESCALATED")]
        if tracks:
            tr = max(tracks, key=lambda x: max([h["area"] for h in x.hist]))
            src = tr.extra.get("src")
            if src:
                gx, gy = w0["gt_source"]; res["loc_err"] = float(np.hypot(src["x_cm"] - gx, src["y_cm"] - gy))
            # IoU at peak area frame
            ks = int(np.argmax([h["area"] for h in tr.hist])); tm = tr.hist[ks]["t"]
            kk = int(np.argmin(np.abs(s["times"] - tm))); gt = w0["gt_masks"][kk]
            if kind == "leak":      # compare last observed mask with GT at last time
                kk = len(s["times"]) - 1; gt = w0["gt_masks"][kk]
                import cv2
                pm = tr.mask > 0
                if pm.shape == gt.shape:
                    res["iou"] = float((pm & gt).sum() / max((pm | gt).sum(), 1))
            res["cls_ok"] = tr.label == kind or (kind == "splash" and tr.label in ("splash", "unknown"))
        res["escalated"] = any(tr.peak_state == "ESCALATED" for tr in m0.tracker.tracks)
    else:
        # false alarm = any SUSPECTED/ESCALATED alert on any wall (condensation must be explained, not alarmed)
        res["false_alarm"] = alerts > 0
        res["escalated"] = any(tr.peak_state == "ESCALATED" for m in mons for tr in m.tracker.tracks)
    return res


rows = []
t0 = time.time()
for cname, cd in CONFIGS.items():
    for kind in POS + NEG:
        for sd in range(a.seeds):
            r = run(kind, 40 + sd, cd); r["config"] = cname; rows.append(r)
    print(f"{cname} done ({time.time()-t0:.0f}s)", flush=True)

with open("validation_runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def agg(cname):
    R = [r for r in rows if r["config"] == cname]
    pos = [r for r in R if r["kind"] in POS]; leak = [r for r in R if r["kind"] == "leak"]; neg = [r for r in R if r["kind"] in NEG]
    det = np.mean([r["delay"] is not None for r in pos]) if pos else 0
    ld = [r["delay"] for r in pos if r["delay"] is not None]
    le = [r["loc_err"] for r in leak if r["loc_err"] is not None]
    io = [r["iou"] for r in leak if r["iou"] is not None]
    fa = np.mean([r["false_alarm"] for r in neg]) if neg else 0
    days = sum(r["hours"] for r in neg) / 24.0
    fa_day = sum(r["false_alarm"] for r in neg) / max(days, 1e-9)
    esc = np.mean([r["escalated"] for r in leak]) if leak else 0
    cl = [r["cls_ok"] for r in pos if r["cls_ok"] is not None]
    return [f"{det*100:.0f}%", f"{np.mean(ld):.2f}" if ld else "-", f"{np.median(le):.1f}" if le else "-",
            f"{np.mean(io):.2f}" if io else "-", f"{esc*100:.0f}%", f"{fa*100:.0f}%", f"{fa_day:.2f}", f"{np.mean(cl)*100:.0f}%" if cl else "-"]


hdr = ["Configuration", "Detected", "Delay h (mean)", "Source err cm (median)", "IoU leak", "Leaks escalated", "Neg. scenarios alarmed", "False alarms / sim-day", "Event class OK"]
with open("validation_report.md", "w") as f:
    f.write("# EchoVein-DIP validation (simulated ground truth)\n\n")
    f.write(f"Setup: {a.seeds} seeds x {len(POS)+len(NEG)} scenarios per configuration, {a.hours:.0f} h each, one analysis / 20 min, "
            f"simulated 2 m x 1.5 m wall, perspective camera 640x480. Positive: {POS}. Negative: {NEG}.\n"
            "These are simulator results: they show the contribution of each module, not real-world accuracy. Repeat with a phone rig.\n\n")
    f.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
    for c in CONFIGS:
        f.write("| " + " | ".join([c] + agg(c)) + " |\n")
    f.write("\n## Per-scenario outcome (full ML configuration)\n\n| scenario | seed | alert | escalated | delay h | source err cm | IoU |\n|---|---|---|---|---|---|---|\n")
    first = list(CONFIGS)[0]
    for r in [r for r in rows if r["config"] == first]:
        f.write(f"| {r['kind']} | {r['seed']} | {'yes' if r['alert_frames'] else 'no'} | {'yes' if r['escalated'] else 'no'} | "
                f"{'' if r['delay'] is None else round(r['delay'], 2)} | {'' if r['loc_err'] is None else round(r['loc_err'], 1)} | {'' if r['iou'] is None else round(r['iou'], 2)} |\n")
print(open("validation_report.md").read())
