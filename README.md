# EchoVein-DIP v2 — phone/CCTV moisture-aware wall monitor

An existing camera (here: a phone) becomes a **moisture-aware, coordinate-reporting monitor**:
it reports *where on the wall* (cm from left/right/top/bottom), *how fast it grows*, *what it probably is*
(leak / splash / condensation / lighting artifact), *where the source probably is*, *which pipe segment is most likely*,
and *what it will reach next* — with an evidence card for every alert.
It complements sensors; it detects **surface evidence only** (a hidden leak that has not reached the surface is not seen).

## Install
    pip install -r requirements.txt        # Python 3.9+, opencv-python, numpy, scikit-learn

## NEW: live scanner (phone/laptop camera in the browser)
See **SCANNER.md**. Print the scale marker first (`python make_marker.py`): the wall width/height are then measured automatically, nothing typed. `python scan.py` → http://localhost:8443 (phone: `python scan.py --https --host 0.0.0.0`). Tests: `python -m unittest tests.test_scanner -v`.

## 60-second demo (no hardware)
    python run.py --demo                   # simulated 2-wall leak -> open http://localhost:8000
Dashboard: **Live CCTV view** (camera, wall boundary, tracked regions, camera-health) and **Leak analysis view**
(rectified wall map, source ellipse, pipe prior, assets-at-risk, evidence card, growth plot, ranking, history, Verify/Dismiss).

## Use your phone
Phone on a tripod, fixed. Source options:
- Android **IP Webcam** app -> `http://PHONE_IP:8080/video` (same Wi-Fi)
- **DroidCam / Iriun / Camo** (Android/iPhone) -> shows as webcam index `0`
- Or photos/video files (`wall.jpg`, `clip.mp4`)

    python calibrate.py --source <src> --width 2.0 --height 1.5           # click 4 corners (TL,TR,BR,BL) -> calib.json
    python capture_baseline.py --source <src> --out baseline_day.png      # wall DRY; repeat per lighting (evening, lights on/off)
    # edit project.example.json (+ layout_wall1.example.json: pipes & sockets in cm on the wall map)
    python run.py --project project.json

Safe test: damp sponge / spray on a painted board or cardboard, away from electrics.

## Pipeline (your 8-step build order -> files)
| Step | What | Where |
|---|---|---|
| 1 | capture, calibration, homography, cm output (top/bottom/left/right) | `calibrate.py`, `evdip/common.py` |
| 2 | dry-reference differencing, threshold, morphology, regions | `evdip/cues.py`, `evdip/pipeline.py` |
| 3 | illumination normalisation, **baseline library**, shadow & occlusion handling, privacy masking | `evdip/baselines.py`, `evdip/cues.py`, `evdip/occlusion.py` |
| 4 | temporal tracking, drying-behaviour classifier, alert state machine | `evdip/tracker.py`, `evdip/eventclass.py`, `evdip/pipeline.py` |
| 5 | source back-tracing (+uncertainty), pipe-layout prior, pipe arc length, assets-at-risk, severity index | `evdip/backtrace.py` |
| 6 | multi-wall ranking, per-wall robust z-score, cross-wall (condensation) evidence | `Fleet` in `evdip/pipeline.py` |
| 7 | thermal module (optional) + dashboard | `evdip/thermal.py`, `evdip/dashboard.py` |
| 8 | validation: simulator, metrics, ablations | `evdip/sim.py`, `validate.py` |
Extra: camera health & self-recalibration (`evdip/health.py`), pluggable evidence record (`evdip/evidence.py`).

## Where AI / ML is used (and what is classical)
- **Pixel classifier (ML):** HistGradientBoosting on per-pixel multi-cue features (darkening, saturation gain, texture loss, sheen loss, context). Learns to reject shadows/exposure changes.
- **Event classifier (ML):** RandomForest on behaviour features of each tracked region (growth, drying, drift, cross-wall, light correlation) -> leak / splash / condensation / artifact.
- **Human-in-the-loop learning:** Verify/Dismiss buttons tune the threshold live and append rows to `out/feedback.csv`; `python train_ml.py --skip-pixel --feedback out/feedback.csv` retrains with them.
- **Classical (stands alone, used as fallback and ablation):** weighted multi-cue fusion, rule-based events, ORB+RANSAC self-recalibration, HOG person detection, regression-based source back-tracing.
- Models ship pre-trained on **simulated** data (`models/`). Retrain with real labelled phone images/feedback before trusting them on a real wall (`python train_ml.py`).

## Validate (the part that makes it credible)
    python validate.py --seeds 3           # -> validation_report.md (+ ablation table), validation_runs.csv
Report only measured results, with the setup. Simulator numbers show module contributions, **not** real-world accuracy:
repeat on a physical rig (sponge/spray at known positions, time-lapse, ground truth) and report localisation error (cm), IoU,
detection delay, false alarms/day. Also run negative tests: shadows, people, lights switching, auto-exposure, condensation, camera nudges.

## Claim discipline
- Camera sees surface evidence; hidden leaks are invisible.
- Position = the wet/anomalous region + an estimated **source area** (ellipse), not a certain pipe break.
- Alerts stay **suspected** until a person verifies. Severity is a **relative index (0-100)**, not litres/hour.
- Pipe match is a probability; homography is valid only for the wall plane (calibrate pipes at other depths separately).
- Accuracy depends on calibration and lighting; quote your measured numbers.

## Privacy
Everything runs on-device; only the wall region is analysed, people are masked and frames with a large occlusion are skipped; stored/served images blank everything outside the wall and blur occluders (`privacy: true`).

## Known limits
- Median/robust lighting model assumes < ~50 % of the wall is wet; very large uniform wetting is handled via cross-wall/wall-level cues.
- Camera self-recalibration needs visible static features around the wall (door frames, posters) or add ArUco/tape markers.
- Analysis rate 5-60 s per wall is plenty (leaks are slow).

## Latest simulated validation
See `validation_report.md` (2 seeds x 8 scenarios x 9 configurations, simulator only). Re-run `python validate.py --seeds 3` after any change.
Caveats: only 2 seeds per cell (so 8-17 % steps are one run each); the simulator is easier than a real wall; occlusion handling and
the illumination library showed no measurable benefit in this simulator (their scenarios are already handled by other modules) - test them on a real rig.
