# EchoVein Live Scanner — integration notes

Adds a browser-based live scanner to the existing EchoVein-DIP project. **No existing processing code was duplicated**: the scanner
calls the existing `WallMonitor` (cues, baseline library, occlusion, self-recalibration, tracking, back-tracing, thermal cue) per frame.
The existing stack is Python/OpenCV with a stdlib-only dashboard, so the scanner is also stdlib (`http.server`) — **no new dependencies**.

## Run
    python scan.py                          # http://localhost:8443  (laptop webcam works on localhost)
    python scan.py --https --host 0.0.0.0   # phone: open https://<laptop-ip>:8443 and accept the self-signed certificate warning
    python scan.py --token SECRET           # require a token (open page as /?token=SECRET) if not on a trusted network
Workflow in the page: **Start camera → Auto-calibrate (marker + wall found, width/height CALCULATED) → Capture dry reference → Start scanning.** No sizes are typed.
Laptop webcam and phone rear/front camera are selectable; camera type, resolution and session are stored in metadata.

### Phone access (camera needs a secure context)
Phones normally block `getUserMedia` on plain `http://192.168.x.x`. Options: (1) `--https` + accept the certificate warning (tested only with curl here);
(2) `mkcert` to create a trusted local cert and pass `--cert/--key` paths via `serve(...)`; (3) a temporary tunnel (`cloudflared tunnel --url https://localhost:8443`
or `ngrok http https://localhost:8443`) — **use `--token`, share the URL only with yourself, and stop the tunnel (Ctrl+C) when finished**.

## Automatic wall width/height (no typing)
A single camera cannot know absolute size by itself (it sees angles, not centimetres), so a **scale marker of known size** is detected automatically:
1. `python make_marker.py --size-cm 15` → prints-ready `marker_15cm.png` (A4, 300 dpi). Print at **100 %** on matte paper, check the 10 cm bar with a ruler, stick it **flat on the wall** (same plane), fully visible. Larger = more accurate (>= 15 cm advised; up to ~18 cm fits A4).
2. **Auto-calibrate** finds the marker (OpenCV ArUco) and the wall outline (edge/contour quadrilateral). The wall outline gives the perspective; the marker gives the scale along both wall axes → **width and height in cm are calculated**, with a Monte-Carlo ±uncertainty (2 sigma).
3. **Difficult case** (wall edges not found, wall partly out of view, or you only care about part of it): tick *I will select the area myself*, drag the 4 corners around the area, press *Measure & calibrate*. The width/height **of that area** are calculated the same way, and the distances L/R/T/B are measured from that area's edges.
4. **No marker at all**: only *relative* units are possible (% of wall height; aspect ratio estimated from perspective with an assumed focal length, approximate). The UI says so, quality is capped, and results are labelled as relative - never as cm.
Measured in simulation (physically consistent pinhole camera, 640x480 to 1280x960, yaw 18-30 deg): width/height error 0.1-2 % with the exact corners (within the stated uncertainty), 0.2-1.5 % with the auto-detected outline in 5 of 6 cases and 6-8 % in one cluttered case - **always check that the drawn corners sit on the real wall corners**. Selecting a physical 100x110 cm sub-area measured 100.7x110.9 cm.
Limits: the marker and the wall/area must be on the same flat plane; the camera must not move between Auto-calibrate and scanning (a moved camera is detected later by the existing camera-health check, then recalibrate); a small marker in the image (<~25 px) or glare on glossy paper reduces accuracy or prevents detection; not yet tested with a real printed marker on a real phone/webcam.

## Files
New: `make_marker.py` (+ `marker_15cm.png`, `marker_10cm.png`), `evdip/autoscale.py`, `scan.py`, `replay.py`, `web/scanner.html`, `evdip/scanner.py` (service), `evdip/scanner_server.py` (HTTP/HTTPS), `evdip/store.py` (SQLite + files),
`evdip/quality.py`, `evdip/walldetect.py`, `evdip/crosswall.py`, `tests/test_scanner.py`, `SCANNER.md`.
Modified (small, backwards-compatible): `evdip/common.py` (`Config.max_state`), `evdip/pipeline.py` (clamp state when `max_state="SUSPECTED"`), `evdip/backtrace.py` (`forecast_growth`).
Config (`ScannerConfig`): `proc_width`, `map_ppm`, `max_frame_bytes`, `max_jobs`, `result_log_interval_s`, `max_snapshots`, `fuse_frames`, `retention_days`, `token`, `persist`.

## API (JSON; frames are raw JPEG bodies)
| Method & path | Purpose |
|---|---|
| `POST /api/sessions` | create scan session `{wall_id, camera_type, ambient_c?, humidity?, lighting?}` |
| `GET /api/sessions`, `GET/DELETE /api/sessions/<id>` | history, detail (references, snapshots, model versions), delete all data |
| `POST …/detect_wall` | auto wall quadrilateral (Canny/Otsu contours); manual correction always possible |
| `POST …/auto_calibrate?marker_cm=15&select_area=0` | high-res frame → finds marker + wall → calculates width/height, calibrates |
| `POST …/calibrate` | corners → homography; size from `auto:true` (marker), manual `wall_w_cm/wall_h_cm`, or relative units |
| `POST …/scale_check` | two taps + known length → measured error % |
| `POST …/reference?category=dry\|earlier\|known_damp\|known_leak_surface\|other_wall` | save reference (only dry/earlier become pixel-aligned baselines) |
| `POST …/frame[?images=1]` | analyse one frame → regions, cm coordinates, indicators, category, latency |
| `POST …/thermal`, `POST …/layout` | optional °C map; optional pipe/asset layout (cm) |
| `POST …/snapshot`, `GET …/report`, `GET …/results`, `POST …/feedback` | save evidence, HTML report, history, verify/dismiss |
Frame validation: ≤6 MB, decodable image, 160–8000 px, aspect must match calibration (else 409). Busy processor → 429 (client drops the frame, no queue). Only fixed routes are served; ids are server-generated; wall ids are regex-restricted.

## What a result contains (`/frame`)
Result category (*No significant visual change / Visual change / Potential moisture-related pattern / Additional inspection recommended / Insufficient image quality*),
and **separate** indicators — image quality, calibration quality, reference reliability, DIP evidence strength, AI score (uncalibrated), thermal, evidence completeness — never merged into a leak probability.
Per region: distances from left/right/top/bottom (cm), width×height, area, pixel position, ± position uncertainty from calibration, localization level (*surface change region, not a confirmed leak source*),
and, only after ≥10 min of observation: probable source zone, pipe prior, relative severity, growth forecast ("if trend continues"). Also recurring-location flag, wall-health score (heuristic), latency, model versions.

## Honest scope / limits
- Live-scan mode reports **surface-change evidence**; behaviour classification (leak/splash/condensation) needs hours of observation (use `run.py` for long-term monitoring), so live regions are "unclassified" and escalation is capped at SUSPECTED/“inspect”.
- ML models are trained on **simulated** data only. No real labelled damp-wall dataset exists in this project, so paint-bubbling, efflorescence and crack classifiers are **not implemented**; train them with `train_ml.py` extensions once data exists. Do not treat any score as validated.
- Not implemented: SSIM/optical-flow/LBP region analysis (LBP is used only for cross-wall appearance statistics), 3D mapping, few-shot adaptation, federated/self-supervised/quantum experiments, building-level maps, confidence calibration, model compression, calibrated thermal trend analysis, interactive digital-twin time-lapse.
- Reference comparison is same-wall, same-session. Other-wall/known-damp references are compared by colour/texture statistics only (never pixel-aligned, never used for localisation).
- Cross-session reuse of a dry baseline requires the camera to be in the same place and recalibrated; not automated.
- Standalone mode (no reference) is low reliability and gives no coordinates.

## Tests
- `python -m unittest tests.test_scanner -v` — 18 automated tests over the real HTTP server with simulated walls: spec coordinate examples (120/600x300=60 cm, 100/800x240=30 cm), quality checks, calibration warnings, wall detection, storage/deletion/retention, invalid/corrupt/oversized frames, path traversal, token auth, 429 overload, missing reference, orientation change, no false alarm on dry/identical frames, damp-region detection + localisation (0.4 cm centroid error in simulation), report/snapshot/feedback, thermal validation, scale check.
- Real-browser tests (headless Chromium via Playwright, **fake camera feeding a simulated wall**): `python tests/ui_test_playwright.py` (camera start -> Auto-calibrate (marker, nothing typed) -> reference -> scan; then the select-area fallback: "No significant visual change" while dry, "Potential moisture-related pattern" when the damp patch appears, region cm table, snapshot, stop) and `python tests/ui_errors_playwright.py` (permission-denied and no-camera messages, 2-tap scale check, report page, session deletion removes files, iPhone-sized viewport). Requires `pip install playwright && playwright install chromium`.
- Running these found and fixed a real bug (duplicate element id sent all results into the resolution dropdown), which the Python tests could not catch.
- **Still to be tested on your hardware** (not possible here): a real laptop webcam, a real phone over HTTPS (certificate prompt, rear/front switching), real damp stains, different paints/materials, controlled lighting changes and camera movement on a real wall, and FPS on your laptop. Simulator latency here: ~0.2 s per 640x480 frame on one CPU core.
- Reproducible replay: `python replay.py --calib-json ... --refs ... --frames dir/` writes a CSV with model versions for fair comparisons.

## Privacy & retention
Frames are processed in memory and not stored unless you capture a reference or press *Save snapshot*; only compact results are logged (every 30 s); snapshots keep the newest 50 per session; sessions older than `retention_days` are purged at start; *Delete* in History removes rows and files. Stored/overlay images blank everything outside the wall and blur occluders. Nothing leaves the laptop.
