"""Reproducible test mode: replay stored images through the same scanner pipeline and write a CSV,
so different model versions/configs can be compared on identical input.
  python replay.py --calib-json calib_session.json --refs dry1.jpg dry2.jpg --frames frames_dir --out replay.csv
(calib json = {"corners_norm":[[x,y]x4],"frame_w":W,"frame_h":H,"wall_w_cm":..,"wall_h_cm":..})"""
import argparse, csv, glob, json, tempfile
import cv2
from evdip.scanner import ScanService, ScannerConfig

ap = argparse.ArgumentParser()
ap.add_argument("--calib-json", required=True); ap.add_argument("--refs", nargs="+", required=True)
ap.add_argument("--frames", required=True); ap.add_argument("--dt-min", type=float, default=10.0); ap.add_argument("--out", default="replay.csv")
a = ap.parse_args()
svc = ScanService(ScannerConfig(data_dir=tempfile.mkdtemp(), allow_sim_time=True, result_log_interval_s=1e9))
sid = svc.create_session({"wall_id": "replay", "camera_type": "replay"})["session_id"]
svc.calibrate(sid, json.load(open(a.calib_json)))
for r in a.refs:
    svc.add_reference(sid, "dry", "replay reference", open(r, "rb").read())
files = sorted(glob.glob(a.frames + "/*.jpg") + glob.glob(a.frames + "/*.png"))
with open(a.out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["frame", "t_h", "category", "n_regions", "x_cm", "y_cm", "area_cm2", "models"])
    for i, p in enumerate(files):
        res = svc.process_frame(sid, open(p, "rb").read(), sim_t=i * a.dt_min / 60.0)
        g = max(res["regions"], key=lambda r: r.get("area_cm2", 0), default={})
        w.writerow([p, i * a.dt_min / 60.0, res["category"], len(res["regions"]), g.get("x_cm"), g.get("y_cm"), g.get("area_cm2"), json.dumps(res["model_versions"])])
print(f"replayed {len(files)} frames -> {a.out}")
