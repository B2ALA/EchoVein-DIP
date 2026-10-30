"""STEP 1 - calibration. Click the 4 wall corners on a phone frame.

  python calibrate.py --source "http://192.168.1.5:8080/video" --width 2.0 --height 1.5
  python calibrate.py --source wall_dry.jpg --width 2.0 --height 1.5
  python calibrate.py --source wall_dry.jpg --width 2 --height 1.5 --corners 120,80,980,95,990,700,100,690
"""
import argparse, cv2, numpy as np
from evdip.common import Source, build_calib, save_calib, Calib, order_corners

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True)
ap.add_argument("--width", type=float, required=True, help="real wall width in metres")
ap.add_argument("--height", type=float, required=True, help="real wall height in metres")
ap.add_argument("--px-per-m", type=int, default=200)
ap.add_argument("--corners", help="x1,y1,...,x4,y4 (skip the GUI)")
ap.add_argument("--out", default="calib.json")
a = ap.parse_args()

src = Source(a.source)
frame = src.read()
if frame is None:
    raise SystemExit("No frame from source")
h, w = frame.shape[:2]

if a.corners:
    v = [float(x) for x in a.corners.split(",")]
    pts = list(zip(v[0::2], v[1::2]))
else:
    scale = min(1.0, 1100 / max(w, h))
    disp0 = cv2.resize(frame, None, fx=scale, fy=scale)
    pts = []
    def click(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x / scale, y / scale))
    cv2.namedWindow("calibrate"); cv2.setMouseCallback("calibrate", click)
    while True:
        d = disp0.copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(d, (int(x * scale), int(y * scale)), 6, (0, 255, 0), -1)
            cv2.putText(d, str(i + 1), (int(x * scale) + 8, int(y * scale)), 0, 0.8, (0, 255, 0), 2)
        cv2.putText(d, "Click 4 wall corners | r=reset  Enter=save  q=quit", (10, 25), 0, 0.6, (0, 255, 255), 2)
        cv2.imshow("calibrate", d)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("r"): pts.clear()
        if k == ord("q"): raise SystemExit
        if k in (13, 10) and len(pts) == 4: break
    cv2.destroyAllWindows()

c = build_calib(pts, a.width, a.height, a.px_per_m, [w, h])
save_calib(c, a.out)
cal = Calib(c)
rect = cal.rectify(frame)
cv2.imwrite("rectified_preview.png", rect)
print(f"Saved {a.out}. Rectified wall map: {cal.W}x{cal.Hpx}px "
      f"({a.px_per_m} px/m, 1 px = {100 / a.px_per_m:.2f} cm). Preview: rectified_preview.png")
