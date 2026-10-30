"""Save a DRY reference of the wall (average of N frames to reduce noise).
  python capture_baseline.py --source "http://192.168.1.5:8080/video" --out baseline_day.png
Tip: capture several (baseline_morning.png, baseline_evening.png ...) for the lighting library later."""
import argparse, time, cv2, numpy as np
from evdip.common import Source
ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True)
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--out", default="baseline.png")
a = ap.parse_args()
s = Source(a.source); acc = None; k = 0
for _ in range(a.n):
    f = s.read()
    if f is None: continue
    acc = f.astype(np.float32) if acc is None else acc + f
    k += 1; time.sleep(0.3)
cv2.imwrite(a.out, (acc / k).astype(np.uint8)); print(f"Saved {a.out} from {k} frames")
