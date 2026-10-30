"""Create the printable scale marker.   python make_marker.py --size-cm 15
Print the PNG at 100 % (no 'fit to page'), check the printed black square with a ruler, stick it FLAT on the wall
(same plane as the wall), fully visible and not wrinkled. Bigger marker = more accurate wall size (>=15 cm advised)."""
import argparse
import cv2, numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--size-cm", type=float, default=15.0); ap.add_argument("--id", type=int, default=0)
ap.add_argument("--dpi", type=int, default=300); a = ap.parse_args()
dpi = a.dpi; px = lambda cm: int(round(cm / 2.54 * dpi))
page = np.full((px(29.7), px(21.0)), 255, np.uint8)
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
side = px(a.size_cm); m = cv2.aruco.generateImageMarker(d, a.id, side, borderBits=1)
if side + px(2) > page.shape[1]:
    raise SystemExit("marker too big for an A4 page (max ~18 cm); print on larger paper instead")
x0 = (page.shape[1] - side) // 2; y0 = px(3.0)
page[y0:y0 + side, x0:x0 + side] = m                           # white margin around it is the paper itself
f = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(page, f"EchoVein scale marker  id {a.id}  - black square = {a.size_cm:g} cm", (px(1), y0 + side + px(1.5)), f, dpi / 160, 0, 3)
cv2.putText(page, "Print at 100% (no fit-to-page). Verify: this bar must measure 10 cm with a ruler.", (px(1), y0 + side + px(2.7)), f, dpi / 250, 0, 2)
yb = y0 + side + px(3.5); cv2.rectangle(page, (px(1), yb), (px(11), yb + px(0.3)), 0, -1)
name = f"marker_{a.size_cm:g}cm.png"; cv2.imwrite(name, page); print("wrote", name, f"({page.shape[1]}x{page.shape[0]} px @ {dpi} dpi, A4)")
