"""Real-browser UI test (pip install playwright && playwright install chromium).
Fake camera = MJPEG of a simulated wall WITH a printed scale marker: 20 s dry, then 20 s with a damp patch, looping.
Nothing about the wall size is typed: it must be measured automatically.   python tests/ui_test_playwright.py"""
import sys, tempfile, time, json
from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from playwright.sync_api import sync_playwright
from evdip.scanner import ScannerConfig
from evdip.scanner_server import serve
from tests.test_scanner import marker_wall

ws, frame, quad = marker_wall()
enc = lambda f: cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
dry = [enc(frame(noise=1.0 + 0.3 * i)) for i in range(3)]; wet = [enc(frame(wet=0.95 * ws.blob(120, 60, 15, aspect=1.2), noise=1.0 + 0.3 * i)) for i in range(2)]
mj = tempfile.mktemp(suffix=".mjpeg")
with open(mj, "wb") as f:
    for i in range(600): f.write(dry[i % 3])
    for i in range(600): f.write(wet[i % 2])
d = tempfile.mkdtemp(); svc, httpd = serve(ScannerConfig(data_dir=d, persist=3), "127.0.0.1", 8450, background=True)
errors = []; out = Path(tempfile.gettempdir())
with sync_playwright() as p:
    b = p.chromium.launch(args=["--no-sandbox", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", f"--use-file-for-fake-video-capture={mj}"])
    pg = b.new_page(viewport={"width": 1300, "height": 1150}); pg.on("dialog", lambda d_: d_.accept())
    pg.on("pageerror", lambda e: errors.append(str(e))); pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    pg.goto("http://localhost:8450/"); pg.click("#bStart"); pg.wait_for_function("document.getElementById('status').textContent.startsWith('camera on')", timeout=15000)
    # --- A) fully automatic: marker + wall found, size calculated (nothing typed)
    pg.click("#bAuto"); pg.wait_for_function("document.getElementById('calOut').textContent.includes('measured automatically')", timeout=20000)
    print("A) auto-calibrate:", pg.inner_text("#calOut").split("\n")[0:2])
    pg.click("#bRef"); pg.wait_for_function("document.getElementById('refs').textContent.includes('saved')", timeout=15000)
    pg.click("#bScan"); pg.wait_for_function("document.getElementById('banner').textContent.includes('reference-comparison')", timeout=30000)
    print("   while dry :", pg.inner_text("#banner"))
    t0 = time.time(); seen = None
    while time.time() - t0 < 60:
        txt = pg.inner_text("#banner")
        if "Potential" in txt or "inspection" in txt: seen = txt; break
        time.sleep(1)
    print(f"   damp phase after {time.time()-t0:.0f}s:", seen); time.sleep(3)
    txt = pg.inner_text("#res"); i = txt.lower().find("regions"); print("   region row:", txt[i:i + 140].replace("\n", " | "))
    pg.screenshot(path=str(out / "ui_scan.png"), full_page=True)
    # --- B) difficult case: user selects the area, size still calculated
    pg.click("#bScan"); pg.click("#bNew"); pg.click("#bAuto") if False else None
    pg.check("#selArea"); pg.click("#bAuto"); pg.wait_for_function("document.getElementById('calOut').textContent.includes('drag')||document.getElementById('calOut').textContent.includes('Drag')", timeout=20000)
    print("B) select-area prompt:", pg.inner_text("#calOut")[:110])
    pg.evaluate(f"S.corners={json.dumps(quad)}; draw()"); pg.click("#bCal"); pg.wait_for_function("document.getElementById('calOut').textContent.includes('measured automatically')", timeout=15000)
    print("   selected area:", pg.inner_text("#calOut").split("\n")[0])
    b.close()
httpd.shutdown(); print("JS errors:", errors or "none")
