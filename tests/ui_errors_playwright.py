"""Browser tests for camera error messages, scale-check dialog, report/delete flow and the mobile layout."""
import sys, tempfile, json, time
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import sync_playwright
from evdip.scanner import ScannerConfig
from evdip.scanner_server import serve
from evdip.sim import simulate

d = tempfile.mkdtemp(); svc, httpd = serve(ScannerConfig(data_dir=d), "127.0.0.1", 8451, background=True)
w = simulate("leak", seed=5, hours=1, dt_min=60)["walls"][0]
mj = tempfile.mktemp(suffix=".mjpeg"); open(mj, "wb").write(cv2.imencode(".jpg", w["frames"][0], [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes() * 60)
quad = (w["sim"].quad / [640, 480]).tolist(); res = {}
with sync_playwright() as p:
    # 1) permission denied   2) no camera at all
    for name, args in (("denied", ["--use-fake-device-for-media-stream", "--deny-permission-prompts"]), ("no-camera", ["--use-fake-ui-for-media-stream"])):
        b = p.chromium.launch(args=["--no-sandbox"] + args); pg = b.new_page(); pg.goto("http://localhost:8451/"); pg.click("#bStart"); time.sleep(2)
        res[name] = pg.inner_text("#camMsg"); b.close()
    # 3) mobile layout + scale check + report + delete
    b = p.chromium.launch(args=["--no-sandbox", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", f"--use-file-for-fake-video-capture={mj}"])
    ctx = b.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True, user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) Mobile")
    pg = ctx.new_page(); errs = []; pg.on("pageerror", lambda e: errs.append(str(e))); pg.on("dialog", lambda dlg: dlg.accept("200"))
    pg.goto("http://localhost:8451/"); pg.click("#bStart"); pg.wait_for_function("document.getElementById('status').textContent.startsWith('camera on')", timeout=15000)
    pg.fill("#ww", "200"); pg.fill("#wh", "150"); pg.evaluate(f"S.corners={json.dumps(quad)}; draw()"); pg.click("#bCal"); pg.wait_for_function("document.getElementById('calOut').textContent.includes('Calibration')", timeout=15000)
    pg.click("#bScale"); box = pg.locator("#ov").bounding_box()
    for q in (quad[0], quad[1]): pg.mouse.click(box["x"] + q[0] * box["width"], box["y"] + q[1] * box["height"])
    pg.wait_for_function("document.getElementById('calOut').textContent.includes('Scale check')", timeout=10000); res["scale"] = pg.inner_text("#calOut").split("Scale check:")[1][:60]
    res["cam_type"] = pg.evaluate("S.sid && 1") and pg.evaluate("camType()")
    pg.click("#bRef"); pg.wait_for_function("document.getElementById('refs').textContent.includes('saved')", timeout=15000)
    pg.click("#bScan"); time.sleep(4); pg.click("#bSnap"); time.sleep(1)
    with ctx.expect_page() as np_: pg.click("#bRep")
    rep = np_.value; rep.wait_for_load_state(); res["report"] = rep.inner_text("body")[:70].replace("\n", " | "); rep.close()
    pg.screenshot(path=str(Path(tempfile.gettempdir()) / "ui_mobile.png"), full_page=False)
    sid = pg.evaluate("S.sid"); pg.evaluate("window.del(S.sid)"); time.sleep(1)
    res["deleted_on_disk"] = not (Path(d) / "sessions" / sid).exists(); res["js_errors"] = errs or "none"; b.close()
httpd.shutdown()
for k, v in res.items(): print(f"{k}: {v}")
