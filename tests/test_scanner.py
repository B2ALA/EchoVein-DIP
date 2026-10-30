"""python -m unittest tests.test_scanner -v   (uses simulated walls; no camera needed)"""
import http.client, json, shutil, sys, tempfile, unittest
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evdip.scanner import ScannerConfig
from evdip.scanner_server import serve
from evdip.sim import simulate
from evdip.walldetect import rect_to_cm, calibration_quality, detect_wall_quad
from evdip.quality import assess_quality
from evdip.store import Store


def jpg(img, q=90):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


class Client:
    def __init__(self, port, token=None):
        self.port, self.token = port, token

    def call(self, method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        h = dict(headers or {})
        if self.token: h["X-EchoVein-Token"] = self.token
        if isinstance(body, (dict, list)): body = json.dumps(body).encode(); h["Content-Type"] = "application/json"
        c.request(method, path, body=body, headers=h); r = c.getresponse(); raw = r.read(); c.close()
        try: return r.status, json.loads(raw)
        except Exception: return r.status, raw


class Maths(unittest.TestCase):
    def test_spec_worked_examples(self):
        self.assertAlmostEqual(rect_to_cm(120, 600, 300), 60.0)       # spec example 1
        self.assertAlmostEqual(rect_to_cm(100, 800, 240), 30.0)       # spec example 2

    def test_quality_flags(self):
        dark = np.full((480, 640, 3), 10, np.uint8); self.assertFalse(assess_quality(dark)["ok"])
        glare = np.random.default_rng(0).integers(100, 160, (480, 640, 3)).astype(np.uint8); glare[:200] = 255
        self.assertFalse(assess_quality(glare)["ok"])
        small = np.random.default_rng(0).integers(60, 200, (100, 100, 3)).astype(np.uint8); self.assertFalse(assess_quality(small)["ok"])

    def test_calibration_quality(self):
        good = calibration_quality([[100, 80], [540, 80], [540, 400], [100, 400]], (640, 480), 300, 210)
        self.assertIn(good["level"], ("good", "fair"))
        edge = calibration_quality([[0, 0], [639, 0], [639, 479], [0, 479]], (640, 480), 300, 240)
        self.assertTrue(any("edge" in w for w in edge["warnings"]))
        degenerate = calibration_quality([[100, 100], [200, 100], [300, 100], [100, 100]], (640, 480), 300, 240)
        self.assertEqual(degenerate["level"], "invalid")

    def test_wall_detection_clean(self):
        img = np.full((480, 640, 3), 30, np.uint8); cv2.fillPoly(img, [np.int32([[120, 90], [520, 70], [540, 400], [100, 420]])], (200, 210, 220))
        d = detect_wall_quad(img); self.assertTrue(d["found"]); self.assertGreater(d["score"], 0.3)

    def test_store_lifecycle(self):
        d = tempfile.mkdtemp(); s = Store(d); sid = s.create_session("w1", "phone", {})
        with self.assertRaises(ValueError): s.create_session("../../etc", "phone", {})
        s.add_ref(sid, "w1", "dry", b"x", {}); s.add_event(sid, "w1", 50, 50, "SUSPECTED", "x")
        self.assertEqual(s.recurring_count("w1", 55, 52, 15, exclude_session="other"), 1)
        self.assertEqual(s.recurring_count("w1", 55, 52, 15, exclude_session=sid), 0)
        self.assertTrue(s.delete_session(sid)); self.assertIsNone(s.get_session(sid)); self.assertFalse((Path(d) / "sessions" / sid).exists())
        shutil.rmtree(d)


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cfg = ScannerConfig(data_dir=cls.dir, allow_sim_time=True, persist=3, result_log_interval_s=0)
        cls.svc, cls.httpd = serve(cfg, "127.0.0.1", 0, background=True)
        cls.c = Client(cls.httpd.server_address[1])
        cls.sim = simulate("leak", seed=5, hours=4, dt_min=10)
        cls.w = cls.sim["walls"][0]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown(); shutil.rmtree(cls.dir, ignore_errors=True)

    def session(self, calibrate=True):
        s, r = self.c.call("POST", "/api/sessions", {"wall_id": "test-wall", "camera_type": "phone-rear", "humidity": 50})
        self.assertEqual(s, 200); sid = r["session_id"]
        if calibrate:
            q = (self.w["sim"].quad / [640, 480]).tolist()
            s, r = self.c.call("POST", f"/api/sessions/{sid}/calibrate", {"corners_norm": q, "frame_w": 640, "frame_h": 480, "wall_w_cm": 200, "wall_h_cm": 150})
            self.assertEqual(s, 200, r)
        return sid

    def test_01_pages_and_security(self):
        s, body = self.c.call("GET", "/"); self.assertEqual(s, 200); self.assertIn(b"getUserMedia", body)
        for p in ("/../../etc/passwd", "/etc/passwd", "/api/sessions/..%2f..%2fx", "/api/sessions/zzzzzzzzzzzz", "/web/scanner.html"):
            self.assertIn(self.c.call("GET", p)[0], (404,), p)
        s, r = self.c.call("POST", "/api/sessions", {"wall_id": "../../x"}); self.assertEqual(s, 400)

    def test_02_invalid_inputs(self):
        sid = self.session(calibrate=False)
        self.assertEqual(self.c.call("POST", f"/api/sessions/{sid}/frame", b"not an image")[0], 400)
        self.assertEqual(self.c.call("POST", f"/api/sessions/{sid}/frame", b"")[0], 400)
        self.assertEqual(self.c.call("POST", f"/api/sessions/{sid}/frame", b"\0" * 6_200_000)[0], 413)
        self.assertEqual(self.c.call("POST", f"/api/sessions/{sid}/calibrate", {"corners_norm": [[0, 0]] * 4, "frame_w": 640, "frame_h": 480, "wall_w_cm": 0, "wall_h_cm": 5})[0], 400)
        self.assertEqual(self.c.call("POST", f"/api/sessions/{sid}/reference?category=dry", jpg(self.w["frames"][0]))[0], 409)   # not calibrated
        self.assertEqual(self.c.call("POST", f"/api/sessions/{sid}/thermal", {"temps": [[1, 2], [3, 4]]})[0], 400)
        self.assertEqual(self.c.call("POST", f"/api/sessions/{sid}/reference?category=bogus", jpg(self.w["frames"][0]))[0], 400)

    def test_03_missing_reference_is_standalone(self):
        sid = self.session()
        s, r = self.c.call("POST", f"/api/sessions/{sid}/frame", jpg(self.w["frames"][0]))
        self.assertEqual(s, 200); self.assertEqual(r["mode"], "standalone"); self.assertIn("none", r["reference_status"])
        self.assertTrue(any("Standalone" in x for x in r["warnings"]))

    def test_04_bad_quality_skipped(self):
        sid = self.session(); dark = np.full((480, 640, 3), 8, np.uint8)
        s, r = self.c.call("POST", f"/api/sessions/{sid}/frame", jpg(dark))
        self.assertEqual(r["category"], "quality"); self.assertEqual(r["regions"], [])

    def test_05_busy_returns_429(self):
        sid = self.session()
        for _ in range(self.svc.cfg.max_jobs): self.svc.sem.acquire()
        try: self.assertEqual(self.c.call("POST", f"/api/sessions/{sid}/frame", jpg(self.w["frames"][0]))[0], 429)
        finally:
            for _ in range(self.svc.cfg.max_jobs): self.svc.sem.release()

    def test_06_full_workflow_detects_and_localises(self):
        sid = self.session()
        for name, img in self.w["baselines"]:
            s, r = self.c.call("POST", f"/api/sessions/{sid}/reference?category=dry&notes={name}", jpg(img)); self.assertEqual(s, 200, r)
        self.assertTrue(r["monitoring"])
        # other-wall reference must be accepted without becoming a baseline
        other = simulate("leak", seed=9, hours=1, dt_min=60)["walls"][0]["frames"][0]
        self.assertEqual(self.c.call("POST", f"/api/sessions/{sid}/reference?category=other_wall", jpg(other))[0], 200)
        times, frames, masks = self.sim["times"], self.w["frames"], self.w["gt_masks"]; t0 = self.w["t0"]
        cats, last, lat = [], None, []
        for k, (t, f) in enumerate(zip(times, frames)):
            s, r = self.c.call("POST", f"/api/sessions/{sid}/frame?images={1 if k == len(times) - 1 else 0}&t={t:.4f}", jpg(f)); self.assertEqual(s, 200, r)
            cats.append((t, r["category"])); last = r; lat.append(r["perf"]["latency_ms"])
            if t < t0 - 0.2: self.assertEqual(r["category"], "none", f"false alarm on dry wall at t={t:.2f}")   # identical/dry frames
        self.assertIn(last["category"], ("potential", "inspect"), cats[-6:])
        reg = max(last["regions"], key=lambda g: g["area_cm2"])
        gt = masks[-1]; ys, xs = np.nonzero(gt); gx, gy = xs.mean() / 2.0, ys.mean() / 2.0       # map px -> cm at 200 px/m
        err = float(np.hypot(reg["x_cm"] - gx, reg["y_cm"] - gy))
        print(f"\n  centroid error {err:.1f} cm; uncertainty +/-{reg['uncertainty_cm']:.1f} cm; mean latency {np.mean(lat):.0f} ms")
        self.assertLess(err, 6.0)
        self.assertAlmostEqual(reg["left_cm"] + reg["right_cm"], 200, delta=0.5); self.assertAlmostEqual(reg["top_cm"] + reg["bottom_cm"], 150, delta=0.5)
        self.assertIn("not a confirmed", reg["localization_level"]); self.assertTrue(reg["polygons"])
        self.assertIn("images", last); self.assertTrue(last["images"]["processed"].startswith("data:image/jpeg"))
        ind = last["indicators"]; self.assertIn("image_quality", ind); self.assertIn("evidence_completeness", ind); self.assertNotIn("leak_probability", json.dumps(last))
        self.assertEqual(last["model_versions"]["evdip"], "2.0")
        # snapshot + report + feedback + results
        self.assertEqual(self.c.call("POST", f"/api/sessions/{sid}/snapshot")[0], 200)
        s, rep = self.c.call("GET", f"/api/sessions/{sid}/report"); self.assertEqual(s, 200); self.assertIn(b"Limitations", rep); self.assertIn(b"NOT a confirmed", rep)
        self.assertEqual(self.c.call("POST", f"/api/sessions/{sid}/feedback", {"track": reg["id"], "verdict": "dismissed"})[0], 200)
        self.assertEqual(self.c.call("POST", f"/api/sessions/{sid}/feedback", {"track": reg["id"], "verdict": "maybe"})[0], 400)
        self.assertGreaterEqual(len(self.c.call("GET", f"/api/sessions/{sid}/results")[1]["results"]), 1)
        # orientation change -> 409, then delete wipes everything
        self.assertEqual(self.c.call("POST", f"/api/sessions/{sid}/frame", jpg(np.rot90(frames[0]).copy()))[0], 409)
        s, det = self.c.call("GET", f"/api/sessions/{sid}"); self.assertEqual(len(det["references"]), 3)
        self.assertEqual(self.c.call("DELETE", f"/api/sessions/{sid}")[0], 200); self.assertEqual(self.c.call("GET", f"/api/sessions/{sid}")[0], 404)
        self.assertFalse((Path(self.dir) / "sessions" / sid).exists())

    def test_07_thermal_and_scale(self):
        sid = self.session()
        t = np.full((24, 32), 22.0); t[8:14, 10:18] = 19.5
        s, r = self.c.call("POST", f"/api/sessions/{sid}/thermal", {"temps": t.tolist(), "units": "C", "device": "MLX90640"}); self.assertEqual(s, 200); self.assertIn("ASSUMED", r["registration"])
        self.assertEqual(self.c.call("POST", f"/api/sessions/{sid}/thermal", {"temps": (t + 900).tolist()})[0], 400)
        q = (self.w["sim"].quad / [640, 480]); s, r = self.c.call("POST", f"/api/sessions/{sid}/scale_check", {"p1_norm": q[0].tolist(), "p2_norm": q[1].tolist(), "length_cm": 200})
        self.assertEqual(s, 200); self.assertLess(abs(r["error_pct"]), 1.0)      # top edge of the wall is 200 cm


def marker_wall(wc=200.0, hc=150.0, marker=(150, 40, 15), seed=1, yaw=18.0):
    """Physically consistent pinhole view of a wall (+ optional ArUco marker). Returns (frame_fn, quad_norm)."""
    from evdip.sim import WallSim, physical_quad
    q = physical_quad(wc, hc, (640, 480), yaw=yaw)
    ws = WallSim(seed=seed, w_m=wc / 100, h_m=hc / 100, ppm=200); ws.quad = q.copy()
    ws.M = cv2.getPerspectiveTransform(np.float32([[0, 0], [ws.W, 0], [ws.W, ws.H], [0, ws.H]]), q)
    ws.mask_cam = cv2.warpPerspective(np.full((ws.H, ws.W), 255, np.uint8), ws.M, (640, 480))
    def frame(wet=None, noise=1.5):
        return ws.to_camera(ws.render_map(wet=wet, marker=(None if marker is None else (*marker, 0)), noise=noise))
    return ws, frame, (q / [640, 480]).tolist()


class AutoDimensions(unittest.TestCase):
    """The wall width/height must be CALCULATED from the camera view - nothing typed."""
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.svc, cls.httpd = serve(ScannerConfig(data_dir=cls.dir, allow_sim_time=True, persist=3, result_log_interval_s=0), "127.0.0.1", 0, background=True)
        cls.c = Client(cls.httpd.server_address[1])

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown(); shutil.rmtree(cls.dir, ignore_errors=True)

    def new(self):
        s, r = self.c.call("POST", "/api/sessions", {"wall_id": "auto-wall", "camera_type": "laptop-webcam"}); return r["session_id"]

    def test_01_fully_automatic_size_and_detection(self):
        ws, frame, qn = marker_wall(); sid = self.new()
        s, r = self.c.call("POST", f"/api/sessions/{sid}/auto_calibrate?marker_cm=15&apply=1", jpg(frame()))
        self.assertEqual(s, 200, r); self.assertEqual(r["status"], "calibrated", r)
        cal = r["calibration"]; w, h = cal["wall_w_cm"], cal["wall_h_cm"]
        print(f"\n  measured automatically: {w:.1f} x {h:.1f} cm (true 200 x 150), +/-{cal['dims_uncertainty_cm'][0]:.1f}/{cal['dims_uncertainty_cm'][1]:.1f} cm, method {cal['dims_method']}")
        self.assertLess(abs(w / 200 - 1), 0.03); self.assertLess(abs(h / 150 - 1), 0.03); self.assertEqual(cal["dims_method"], "marker")
        for i in range(2):
            self.assertEqual(self.c.call("POST", f"/api/sessions/{sid}/reference?category=dry", jpg(frame(noise=1.0 + i)))[0], 200)
        gx, gy = 120.0, 60.0; last = None
        for k in range(6):
            wet = min(1.0, 0.5 + 0.2 * k) * ws.blob(gx, gy, 14, aspect=1.2)
            s, last = self.c.call("POST", f"/api/sessions/{sid}/frame?t={k*0.02:.3f}", jpg(frame(wet=wet))); self.assertEqual(s, 200, last)
        self.assertEqual(last["units"], "cm"); self.assertTrue(last["regions"], last["category"])
        g = max(last["regions"], key=lambda x: x["area_cm2"]); err = float(np.hypot(g["x_cm"] - gx, g["y_cm"] - gy))
        print(f"  region at ({g['x_cm']:.0f},{g['y_cm']:.0f}) cm vs true ({gx:.0f},{gy:.0f}): {err:.1f} cm error (+/-{g['uncertainty_cm']:.1f})")
        self.assertLess(err, 6.0)

    def test_02_select_area_fallback(self):
        ws, frame, qn = marker_wall(); sid = self.new()
        s, r = self.c.call("POST", f"/api/sessions/{sid}/auto_calibrate?marker_cm=15&select_area=1", jpg(frame()))
        self.assertEqual(r["status"], "needs_area"); self.assertTrue(r["scale_found"])
        # user drags corners around the whole wall -> size is calculated
        s, r = self.c.call("POST", f"/api/sessions/{sid}/calibrate", {"corners_norm": qn, "frame_w": 640, "frame_h": 480, "auto": True}); self.assertEqual(s, 200, r)
        self.assertLess(abs(r["calibration"]["wall_w_cm"] / 200 - 1), 0.03)
        # ...or around a SMALLER physical area (left 100 cm x full height): its OWN width/height are measured
        Hr = cv2.getPerspectiveTransform(np.float32([[0, 0], [200, 0], [200, 150], [0, 150]]), np.float32(np.array(qn) * [640, 480]))
        area = (cv2.perspectiveTransform(np.float32([[[0, 20], [100, 20], [100, 130], [0, 130]]]), Hr)[0] / [640, 480]).tolist()
        s, r = self.c.call("POST", f"/api/sessions/{sid}/calibrate", {"corners_norm": area, "frame_w": 640, "frame_h": 480, "auto": True}); self.assertEqual(s, 200, r)
        print(f"\n  selected area measured: {r['calibration']['wall_w_cm']:.1f} x {r['calibration']['wall_h_cm']:.1f} cm (true 100 x 110)")
        self.assertLess(abs(r["calibration"]["wall_w_cm"] / 100 - 1), 0.04); self.assertLess(abs(r["calibration"]["wall_h_cm"] / 110 - 1), 0.04)

    def test_03_no_marker_gives_relative_units_only(self):
        ws, frame, qn = marker_wall(marker=None); sid = self.new()
        s, r = self.c.call("POST", f"/api/sessions/{sid}/auto_calibrate?marker_cm=15", jpg(frame()))
        self.assertEqual(r["status"], "needs_scale"); self.assertFalse(r["scale_found"])
        s, r = self.c.call("POST", f"/api/sessions/{sid}/calibrate", {"corners_norm": qn, "frame_w": 640, "frame_h": 480}); self.assertEqual(s, 200, r)
        cal = r["calibration"]; self.assertEqual(cal["units"], "relative"); self.assertEqual(cal["wall_h_cm"], 100.0)
        print(f"\n  relative mode: aspect {cal['wall_w_cm']/100:.3f} (true 1.333)"); self.assertLess(abs(cal["wall_w_cm"] / 133.3 - 1), 0.08)
        self.assertTrue(any("NO scale" in w for w in cal["quality"]["warnings"])); self.assertLessEqual(cal["quality"]["score"], 0.55)

    def test_05_odd_measured_sizes_do_not_crash(self):
        """Regression: measured sizes give odd map dimensions (e.g. 203.3 x 151.7 cm) - ML scoring must still work."""
        ws, frame, qn = marker_wall(); sid = self.new()
        for dims in ((203.3, 151.7), (199.9, 149.4), (187.1, 143.3)):
            s, r = self.c.call("POST", f"/api/sessions/{sid}/calibrate", {"corners_norm": qn, "frame_w": 640, "frame_h": 480, "wall_w_cm": dims[0], "wall_h_cm": dims[1]}); self.assertEqual(s, 200, r)
            self.assertEqual(self.c.call("POST", f"/api/sessions/{sid}/reference?category=dry", jpg(frame()))[0], 200)
            for k in range(3):
                s, r = self.c.call("POST", f"/api/sessions/{sid}/frame?t={k*0.01}", jpg(frame())); self.assertEqual(s, 200, (dims, r))

    def test_04_guards(self):
        ws, frame, qn = marker_wall(); sid = self.new()
        self.assertEqual(self.c.call("POST", f"/api/sessions/{sid}/calibrate", {"corners_norm": qn, "frame_w": 640, "frame_h": 480, "auto": True})[0], 409)
        self.assertEqual(self.c.call("POST", f"/api/sessions/{sid}/auto_calibrate?marker_cm=0.5", jpg(frame()))[0], 400)
        self.assertEqual(self.c.call("POST", f"/api/sessions/{sid}/auto_calibrate?marker_cm=15", b"junk")[0], 400)


class Auth(unittest.TestCase):
    def test_token(self):
        d = tempfile.mkdtemp(); svc, h = serve(ScannerConfig(data_dir=d, token="s3cret"), "127.0.0.1", 0, background=True)
        try:
            p = h.server_address[1]
            self.assertEqual(Client(p).call("GET", "/api/sessions")[0], 401)
            self.assertEqual(Client(p, "s3cret").call("GET", "/api/sessions")[0], 200)
            self.assertEqual(Client(p).call("GET", "/api/sessions?token=s3cret")[0], 200)
        finally:
            h.shutdown(); shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
