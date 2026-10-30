"""ScanService: live-frame processing on top of the existing WallMonitor pipeline."""
import base64, hashlib, html, json, math, threading, time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from . import __version__
from .common import Calib, Config, build_calib
from .baselines import _small_L
from .pipeline import WallMonitor, load_pixel_model
from .quality import assess_quality
from .walldetect import detect_wall_quad, calibration_quality, scale_check, cm_per_cam_px
from .crosswall import descriptor, compare
from .backtrace import forecast_growth, load_layout
from .store import Store, WALL_RE
from .common import order_corners
from .thermal import register, thermal_cue
from .autoscale import detect_markers, measure_quad, estimate_aspect

ROOT = Path(__file__).resolve().parents[1]
ALIGNED = ("dry", "earlier")                       # pixel-aligned same-wall references
CATEGORIES = ("dry", "earlier", "known_damp", "known_leak_surface", "other_wall", "thermal")
CATEGORY_TEXT = {"none": "No significant visual change detected", "change": "Visual change detected",
                 "potential": "Potential moisture-related pattern detected", "inspect": "Additional inspection recommended",
                 "quality": "Insufficient image quality for assessment"}
LIMITS = ["Detects surface appearance changes only; hidden leaks that have not reached the surface are not seen.",
          "Reported positions are of the changed/damp surface region, NOT a confirmed leak source or pipe-break point.",
          "Colour/darkness/thermal changes have non-leak causes (shadows, stains, paint, condensation, heating); a person must verify.",
          "Physical measurements are valid only for the calibrated wall plane and the stated calibration quality.",
          "Scores are relative evidence strengths, not calibrated probabilities of leakage."]


class ApiError(Exception):
    def __init__(self, status, msg):
        super().__init__(msg); self.status, self.msg = status, msg


@dataclass
class ScannerConfig:
    data_dir: str = "data"
    proc_width: int = 640
    map_ppm: int = 200
    max_frame_bytes: int = 6_000_000
    max_jobs: int = 2
    result_log_interval_s: float = 30.0
    max_snapshots: int = 50
    fuse_frames: int = 3
    idle_interval_ms: int = 1500
    active_interval_ms: int = 600
    retention_days: int = 30
    allow_sim_time: bool = False
    token: str = ""
    persist: int = 3


def model_versions():
    out = {"evdip": __version__}
    for n in ("pixel_clf", "event_clf"):
        p = ROOT / "models" / f"{n}.joblib"
        out[n] = hashlib.sha1(p.read_bytes()).hexdigest()[:8] if p.exists() else None
    return out


def _b64(img, q=75, w=None):
    if img is None:
        return None
    if w and img.shape[1] > w:
        img = cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


class Runtime:
    def __init__(self):
        self.monitor = None; self.cal = None; self.proc = None; self.t0 = time.time()
        self.frames = deque(maxlen=3); self.last_jpeg = None; self.last_result = None
        self.thermal = None; self.lock = threading.Lock(); self.last_log = 0.0
        self.descs = {}; self.n_frames = 0; self.persist_count = {}; self.logged_events = set(); self.lat = deque(maxlen=30)


class ScanService:
    def __init__(self, cfg=None):
        self.cfg = cfg or ScannerConfig()
        self.store = Store(self.cfg.data_dir); self.rt = {}
        self.sem = threading.BoundedSemaphore(self.cfg.max_jobs)
        self.pixel_model = load_pixel_model(str(ROOT / "models" / "pixel_clf.joblib"))
        self.versions = model_versions()

    # ---------------------------------------------------------------- helpers
    def _sess(self, sid):
        s = self.store.get_session(sid)
        if not s:
            raise ApiError(404, "unknown session")
        return s

    def _rt(self, sid):
        if sid not in self.rt:
            self.rt[sid] = Runtime()
            s = self._sess(sid)
            if s["calib"]:
                self.rt[sid].proc = tuple(s["calib"]["image_size"]); self.rt[sid].cal = Calib(s["calib"])
        return self.rt[sid]

    def decode(self, data, rt=None):
        if not data:
            raise ApiError(400, "empty body")
        if len(data) > self.cfg.max_frame_bytes:
            raise ApiError(413, f"frame larger than {self.cfg.max_frame_bytes} bytes")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ApiError(400, "not a valid JPEG/PNG image")
        h, w = img.shape[:2]
        if w < 160 or h < 120 or w > 8000 or h > 8000:
            raise ApiError(400, "image dimensions out of range (160x120 .. 8000x8000)")
        if rt is not None and rt.proc:
            pw, ph = rt.proc
            if abs(w / h - pw / ph) > 0.02 * (pw / ph):
                raise ApiError(409, "frame orientation/aspect changed since calibration - recalibrate")
            if (w, h) != (pw, ph):
                img = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
        elif w > self.cfg.proc_width:
            img = cv2.resize(img, (self.cfg.proc_width, int(h * self.cfg.proc_width / w)), interpolation=cv2.INTER_AREA)
        return img

    # --------------------------------------------------------------- sessions
    def create_session(self, body):
        meta = {k: body.get(k) for k in ("notes", "ambient_c", "humidity", "lighting", "device") if body.get(k) is not None}
        for k in ("ambient_c", "humidity"):
            if k in meta:
                try: meta[k] = float(meta[k])
                except (TypeError, ValueError): raise ApiError(400, f"{k} must be a number")
        try:
            sid = self.store.create_session(str(body.get("wall_id", "")), str(body.get("camera_type", "unknown")), meta)
        except ValueError as e:
            raise ApiError(400, str(e))
        return {"session_id": sid}

    def session_detail(self, sid):
        s = self._sess(sid); rt = self._rt(sid)
        s["references"] = [{k: v for k, v in r.items() if k != "path"} for r in self.store.list_refs(sid)]
        s["wall_history_refs"] = [{k: v for k, v in r.items() if k != "path"} for r in self.store.list_refs(wall_id=s["wall_id"])][:20]
        s["monitoring"] = rt.monitor is not None; s["model_versions"] = self.versions
        s["snapshots"] = self.store.list_snapshots(sid); s["last_result"] = rt.last_result
        return s

    def detect_wall(self, sid, data):
        self._sess(sid); img = self.decode(data)
        r = detect_wall_quad(img); r["quality"] = assess_quality(img); return r

    def decode_hires(self, data):
        if not data or len(data) > self.cfg.max_frame_bytes:
            raise ApiError(413 if data else 400, "empty or oversized image")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None or min(img.shape[:2]) < 240 or max(img.shape[:2]) > 8000:
            raise ApiError(400, "invalid image (needs >= 240 px, <= 8000 px)")
        return img

    def auto_calibrate(self, sid, data, marker_cm=15.0, apply=True, select_area=False):
        """Measure wall width/height automatically: find the scale marker + the wall outline (or the user's area)."""
        s = self._sess(sid); rt = self._rt(sid)
        if not (3.0 <= marker_cm <= 100.0):
            raise ApiError(400, "marker_cm must be 3..100")
        img = self.decode_hires(data); H, W = img.shape[:2]
        q = assess_quality(img); marks = detect_markers(img)
        wall = {"found": False, "corners_norm": None, "score": 0.0} if select_area else detect_wall_quad(img)
        out = {"image_size": [W, H], "quality": q, "markers_found": len(marks), "wall_found": bool(wall["found"]), "wall_score": wall["score"],
               "corners_norm": wall["corners_norm"], "warnings": list(q["flags"]), "scale_found": False}
        if not marks:
            out.update(status="needs_scale", message="No scale marker found. Print and stick the EchoVein marker flat on the wall (python make_marker.py), keep it fully visible, then retry. "
                       "Without it only relative units (% of wall height) are possible.")
            if wall["found"]:
                out["aspect_estimate"] = estimate_aspect(np.array(wall["corners_norm"]) * [W, H], W, H)
            return out
        m = marks[0]; side = m["side_px"] * self.cfg.proc_width / W if W > self.cfg.proc_width else m["side_px"]
        ref = {"marker_corners_norm": (m["corners"] / [W, H]).tolist(), "marker_cm": float(marker_cm), "img_size": [W, H], "marker_side_px": m["side_px"], "n_markers": len(marks), "time": time.time()}
        s["meta"]["scale_ref"] = ref; self.store.set_meta(sid, s["meta"])
        out.update(scale_found=True, marker={"id": m["id"], "side_px": m["side_px"], "size_cm": marker_cm})
        if m["side_px"] < 25:
            out["warnings"].append(f"marker is small in the image ({m['side_px']:.0f}px): move closer or use a bigger marker for accuracy")
        if not wall["found"] or wall["score"] < 0.2:
            out.update(status="needs_area", message="Marker found, but the wall edges are not clear. Drag the 4 corners around the wall (or just the area you want to measure); its width and height are then calculated automatically.")
            return out
        r = measure_quad(wall["corners_norm"], ref["marker_corners_norm"], marker_cm, W, H)
        if r is None:
            out.update(status="needs_area", message="Could not compute size from this view: adjust the corners manually."); return out
        out.update(width_cm=r["width_cm"], height_cm=r["height_cm"], uncertainty_cm=r["uncertainty_cm"]); out["warnings"] += r["warnings"]
        if wall["score"] < 0.35:
            out["warnings"].append("wall outline detection is uncertain: check that the corners sit on the real wall corners")
        if apply and not r["warnings"]:
            cal = self.calibrate(sid, {"corners_norm": wall["corners_norm"], "frame_w": W, "frame_h": H, "auto": True})
            out.update(status="calibrated", calibration=cal["calibration"], monitoring=cal["monitoring"], units="cm")
        else:
            out.update(status="proposal", message="Check/adjust the corners, then press Calibrate.")
        return out

    def calibrate(self, sid, body):
        s = self._sess(sid); rt = self._rt(sid)
        try:
            cn = np.array(body["corners_norm"], float).reshape(4, 2)
            fw, fh = float(body["frame_w"]), float(body["frame_h"])
        except Exception:
            raise ApiError(400, "need corners_norm (4x2), frame_w, frame_h")
        if not np.isfinite(cn).all() or not (cn >= -0.01).all() or not (cn <= 1.01).all() or fw < 160 or fh < 120:
            raise ApiError(400, "invalid corners or frame size")
        pw = int(min(self.cfg.proc_width, fw)); ph = int(round(pw * fh / fw)); corners = cn * [pw, ph]
        units, method, unc, extra_warn = "cm", "manual", None, []
        if cv2.contourArea(order_corners(corners).astype(np.float32)) < 0.01 * pw * ph:
            raise ApiError(400, "selected area is too small or degenerate: drag the 4 corners around the wall / area")
        if ("wall_w_cm" in body or "wall_h_cm" in body) and not body.get("auto"):
            try:
                if not (float(body.get("wall_w_cm", 0)) > 0 and float(body.get("wall_h_cm", 0)) > 0): raise ValueError
            except (TypeError, ValueError):
                raise ApiError(400, "manual dimensions must be positive numbers (or leave them empty to measure automatically)")
        if body.get("auto"):
            ref = s["meta"].get("scale_ref")
            if not ref:
                raise ApiError(409, "no scale marker measured yet: run Auto-calibrate with the marker visible (or enter dimensions manually / use relative mode)")
            r = measure_quad(cn, ref["marker_corners_norm"], ref["marker_cm"], *ref["img_size"])
            if r is None:
                raise ApiError(422, "cannot compute size from these corners")
            wc, hc, unc, method, extra_warn = r["width_cm"], r["height_cm"], r["uncertainty_cm"], "marker", r["warnings"]
        elif body.get("wall_w_cm") and body.get("wall_h_cm"):
            try: wc, hc = float(body["wall_w_cm"]), float(body["wall_h_cm"])
            except (TypeError, ValueError): raise ApiError(400, "dimensions must be numbers")
        else:                                                    # no scale at all -> relative units
            asp = estimate_aspect(corners, pw, ph)
            if not np.isfinite(asp) or not 0.1 < asp < 10:
                raise ApiError(422, "cannot estimate the wall shape from these corners")
            wc, hc, units, method = 100.0 * asp, 100.0, "relative", "relative"
            extra_warn = ["NO scale reference: positions/sizes are relative (% of wall height), not centimetres. Use the marker for real units."]
        if not (10 <= wc <= 5000 and 10 <= hc <= 5000) and units == "cm":
            raise ApiError(400, "invalid wall size (10..5000 cm)")
        ppm = int(min(self.cfg.map_ppm, 800 / (max(wc, hc) / 100.0)))
        err = None; ro = body.get("reference_object")
        q = calibration_quality(corners, (pw, ph), wc, hc)
        if q["level"] == "invalid":
            raise ApiError(422, q["warnings"][0])
        c = build_calib(corners, wc / 100.0, hc / 100.0, ppm, [pw, ph])
        if ro:
            try:
                p1, p2 = np.array(ro["p1_norm"], float) * [pw, ph], np.array(ro["p2_norm"], float) * [pw, ph]
                sc = scale_check(Calib(c), p1, p2, float(ro["length_cm"])); err = sc["error_pct"]
                q = calibration_quality(corners, (pw, ph), wc, hc, err); q["scale_check"] = sc
            except Exception:
                raise ApiError(400, "reference_object needs p1_norm, p2_norm, length_cm")
        q["warnings"] = q["warnings"] + extra_warn
        if unc:
            rel = max(unc[0] / wc, unc[1] / hc)
            if rel > 0.05:
                q["warnings"].append(f"size uncertainty ~{rel*100:.0f}%: use a bigger marker / move closer"); q["score"] = max(0.0, q["score"] - 0.2)
            q["rel_scale_err"] = rel
        if units == "relative":
            q["score"] = min(q["score"], 0.55)
        q["level"] = "good" if q["score"] >= 0.75 else "fair" if q["score"] >= 0.5 else "poor"
        c.update(quality=q, created=time.time(), wall_w_cm=wc, wall_h_cm=hc, units=units, dims_method=method, dims_uncertainty_cm=unc)
        self.store.set_calib(sid, c)
        rt.proc, rt.cal = (pw, ph), Calib(c); rt.monitor = None; rt.descs = {}
        self._maybe_build(sid)
        return {"calibration": {k: c[k] for k in ("wall_w_cm", "wall_h_cm", "map_size", "px_per_m", "image_size", "quality", "units", "dims_method", "dims_uncertainty_cm")},
                "monitoring": rt.monitor is not None}

    def scale_check(self, sid, body):
        rt = self._rt(sid)
        if rt.cal is None:
            raise ApiError(409, "calibrate first")
        try:
            pw, ph = rt.proc
            return scale_check(rt.cal, np.array(body["p1_norm"], float) * [pw, ph], np.array(body["p2_norm"], float) * [pw, ph], float(body["length_cm"]))
        except (KeyError, ValueError, TypeError):
            raise ApiError(400, "need p1_norm, p2_norm, length_cm")

    def set_layout(self, sid, body):
        s = self._sess(sid)
        try:
            lay = {"pipes": [{"name": str(p["name"])[:40], "points": [[float(x), float(y)] for x, y in p["points"]], "start_m": float(p.get("start_m", 0))} for p in body.get("pipes", [])],
                   "assets": [{"name": str(a["name"])[:40], "x": float(a["x"]), "y": float(a["y"]), "type": str(a.get("type", "other"))[:20]} for a in body.get("assets", [])]}
        except Exception:
            raise ApiError(400, "layout: pipes[{name,points[[x,y]..]}], assets[{name,x,y,type}] in cm on the wall")
        s["meta"]["layout"] = lay; self.store.set_meta(sid, s["meta"]); self._rt(sid).monitor = None; self._maybe_build(sid)
        return {"ok": True}

    # ------------------------------------------------------------- references
    def add_reference(self, sid, category, notes, data, force=False):
        s = self._sess(sid); rt = self._rt(sid)
        if category not in CATEGORIES:
            raise ApiError(400, f"category must be one of {CATEGORIES}")
        img = self.decode(data, rt); q = assess_quality(img)
        if category in ALIGNED and not q["ok"] and not force:
            raise ApiError(422, "reference image quality insufficient: " + "; ".join(q["flags"]))
        if category in ALIGNED and rt.cal is None:
            raise ApiError(409, "calibrate the wall before capturing a dry/earlier reference")
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        meta = {"notes": str(notes)[:300], "camera_type": s["camera_type"], "resolution": [img.shape[1], img.shape[0]],
                "calibration": (s["calib"] or {}).get("quality"), "lighting": s["meta"].get("lighting"), "ambient_c": s["meta"].get("ambient_c"),
                "humidity": s["meta"].get("humidity"), "quality": q["score"], "condition_note": str(notes)[:300]}
        rid = self.store.add_ref(sid, s["wall_id"], category, jpg.tobytes(), meta)
        rt.descs = {}
        if category in ALIGNED:
            rt.monitor = None; self._maybe_build(sid)
        return {"reference_id": rid, "category": category, "quality": q, "monitoring": rt.monitor is not None}

    def _aligned_refs(self, sid):
        out = []
        for r in self.store.list_refs(sid):
            if r["category"] in ALIGNED:
                img = cv2.imread(r["path"])
                if img is not None:
                    out.append((r["id"], img))
        return out

    def _maybe_build(self, sid):
        rt = self._rt(sid)
        if rt.cal is None:
            return
        refs = [(i, cv2.resize(im, rt.proc, interpolation=cv2.INTER_AREA) if (im.shape[1], im.shape[0]) != rt.proc else im) for i, im in self._aligned_refs(sid)]
        if not refs:
            return
        lay = load_layout(None); s = self._sess(sid)
        if s["meta"].get("layout"):
            lay = s["meta"]["layout"]
        cfg = Config(fusion="ml", classifier="rules", persist=self.cfg.persist, max_state="SUSPECTED", privacy=True)
        rt.monitor = WallMonitor(sid, rt.cal, refs, layout=lay, cfg=cfg, pixel_model=self.pixel_model, event_model_path=str(ROOT / "models" / "event_clf.joblib"))
        rt.t0 = time.time(); rt.persist_count = {}

    # ---------------------------------------------------------------- thermal
    def set_thermal(self, sid, body):
        s = self._sess(sid); rt = self._rt(sid)
        try:
            arr = np.array(body["temps"], np.float32)
        except Exception:
            raise ApiError(400, "temps must be a 2-D array of numbers")
        units = str(body.get("units", "C"))
        if arr.ndim != 2 or min(arr.shape) < 4 or max(arr.shape) > 1024 or not np.isfinite(arr).all():
            raise ApiError(400, "thermal array must be finite, 2-D, between 4x4 and 1024x1024")
        if units == "C" and (arr.min() < -40 or arr.max() > 300):
            raise ApiError(400, "temperatures outside -40..300 C: check units")
        if rt.cal is None:
            raise ApiError(409, "calibrate first (thermal map is registered to the wall map)")
        corners = body.get("corners")
        if corners:
            tm = register(arr, np.array(corners, np.float32), (rt.cal.W, rt.cal.Hpx))
            reg = "registered with supplied corners"
        else:
            tm = cv2.resize(arr, (rt.cal.W, rt.cal.Hpx)); reg = "ASSUMED to cover exactly the calibrated wall (no corners supplied)"
        rt.thermal = {"map": tm, "units": units, "device": str(body.get("device", ""))[:60], "registration": reg, "time": time.time()}
        np.save(self.store.sdir(sid) / "thermal_latest.npy", arr)
        if rt.monitor is not None:
            rt.monitor.thermal_cue = thermal_cue(tm)
        return {"ok": True, "registration": reg, "min": float(arr.min()), "max": float(arr.max()), "units": units}

    # ------------------------------------------------------------------ frame
    def process_frame(self, sid, data, want_images=False, sim_t=None):
        if not self.sem.acquire(blocking=False):
            raise ApiError(429, "processor busy - drop frame")
        try:
            rt = self._rt(sid)
            with rt.lock:
                return self._process(sid, rt, data, want_images, sim_t)
        finally:
            self.sem.release()

    def _process(self, sid, rt, data, want_images, sim_t):
        t_start = time.perf_counter()
        s = self._sess(sid)
        frame = self.decode(data, rt)
        rt.last_jpeg = data; rt.n_frames += 1
        q = assess_quality(frame)
        calq = (s["calib"] or {}).get("quality")
        res = {"scan_id": sid, "wall_id": s["wall_id"], "timestamp": time.time(), "camera_type": s["camera_type"], "quality": q,
               "calibrated": rt.cal is not None, "model_versions": self.versions, "limitations": LIMITS, "warnings": [], "regions": []}
        if calq:
            res["wall"] = {"width_cm": s["calib"]["wall_w_cm"], "height_cm": s["calib"]["wall_h_cm"], "calibration": calq, "units": s["calib"].get("units", "cm"),
                           "dims_method": s["calib"].get("dims_method", "manual"), "dims_uncertainty_cm": s["calib"].get("dims_uncertainty_cm")}
            res["units"] = s["calib"].get("units", "cm")
            res["warnings"] += calq.get("warnings", [])
        # frame fusion for stability (only while the scene is static)
        use = frame
        rt.frames.append(frame)
        if self.cfg.fuse_frames >= 3 and len(rt.frames) == 3:
            d = [float(np.mean(cv2.absdiff(rt.frames[i], rt.frames[i + 1]))) for i in range(2)]
            if max(d) < 3.0:
                use = np.median(np.stack(list(rt.frames)), axis=0).astype(np.uint8)
        if not q["ok"]:
            res.update(category="quality", mode="skipped", reference_status="not evaluated", warnings=res["warnings"] + q["flags"])
            return self._finish(sid, rt, res, t_start, None, want_images)
        if rt.monitor is None:
            res["mode"] = "standalone"; res["reference_status"] = "none: capture a dry reference to enable comparison"
            res["warnings"].append("Standalone inspection: no dry reference/calibration - low reliability, no wall coordinates." if rt.cal is None else
                                   "Standalone inspection: no dry reference captured yet.")
            res["regions"] = self._standalone(use, rt)
            res["category"] = "change" if res["regions"] else "none"
            return self._finish(sid, rt, res, t_start, None, want_images)
        m = rt.monitor
        t = float(sim_t) if (sim_t is not None and self.cfg.allow_sim_time) else (time.time() - rt.t0) / 3600.0
        m.analyze(use, t); snap = m.finalize(0.0)
        info = snap["info"]; res["mode"] = "reference-comparison"
        res["warnings"] += info.get("warnings", [])
        mis = self._ref_mismatch(m)
        res["reference_status"] = f"compared with reference '{snap['baseline']}'"
        res["regions"] = self._regions(sid, rt, m, snap, s)
        res["_snap"] = snap; res["_mis"] = mis
        res["category"] = self._category(res, snap)
        return self._finish(sid, rt, res, t_start, m, want_images)

    def _ref_mismatch(self, m):
        try:
            return float(m.lib.mismatch(0 if len(m.lib) == 1 else m.lib.select(m._cur)[0], _small_L(m._cur)))
        except Exception:
            return 0.0

    def _standalone(self, img, rt):
        """No reference: darkness relative to the wall's own smooth background (low reliability)."""
        sm = cv2.resize(img, None, fx=0.5, fy=0.5)
        L = cv2.cvtColor(sm, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32); Lb = cv2.GaussianBlur(L, (0, 0), 3)
        bg = cv2.GaussianBlur(L, (0, 0), 40)
        mask = ((Lb / (bg + 1) < 0.85)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(mask); out = []
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] > 0.004 * mask.size:
                x, y, w, h = st[i, :4]
                out.append({"id": i, "state": "WATCH", "label": "unreferenced darker area", "reliability": "low",
                            "polygon": [[x / mask.shape[1], y / mask.shape[0]], [(x + w) / mask.shape[1], y / mask.shape[0]],
                                        [(x + w) / mask.shape[1], (y + h) / mask.shape[0]], [x / mask.shape[1], (y + h) / mask.shape[0]]]})
        return out[:8]

    def _regions(self, sid, rt, m, snap, s):
        cal = rt.cal; ppc = cal.px_per_cm; pw, ph = rt.proc; out = []
        cq = (s["calib"] or {}).get("quality", {}); cerr = cq.get("corner_err_px", 3.0)
        tm = rt.thermal["map"] if rt.thermal else None
        for tr in m.tracker.tracks:
            if not tr.active or not tr.last()["present"] or tr.mask is None:
                continue
            d = next((x for x in snap["tracks"] if x["id"] == tr.id), None)
            if d is None:
                continue
            bx, by, bw, bh = cv2.boundingRect(tr.mask)
            cnts, _ = cv2.findContours(tr.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polys = []
            for c in cnts:
                c = cv2.approxPolyDP(c, 2.0, True).reshape(-1, 2)
                if len(c) >= 3:
                    p = cal.map_to_image(c, m._T); polys.append((p / [pw, ph]).clip(0, 1).tolist())
            cx, cy = d["x_cm"], d["y_cm"]
            cam = cal.map_to_image([[cx * ppc, cy * ppc]], m._T)[0]
            span_s = (tr.hist[-1]["t"] - tr.hist[0]["t"]) * 3600.0
            long_obs = span_s >= 600
            n_obs = len(tr.present_hist)
            recur = self.store.recurring_count(s["wall_id"], cx, cy, 15.0, sid) if tr.state in ("SUSPECTED", "ESCALATED") else 0
            reg = {"id": tr.id, "state": d["state"], "label": tr.label if long_obs else "unclassified (needs longer observation)",
                   "reason": d["reason"], "observations": n_obs, "observed_seconds": span_s,
                   "pixel_xy": [float(cam[0]), float(cam[1])], "frame_size": [pw, ph],
                   "x_cm": cx, "y_cm": cy, "left_cm": d["left_cm"], "right_cm": d["right_cm"], "top_cm": d["top_cm"], "bottom_cm": d["bottom_cm"],
                   "width_cm": bw / ppc, "height_cm": bh / ppc, "area_cm2": d["area_cm2"],
                   "uncertainty_cm": float(np.hypot(cm_per_cam_px(cal, cx * ppc, cy * ppc, m._T) * max(cerr, 1.0) + 0.5 / ppc,
                                                    max(min(d["left_cm"], d["right_cm"]), min(d["top_cm"], d["bottom_cm"])) * cq.get("rel_scale_err", 0.0))),
                   "units": (s["calib"] or {}).get("units", "cm"),
                   "polygons": polys, "localization_level": "surface change region (not a confirmed leak source)",
                   "severity_relative": d["severity"] if long_obs else None,
                   "growth_cm_per_h": d["growth_cm_h"] if long_obs else None,
                   "source_zone": d["source"] if long_obs else None,
                   "pipes": d["pipes"][:2] if long_obs else [], "assets": d["assets"],
                   "forecast": forecast_growth(tr), "recurring_previous_sessions": recur,
                   "ai_score": {"value": float(tr.last()["score"]), "meaning": "mean pixel-classifier/fusion score in region; relative, uncalibrated"},
                   "cues": {"darkening": tr.last()["dark"], "texture_loss": tr.last()["tex"], "saturation_change": tr.last()["sat"]}}
            if tm is not None:
                sel = tr.mask > 0
                vals = tm[sel]
                if vals.size:
                    reg["thermal"] = {"mean": float(vals.mean()), "min": float(vals.min()), "max": float(vals.max()), "units": rt.thermal["units"],
                                      "delta_vs_wall_median": float(vals.mean() - np.median(tm)),
                                      "note": "thermal anomaly is not proof of leakage (ambient, insulation, airflow, sun)"}
            if tr.state in ("SUSPECTED", "ESCALATED") and (tr.id, tr.state) not in rt.logged_events:
                rt.logged_events.add((tr.id, tr.state)); self.store.add_event(sid, s["wall_id"], cx, cy, tr.state, tr.label)
            out.append(reg)
        return out

    def _category(self, res, snap):
        regs = res["regions"]
        if not regs:
            return "none"
        if not any(r["state"] in ("SUSPECTED", "ESCALATED", "VERIFIED") for r in regs):
            return "change"
        strong = any(r["observations"] >= 10 or (r["severity_relative"] or 0) > 40 or r["recurring_previous_sessions"] > 0 or
                     any(a["electrical"] and a["status"] in ("reached", "approaching") for a in r["assets"]) for r in regs
                     if r["state"] in ("SUSPECTED", "ESCALATED"))
        return "inspect" if strong else "potential"

    def _finish(self, sid, rt, res, t_start, m, want_images):
        snap = res.pop("_snap", None); mis = res.pop("_mis", 0.0)
        regs = res["regions"]; q = res["quality"]
        res["warnings"] = list(dict.fromkeys(res["warnings"]))
        env = self._sess(sid)["meta"]
        if env.get("humidity") is not None and env["humidity"] > 70 and regs:
            res["warnings"].append(f"High humidity ({env['humidity']:.0f}%): condensation can mimic damp patches.")
        if mis > 0.10:
            res["warnings"].append("Current lighting/appearance differs from the stored references (data drift): capture a new reference.")
        det = max([r.get("ai_score", {}).get("value", 0) for r in regs], default=0.0) if m else None
        ai_on = bool(m and m.cfg.fusion == "ml")
        res["indicators"] = {
            "image_quality": {"score": q["score"], "ok": q["ok"], "flags": q["flags"]},
            "calibration_quality": ((res.get("wall") or {}).get("calibration") or {"level": "none"}),
            "reference_reliability": ({"score": float(np.clip(1 - mis / 0.15, 0, 1)), "mismatch": mis, "note": "how well a stored reference explains current lighting"} if m else {"available": False}),
            "dip_detection_strength": ({"value": det, "note": "relative strength of image-change evidence (not a probability)"} if m else {"available": False}),
            "ai_prediction": ({"value": det, "note": "pixel classifier trained on SIMULATED data only; uncalibrated score; validate on your walls"} if ai_on else {"available": False, "note": "model missing or disabled"}),
            "thermal_anomaly": ({"available": True, "device": rt.thermal["device"], "registration": rt.thermal["registration"]} if rt.thermal else {"available": False}),
            "evidence_completeness": {"visible_image": True, "calibration": rt.cal is not None, "dry_reference": m is not None, "thermal": rt.thermal is not None,
                                      "ai_model": ai_on, "temperature_humidity": env.get("humidity") is not None or env.get("ambient_c") is not None}}
        area = sum(r.get("area_cm2", 0.0) for r in regs); wall_area = (res["wall"]["width_cm"] * res["wall"]["height_cm"]) if res.get("wall") else None
        pen = []
        if wall_area:
            pen.append(("affected area", min(40.0, 400.0 * area / wall_area)))
        pen.append(("persistence/state", 30.0 if any(r["state"] == "ESCALATED" for r in regs) else 15.0 if any(r["state"] == "SUSPECTED" for r in regs) else 5.0 if regs else 0.0))
        pen.append(("recurring location", 10.0 if any(r.get("recurring_previous_sessions") for r in regs) else 0.0))
        res["wall_health"] = {"score": max(0.0, 100.0 - sum(p for _, p in pen)), "breakdown": [{"factor": f, "penalty": round(p, 1)} for f, p in pen],
                              "note": "heuristic monitoring score for explanation only; not validated"}
        res["category_text"] = CATEGORY_TEXT[res["category"]]
        res["next_interval_ms"] = self.cfg.active_interval_ms if (regs and q["ok"]) else self.cfg.idle_interval_ms
        lat = (time.perf_counter() - t_start) * 1000.0; rt.lat.append(lat)
        res["perf"] = {"latency_ms": lat, "mean_latency_ms": float(np.mean(rt.lat)), "frames_processed": rt.n_frames}
        if want_images and m is not None and hasattr(m, "images"):
            res["images"] = {"processed": _b64(m.images["map"], 70, 360), "mask": _b64(m.images["heat"], 70, 360), "reference_vs_now": _b64(m.images["evidence"], 70, 520)}
        res["thermal_present"] = rt.thermal is not None
        if m is not None:
            self._last_images = None
        rt.last_result = {k: v for k, v in res.items() if k != "images"}
        if time.time() - rt.last_log > self.cfg.result_log_interval_s:      # do not store every frame
            rt.last_log = time.time()
            self.store.add_result(sid, {"category": res["category"], "n_regions": len(regs), "quality": q["score"], "health": res["wall_health"]["score"],
                                        "regions": [{k: r.get(k) for k in ("id", "state", "x_cm", "y_cm", "area_cm2")} for r in regs]})
        return res

    # --------------------------------------------------------- feedback/snap
    def feedback(self, sid, track, verdict):
        rt = self._rt(sid)
        if verdict not in ("verified", "dismissed") or rt.monitor is None:
            raise ApiError(400, "verdict must be verified|dismissed and monitoring must be active")
        for tr in rt.monitor.tracker.tracks:
            if tr.id == track:
                tr.verdict = verdict; rt.monitor._update_state(tr)
                return {"ok": True, "state": tr.state}
        raise ApiError(404, "unknown region")

    def snapshot(self, sid):
        rt = self._rt(sid)
        if rt.last_jpeg is None or rt.last_result is None:
            raise ApiError(409, "no frame processed yet")
        files = {"frame": ("jpg", rt.last_jpeg)}
        m = rt.monitor
        if m is not None and hasattr(m, "images"):
            for k in ("camera", "map", "heat"):
                ok, b = cv2.imencode(".jpg", m.images[k], [cv2.IMWRITE_JPEG_QUALITY, 88]); files[k] = ("jpg", b.tobytes())
        nid = self.store.add_snapshot(sid, files, rt.last_result, self.cfg.max_snapshots)
        return {"snapshot_id": nid}

    def results(self, sid):
        self._sess(sid); return {"results": self.store.list_results(sid)}

    def delete(self, sid):
        self.rt.pop(sid, None)
        if not self.store.delete_session(sid):
            raise ApiError(404, "unknown session")
        return {"deleted": sid}

    # ------------------------------------------------------------------ report
    def report_html(self, sid):
        s = self._sess(sid); snap = self.store.last_snapshot(sid)
        if not snap:
            raise ApiError(409, "no snapshot saved yet: press 'Save snapshot' first")
        r = snap["result"]; e = html.escape
        def img(k, cap):
            p = snap["files"].get(k)
            if p and Path(p).exists():
                return f"<figure><img src='data:image/jpeg;base64,{base64.b64encode(Path(p).read_bytes()).decode()}' style='max-width:100%'><figcaption>{e(cap)}</figcaption></figure>"
            return ""
        c = s["calib"] or {}; q = c.get("quality", {})
        regs = "".join(f"<tr><td>{g['id']}</td><td>{e(g['state'])}</td><td>{e(str(g['label']))}</td><td>{g['left_cm']:.0f}</td><td>{g['right_cm']:.0f}</td><td>{g['top_cm']:.0f}</td><td>{g['bottom_cm']:.0f}</td>"
                       f"<td>{g['width_cm']:.0f} x {g['height_cm']:.0f}</td><td>{g['area_cm2']:.0f}</td><td>&plusmn;{g['uncertainty_cm']:.1f}</td></tr>" for g in r.get("regions", []) if "x_cm" in g)
        refs = "".join(f"<tr><td>{e(x['category'])}</td><td>{time.strftime('%Y-%m-%d %H:%M', time.localtime(x['created']))}</td><td>{e(str(x['meta'].get('notes', '')))}</td></tr>" for x in self.store.list_refs(sid))
        ind = r.get("indicators", {})
        return f"""<!doctype html><meta charset=utf-8><title>EchoVein report {e(sid)}</title>
<style>body{{font:14px system-ui;max-width:900px;margin:20px auto;padding:0 12px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #8886;padding:4px 6px;text-align:left}}.box{{background:#f90;color:#000;padding:8px;border-radius:6px}}</style>
<h1>EchoVein-DIP scan report</h1><p>Wall <b>{e(s['wall_id'])}</b> · scan <b>{e(sid)}</b> · camera {e(s['camera_type'])} · snapshot {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(snap['created']))}<br>
Models: {e(json.dumps(r.get('model_versions', {})))}</p>
<h2>Summary</h2><p><b>{e(r.get('category_text', ''))}</b> — mode: {e(str(r.get('mode')))}; {e(str(r.get('reference_status')))}</p>
<h2>Calibration</h2><p>Wall {c.get('wall_w_cm', '?')} x {c.get('wall_h_cm', '?')} cm; quality {e(str(q.get('level')))} ({q.get('score', 0):.2f}); corner uncertainty ~{q.get('corner_err_px', 0):.1f}px.<br>{e('; '.join(q.get('warnings', [])))}</p>
<h2>Regions (cm from wall edges, surface-change locations only)</h2><table><tr><th>ID</th><th>State</th><th>Label</th><th>Left</th><th>Right</th><th>Top</th><th>Bottom</th><th>W x H</th><th>Area cm²</th><th>Unc.</th></tr>{regs or '<tr><td colspan=10>none</td></tr>'}</table>
<h2>Images</h2>{img('frame', 'Original frame')}{img('camera', 'Detections on camera view (privacy-masked)')}{img('map', 'Rectified wall map')}{img('heat', 'Evidence heat-map')}
<h2>References</h2><table><tr><th>Category</th><th>Captured</th><th>Notes</th></tr>{refs}</table>
<h2>Reliability indicators (separate, not combined)</h2><pre>{e(json.dumps(ind, indent=1, default=float))}</pre>
<h2>Warnings</h2><ul>{''.join(f'<li>{e(w)}</li>' for w in r.get('warnings', []))}</ul>
<h2>Limitations</h2><div class=box><ul>{''.join(f'<li>{e(w)}</li>' for w in LIMITS)}</ul></div>"""
