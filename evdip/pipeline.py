"""WallMonitor: full per-wall pipeline.  analyze() -> finalize() so a Fleet can add cross-wall context."""
import os, time
import cv2
import numpy as np
from .common import Calib, Config
from .baselines import BaselineLibrary
from .cues import cue_maps, fuse_weighted, feature_stack
from .occlusion import OcclusionDetector, privacy_blur
from .health import CameraHealth
from .tracker import RegionTracker
from .eventclass import EventClassifier, track_features
from .backtrace import estimate_source, pipe_prior, assets_at_risk, severity_index, growth_rate, load_layout
from .evidence import Evidence

ORDER = ["NORMAL", "WATCH", "SUSPECTED", "ESCALATED", "VERIFIED"]


def load_pixel_model(path="models/pixel_clf.joblib"):
    if os.path.exists(path):
        import joblib
        return joblib.load(path)
    return None


class WallMonitor:
    def __init__(self, name, calib, baselines, ref_frame=None, layout=None, cfg=None,
                 pixel_model="auto", event_model_path="models/event_clf.joblib"):
        self.name, self.cfg = name, cfg or Config()
        self.cal = calib if isinstance(calib, Calib) else Calib(calib)
        self.lib = BaselineLibrary()
        for n, rect_or_frame in baselines:           # baselines given as raw camera frames
            self.lib.add(n, self.cal.rectify(rect_or_frame))
        self.layout = layout or {"pipes": [], "assets": []}
        self.pixel_model = load_pixel_model() if pixel_model == "auto" else pixel_model
        if self.cfg.fusion == "ml" and self.pixel_model is None:
            self.cfg.fusion = "weighted"
        self.clf = EventClassifier(self.cfg.classifier if self.cfg.use_classifier else "rules", event_model_path)
        self.tracker = RegionTracker(self.cal.px_per_cm)
        self.occ = OcclusionDetector() if self.cfg.use_occlusion else None
        first = baselines[0][1] if baselines else ref_frame
        self.health = CameraHealth(ref_frame if ref_frame is not None else first, self.cal.corners)
        self.wall_mask_cam = None
        self.t = 0.0
        self.anom_hist, self.events, self.series = [], [], []
        self.last = {}
        self.cross_frac = 0.0
        self.skipped = 0
        self._cur = None

    # ---------------------------------------------------------------- analyze
    def analyze(self, frame, t):
        cfg, cal = self.cfg, self.cal
        self.t = t
        if self.wall_mask_cam is None:
            self.wall_mask_cam = cal.wall_mask_cam(frame.shape)
        info = {"t": t, "warnings": [], "occluded": 0.0, "skipped": False, "baseline": None}
        # --- camera health / self-recalibration
        h = self.health.check(frame)
        T = None
        info["warnings"] += h["warnings"]; info["shift_px"] = h["shift_px"]; info["calibration_lost"] = h["calibration_lost"]
        if cfg.use_selfcal and h["T"] is not None and h["shift_px"] > self.health.tol and not h["calibration_lost"]:
            T = h["T"]
        if h["calibration_lost"] and cfg.use_selfcal:
            info["skipped"] = True
        # --- occlusion
        occ_rect = np.zeros((cal.Hpx, cal.W), np.uint8)
        occ_cam = None
        if self.occ is not None:
            occ_cam = self.occ.update(frame, self.wall_mask_cam)
            if T is not None:
                occ_cam = occ_cam  # mask is tiny shift-wise; fine
            occ_rect = cv2.warpPerspective(occ_cam, cal.H if T is None else cal.H @ T, (cal.W, cal.Hpx))
            occ_rect = cv2.dilate((occ_rect > 0).astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
        info["occluded"] = float((occ_rect > 0).mean())
        if info["occluded"] > 0.25:
            info["skipped"] = True; info["warnings"].append("wall occluded - analysis skipped")
        self._frame, self._T, self._occ_cam, self._info = frame, T, occ_cam, info
        cur = cal.rectify(frame, T)
        self._cur = cur
        if info["skipped"]:
            self.skipped += 1
            self._pending = None
            self.last.update({"info": info})
            return {"anomaly": 0.0, "skipped": True, "wall_tex": 0.0, "wall_sat": 0.0}
        # --- baseline selection
        if cfg.use_library and len(self.lib) > 1:
            bi, mis = self.lib.select(cur)
        else:
            bi, mis = 0, 0.0
        ref = self.lib.items[bi]["rect"]; info["baseline"] = self.lib.items[bi]["name"]; info["baseline_mismatch"] = mis
        # --- cues + fusion
        c = cue_maps(ref, cur, use_illum=cfg.use_illum, thermal=getattr(self, "thermal_cue", None))
        if cfg.fusion == "ml" and self.pixel_model is not None and cfg.use_multicue:
            F = feature_stack(c); sh = F.shape[:2]
            # per-channel resize: cv2.resize(INTER_AREA) rejects >4 channels unless the ratio is exactly 2 (odd map sizes!)
            small = np.dstack([cv2.resize(F[:, :, i], (sh[1] // 2, sh[0] // 2), interpolation=cv2.INTER_AREA) for i in range(F.shape[2])])
            p = self.pixel_model.predict_proba(small.reshape(-1, F.shape[2]))[:, 1].reshape(small.shape[:2])
            score = cv2.resize(p.astype(np.float32), (sh[1], sh[0]))
            score = cv2.GaussianBlur(score, (0, 0), 2.0)
        else:
            score = fuse_weighted(c, multicue=cfg.use_multicue)
        score[occ_rect > 0] = 0
        thr = cfg.thr * cfg.thr_scale
        m = (score > thr).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        ppc = cal.px_per_cm; comps = []
        for i in range(1, n):
            area = st[i, cv2.CC_STAT_AREA] / ppc ** 2
            if area < cfg.min_area_cm2:
                continue
            mk = (lab == i).astype(np.uint8) * 255; sel = lab == i
            comps.append(dict(mask=mk, area=float(area), cx=float(cen[i][0] / ppc), cy=float(cen[i][1] / ppc),
                              dark=float(c["dark"][sel].mean()), tex=float(c["tex"][sel].mean()),
                              sat=float(c["sat"][sel].mean()), score=float(score[sel].mean())))
        frac = float((m > 0).mean())
        anomaly = frac + 0.5 * max(c["wall_tex"], c["wall_sat"])
        self._pending = dict(comps=comps, score=score, cues=c, mask=m, ref=ref, anomaly=anomaly)
        return {"anomaly": anomaly, "skipped": False, "wall_tex": c["wall_tex"], "wall_sat": c["wall_sat"], "frac": frac}

    # --------------------------------------------------------------- finalize
    def finalize(self, cross_frac=0.0):
        cfg, cal = self.cfg, self.cal
        self.cross_frac = cross_frac
        info, t = self._info, self.t
        P = self._pending
        if P is None:                                    # skipped frame: keep state, no update
            self._render(None); return self.snapshot()
        comps, c = P["comps"], P["cues"]
        if cfg.use_temporal:
            tracks = self.tracker.update(comps, t, c["light_level"])
        else:                                            # ablation: every frame stands alone
            self.tracker = RegionTracker(cal.px_per_cm); tracks = self.tracker.update(comps, t, c["light_level"])
        self.anom_hist.append(P["anomaly"])
        for tr in tracks:
            self._classify(tr, c)
            self._update_state(tr)
        self._render(P)
        self.series.append({"t": t, "anomaly": P["anomaly"], "tracks": {tr.id: {"area": tr.last()["area"], "score": tr.last()["score"]} for tr in tracks}})
        return self.snapshot()

    def _classify(self, tr, c):
        f = track_features(tr, self.cal.wall_area_cm2, self.cross_frac, c["wall_tex"], c["wall_sat"])
        tr.feat = f
        if len(tr.present_hist) < 2 and tr.last()["present"]:
            tr.label, tr.probs = "unknown", {}
        else:
            tr.label, tr.probs = self.clf.predict(f)
        tr.extra["src"] = estimate_source(tr)
        tr.extra["pipes"] = pipe_prior(tr.extra["src"], self.layout)
        tr.extra["assets"] = assets_at_risk(tr, self.layout)
        tr.extra["severity"] = severity_index(tr, self.cal.wall_area_cm2)
        tr.extra["growth_cm_h"], _ = growth_rate(tr)

    def _update_state(self, tr):
        """NORMAL -> WATCH -> SUSPECTED -> ESCALATED -> VERIFIED (+ DISMISSED / RESOLVED / EXPLAINED)."""
        ph = tr.present_hist
        if tr.verdict == "dismissed":
            tr.state, tr.reason = "DISMISSED", "dismissed by a person"; return
        if tr.verdict == "verified":
            tr.state, tr.reason = "VERIFIED", "verified by a person"; return
        if not tr.active:
            tr.state, tr.reason = "RESOLVED", "no longer visible (dried / gone)"; return
        n = len(ph)
        if tr.label in ("artifact", "condensation"):
            tr.state = "WATCH"; tr.reason = f"explained as {tr.label}"; return
        if n < self.cfg.persist and self.cfg.use_temporal:
            tr.state, tr.reason = "WATCH", f"seen {n}x, need {self.cfg.persist}"; return
        tr.state, tr.reason = "SUSPECTED", "persistent damp region"
        sev, g = tr.extra.get("severity", 0), tr.extra.get("growth_cm_h", 0)
        elec = [a for a in tr.extra.get("assets", []) if a["electrical"] and a["status"] in ("reached", "approaching")
                and (a["t_min_h"] is None or a["t_min_h"] < 2.0)]
        if (tr.label == "leak" and n >= 3 and g > 1.0) or sev > 60 or elec:
            tr.state = "ESCALATED"
            tr.reason = "growing region classified as leak" if not elec else "electrical asset at risk"
            if self.cfg.max_state == "SUSPECTED":
                tr.state, tr.reason = "SUSPECTED", "persistent damp region (escalation disabled in live-scan mode)"
        if ORDER.index(tr.state) > ORDER.index(tr.peak_state):
            tr.peak_state = tr.state
            self.events.append({"t": t if (t := self.t) is not None else 0, "track": tr.id, "state": tr.state,
                                "label": tr.label, "reason": tr.reason})

    # ---------------------------------------------------------------- outputs
    def snapshot(self):
        out_tracks = []
        for tr in self.tracker.tracks:
            ph = tr.present_hist
            if not ph:
                continue
            l = tr.last()
            out_tracks.append({
                "id": tr.id, "state": tr.state, "peak_state": tr.peak_state, "label": tr.label,
                "probs": {k: round(v, 2) for k, v in tr.probs.items()}, "reason": tr.reason,
                "x_cm": ph[-1]["cx"], "y_cm": ph[-1]["cy"], "area_cm2": l["area"],
                "left_cm": ph[-1]["cx"], "right_cm": self.cal.wall_w_cm - ph[-1]["cx"],
                "top_cm": ph[-1]["cy"], "bottom_cm": self.cal.wall_h_cm - ph[-1]["cy"],
                "source": tr.extra.get("src"), "pipes": tr.extra.get("pipes", []), "assets": tr.extra.get("assets", []),
                "severity": tr.extra.get("severity", 0.0), "growth_cm_h": tr.extra.get("growth_cm_h", 0.0),
                "active": tr.active, "verdict": tr.verdict,
                "history": [{"t": x["t"], "area": x["area"], "score": x["score"]} for x in tr.hist]})
        status = "NORMAL"
        for tr in out_tracks:
            if tr["active"] or tr["peak_state"] in ("ESCALATED", "VERIFIED"):
                s = tr["state"]
                if s in ORDER and ORDER.index(s) > ORDER.index(status):
                    status = s
        z = self.robust_z()
        snap = {"wall": self.name, "t": self.t, "status": status, "tracks": out_tracks, "info": self._info,
                "baseline": self._info.get("baseline"), "anomaly": self.anom_hist[-1] if self.anom_hist else 0.0,
                "z": z, "cross_frac": self.cross_frac, "events": self.events[-30:], "cfg_thr_scale": self.cfg.thr_scale,
                "evidence": [Evidence("visible", self.name, tr["x_cm"], tr["y_cm"],
                                      min(1.0, 0.4 + tr["severity"] / 200 + 0.2 * (tr["state"] in ("SUSPECTED", "ESCALATED"))),
                                      self.t, {"label": tr["label"]}).to_dict() for tr in out_tracks if tr["active"]]}
        self.last = snap
        return snap

    def robust_z(self):
        """Wall's own anomaly vs its own history (median/MAD) - used before any peer comparison."""
        h = np.array(self.anom_hist[:-1]) if len(self.anom_hist) > 1 else np.array([])
        if len(h) < 8:
            return 0.0
        med = np.median(h); mad = np.median(np.abs(h - med)) * 1.4826 + 0.01
        return float((self.anom_hist[-1] - med) / mad)

    def _render(self, P):
        cal, cfg = self.cal, self.cfg
        frame = self._frame
        cam = frame.copy()
        if cfg.privacy:
            cam = privacy_blur(cam, self.wall_mask_cam, self._occ_cam)
        ref = P["ref"] if P else self.lib.items[0]["rect"]
        cur = self._cur.copy()
        vis = cur.copy()
        colors = {"WATCH": (0, 200, 255), "SUSPECTED": (0, 120, 255), "ESCALATED": (0, 0, 255), "VERIFIED": (255, 0, 255),
                  "RESOLVED": (120, 200, 120), "DISMISSED": (150, 150, 150)}
        ppc = cal.px_per_cm
        for tr in self.tracker.tracks:
            if not tr.active or tr.mask is None or not tr.last()["present"]:
                continue
            col = colors.get(tr.state, (255, 255, 255))
            cnts, _ = cv2.findContours(tr.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, col, 2)
            src = tr.extra.get("src")
            if src and tr.state in ("SUSPECTED", "ESCALATED", "VERIFIED"):
                ctr = (int(src["x_cm"] * ppc), int(src["y_cm"] * ppc))
                axes = (int(2 * src["sx_cm"] * ppc), int(2 * src["sy_cm"] * ppc))
                cv2.ellipse(vis, ctr, axes, 0, 0, 360, (255, 255, 0), 1)
                cv2.drawMarker(vis, ctr, (255, 255, 0), cv2.MARKER_TILTED_CROSS, 14, 2)
                cv2.putText(vis, "src", (ctr[0] + 6, ctr[1] - 6), 0, 0.45, (255, 255, 0), 1)
            ph = tr.present_hist[-1]
            cv2.putText(vis, f"#{tr.id} {tr.label} {tr.state}", (int(ph["cx"] * ppc) - 40, int(ph["cy"] * ppc)), 0, 0.45, col, 1)
        for pipe in self.layout.get("pipes", []):
            pts = (np.array(pipe["points"]) * ppc).astype(np.int32)
            cv2.polylines(vis, [pts], False, (255, 120, 0), 1)
        for a in self.layout.get("assets", []):
            cv2.circle(vis, (int(a["x"] * ppc), int(a["y"] * ppc)), 5, (0, 255, 255), -1)
        cv2.putText(vis, f"{self.name}  {self.last.get('status', '')}", (8, 18), 0, 0.55, (255, 255, 255), 2)
        # camera overlay (regions projected back to the photo)
        for tr in self.tracker.tracks:
            if tr.active and tr.mask is not None and tr.last()["present"]:
                cnts, _ = cv2.findContours(tr.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cn in cnts:
                    pts = cal.map_to_image(cn.reshape(-1, 2), self._T).astype(np.int32)
                    cv2.polylines(cam, [pts], True, colors.get(tr.state, (255, 255, 255)), 2)
        for w in self._info.get("warnings", [])[:2]:
            cv2.putText(cam, w[:60], (8, cam.shape[0] - 10 - 18 * self._info["warnings"].index(w)), 0, 0.45, (0, 200, 255), 1)
        heat = np.zeros_like(cur) if P is None else cv2.applyColorMap((np.clip(P["score"], 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        card = np.vstack([np.hstack([ref, cur]), np.hstack([heat, vis])])
        for (x, y, s) in [(8, 18, "dry baseline"), (cal.W + 8, 18, "now"), (8, cal.Hpx + 18, "evidence score"), (cal.W + 8, cal.Hpx + 18, "result")]:
            cv2.putText(card, s, (x, y), 0, 0.55, (255, 255, 255), 2); cv2.putText(card, s, (x, y), 0, 0.55, (0, 0, 0), 1)
        self.images = {"camera": cam, "map": vis, "evidence": card, "heat": heat}


class Fleet:
    """Several walls: per-wall robust statistics first, then cross-wall (condensation) evidence + ranking."""

    def __init__(self, monitors):
        self.mon = {m.name: m for m in monitors}
        self.ranking = []

    def step(self, frames, t):
        res = {n: self.mon[n].analyze(f, t) for n, f in frames.items() if n in self.mon}
        zs = {n: self.mon[n].robust_z() if self.mon[n].anom_hist else 0 for n in res}
        # cross-wall: fraction of walls showing a wide anomaly (high wall-level tex/sat shift or large wet fraction)
        wide = [n for n, r in res.items() if not r["skipped"] and (r["wall_tex"] > 0.12 or r.get("frac", 0) > 0.15)]
        cross = len(wide) / max(len(res), 1) if len(res) > 1 else 0.0
        snaps = {n: self.mon[n].finalize(cross) for n in res}
        rank = []
        for n, s in snaps.items():
            sev = max([tr["severity"] for tr in s["tracks"] if tr["active"]], default=0.0)
            score = ORDER.index(s["status"]) * 100 + sev + max(s["z"], 0)
            rank.append({"wall": n, "status": s["status"], "severity": sev, "z": s["z"], "score": score})
        self.ranking = sorted(rank, key=lambda r: -r["score"])
        return snaps

    def feedback(self, wall, track_id, verdict, feedback_csv=None):
        m = self.mon[wall]
        for tr in m.tracker.tracks:
            if tr.id == track_id:
                tr.verdict = verdict
                m._update_state(tr)
                # tune threshold: dismissed -> slightly less sensitive; verified -> slightly more
                m.cfg.thr_scale = float(np.clip(m.cfg.thr_scale * (1.05 if verdict == "dismissed" else 0.97), 0.7, 1.4))
                if feedback_csv and getattr(tr, "feat", None):
                    import csv
                    new = not os.path.exists(feedback_csv)
                    with open(feedback_csv, "a", newline="") as f:
                        w = csv.writer(f)
                        if new:
                            w.writerow(["wall", "verdict", "label_pred"] + list(tr.feat.keys()))
                        w.writerow([wall, verdict, tr.label] + [f"{v:.4f}" for v in tr.feat.values()])
                return True
        return False
