"""Run EchoVein-DIP.
  python run.py --demo                       # simulated 2-wall leak, no hardware, opens dashboard
  python run.py --project project.json       # real phone(s)
project.json: see project.example.json"""
import argparse, json, os, time
import cv2, numpy as np
from evdip.common import Calib, Config, Source
from evdip.pipeline import WallMonitor, Fleet
from evdip.backtrace import load_layout
from evdip.dashboard import Dashboard
from evdip.evidence import fuse_evidence

ap = argparse.ArgumentParser()
ap.add_argument("--project")
ap.add_argument("--demo", action="store_true")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--speed", type=float, default=1.0, help="demo seconds per simulated step")
ap.add_argument("--out", default="out")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def save_cards(fleet, tag):
    for n, m in fleet.mon.items():
        if hasattr(m, "images"):
            cv2.imwrite(os.path.join(a.out, f"{n}_{tag}_evidence.jpg"), m.images["evidence"])


if a.demo:
    from evdip.sim import simulate
    sim = simulate("leak", seed=3, hours=10, dt_min=15, n_walls=2)
    sx, sy = sim["walls"][0]["gt_source"]
    layout = {"pipes": [{"name": "cold supply (vertical)", "points": [[sx + 4, 0], [sx + 4, 80], [sx + 60, 80]], "start_m": 0},
                        {"name": "hot supply (horizontal)", "points": [[0, sy + 45], [200, sy + 45]], "start_m": 0}],
              "assets": [{"name": "socket S1", "x": sx + 38, "y": sy + 42, "type": "socket"}]}
    mons = []
    for i, w in enumerate(sim["walls"]):
        mons.append(WallMonitor(f"wall{i+1}", Calib(w["sim"].calib_dict()), w["baselines"], layout=layout if i == 0 else None,
                                cfg=Config(fusion="ml")))
    fleet = Fleet(mons); dash = Dashboard(fleet, a.port, os.path.join(a.out, "feedback.csv")); dash.start()
    print(f"GROUND TRUTH source: ({sx:.0f}, {sy:.0f}) cm on wall1")
    for k, t in enumerate(sim["times"]):
        snaps = fleet.step({f"wall{i+1}": w["frames"][k] for i, w in enumerate(sim["walls"])}, float(t))
        dash.update(snaps)
        s = snaps["wall1"]
        print(f"t={t:5.2f}h wall1={s['status']:<9} wall2={snaps['wall2']['status']:<9}",
              [(tr["label"], tr["state"], round(tr["x_cm"]), round(tr["y_cm"])) for tr in s["tracks"] if tr["active"]])
        time.sleep(a.speed)
    save_cards(fleet, "demo")
    print("Demo finished; dashboard stays up (Ctrl+C to quit). Evidence cards saved in", a.out)
    while True:
        time.sleep(1)

cfgj = json.load(open(a.project))
interval = cfgj.get("interval_s", 10)
cfg = Config(**cfgj.get("config", {}))
srcs, mons, thermals = {}, [], {}
for w in cfgj["walls"]:
    srcs[w["name"]] = Source(w["source"])
    bl = [(os.path.basename(p), cv2.imread(p)) for p in w["baselines"]]
    layout = load_layout(w.get("layout"))
    mons.append(WallMonitor(w["name"], Calib(w["calib"]), bl, layout=layout, cfg=Config(**cfg.to_dict()),
                            pixel_model="auto"))
    if w.get("thermal"):
        thermals[w["name"]] = w["thermal"]
fleet = Fleet(mons)
dash = Dashboard(fleet, cfgj.get("port", a.port), os.path.join(a.out, "feedback.csv")); dash.start()
log = open(os.path.join(a.out, "events.jsonl"), "a")
t_start = time.time()
cams = {w["name"]: w["source"] for w in cfgj["walls"]}
fails = {n: 0 for n in srcs}

while True:
    t0 = time.time()
    t = (t0 - t_start) / 3600.0
    frames = {}

    for n, s in list(srcs.items()):
        f = s.read()

        if f is None:
            fails[n] += 1
            if fails[n] >= 5:
                try:
                    s.release()
                    srcs[n] = Source(cams[n])
                    fails[n] = 0
                    print(f"{n}: camera reconnected")
                except IOError:
                    print(f"{n}: camera still offline")
        else:
            fails[n] = 0
            frames[n] = f
    for n, th in thermals.items():             # optional thermal evidence (cool anomaly)
        from evdip.thermal import load_temp, register, thermal_cue
        m = fleet.mon[n]
        try:
            tm = register(load_temp(th["temp_file"]), th["corners"], (m.cal.W, m.cal.Hpx))
            m.thermal_cue = thermal_cue(tm)
        except Exception as e:
            print("thermal skipped:", e)
    try:
        snaps = fleet.step(frames, t)
    except Exception:
        import traceback
        traceback.print_exc()
        time.sleep(interval)
        continue
    dash.update(snaps)
    for n, s in snaps.items():
        for tr in s["tracks"]:
            if tr["active"]:
                print(f"[{time.strftime('%H:%M:%S')}] {n}: {tr['state']} {tr['label']} at x={tr['x_cm']:.0f} y={tr['y_cm']:.0f} cm "
                      f"(L{tr['left_cm']:.0f} R{tr['right_cm']:.0f} T{tr['top_cm']:.0f} B{tr['bottom_cm']:.0f}) sev={tr['severity']:.0f}")
        log.write(json.dumps({"t": t, "wall": n, "status": s["status"], "tracks": s["tracks"]}, default=float) + "\n"); log.flush()
    time.sleep(max(0, interval - (time.time() - t0)))
