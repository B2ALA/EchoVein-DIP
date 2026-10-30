"""Zero-dependency web dashboard (stdlib http.server). Two views: Live camera / Leak analysis."""
import json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>EchoVein-DIP</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0f1419;--card:#1a212b;--tx:#e6edf3;--mu:#8b98a5;--ok:#2ea043;--w:#d29922;--s:#f0883e;--e:#f85149;--v:#bc8cff}
@media (prefers-color-scheme: light){:root{--bg:#f4f6f8;--card:#fff;--tx:#1b2430;--mu:#5b6877}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:14px system-ui,sans-serif}
header{display:flex;gap:12px;align-items:center;padding:10px 16px;border-bottom:1px solid #3333}
header h1{font-size:16px;margin:0;flex:1}button,select{background:var(--card);color:var(--tx);border:1px solid #5555;border-radius:6px;padding:6px 10px;cursor:pointer}
button.on{outline:2px solid #58a6ff}main{display:grid;grid-template-columns:2fr 1fr;gap:12px;padding:12px}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--card);border-radius:10px;padding:10px;margin-bottom:12px}.card h2{font-size:13px;margin:0 0 8px;color:var(--mu);text-transform:uppercase;letter-spacing:.04em}
img{max-width:100%;border-radius:6px;display:block}.pill{padding:2px 8px;border-radius:99px;font-size:12px;color:#fff}
.NORMAL{background:var(--ok)}.WATCH{background:var(--w)}.SUSPECTED{background:var(--s)}.ESCALATED{background:var(--e)}.VERIFIED{background:var(--v)}.RESOLVED,.DISMISSED{background:#6e7781}
table{width:100%;border-collapse:collapse}td,th{padding:4px 6px;text-align:left;border-bottom:1px solid #3333;font-size:13px}
.warn{color:var(--w)}canvas{width:100%;height:150px;background:#0002;border-radius:6px}.mu{color:var(--mu)}
</style></head><body>
<header><h1>EchoVein-DIP · moisture-aware wall monitor</h1>
<select id="wall"></select><button id="v1" class="on">Live CCTV view</button><button id="v2">Leak analysis view</button></header>
<main><div id="left"></div><div id="right"></div></main>
<script>
let view=1,S=null;const $=id=>document.getElementById(id);
$('v1').onclick=()=>{view=1;$('v1').className='on';$('v2').className='';draw()};
$('v2').onclick=()=>{view=2;$('v2').className='on';$('v1').className='';draw()};
$('wall').onchange=draw;
async function poll(){try{S=await (await fetch('/api/state')).json();const sel=$('wall');if(sel.options.length!=S.walls.length){sel.innerHTML=S.walls.map(w=>`<option>${w.wall}</option>`).join('')}draw()}catch(e){}setTimeout(poll,1500)}
function pill(s){return `<span class="pill ${s}">${s}</span>`}
function plot(id,series,label){const c=$(id);if(!c)return;const x=c.getContext('2d');c.width=c.clientWidth*2;c.height=300;x.clearRect(0,0,c.width,c.height);
 let all=series.flatMap(s=>s.pts);if(!all.length)return;const tx=all.map(p=>p[0]),ty=all.map(p=>p[1]);const x0=Math.min(...tx),x1=Math.max(...tx)+1e-6,y1=Math.max(...ty,1e-6);
 series.forEach((s,i)=>{x.beginPath();x.strokeStyle=['#58a6ff','#f0883e','#3fb950','#bc8cff'][i%4];x.lineWidth=3;s.pts.forEach((p,j)=>{const px=20+(p[0]-x0)/(x1-x0)*(c.width-40),py=c.height-20-(p[1]/y1)*(c.height-40);j?x.lineTo(px,py):x.moveTo(px,py)});x.stroke()});
 x.fillStyle='#8b98a5';x.font='22px sans-serif';x.fillText(label+'  (max '+y1.toFixed(1)+')',24,26)}
function fb(w,t,v){fetch('/api/feedback',{method:'POST',body:JSON.stringify({wall:w,track:t,verdict:v})})}
function draw(){if(!S)return;const name=$('wall').value||S.walls[0].wall;const W=S.walls.find(w=>w.wall==name)||S.walls[0];const ts=Date.now();
 const warn=(W.info.warnings||[]).map(x=>`<div class="warn">⚠ ${x}</div>`).join('');
 const tracks=W.tracks.filter(t=>t.active||['ESCALATED','VERIFIED'].includes(t.peak_state));
 const rank=S.ranking.map(r=>`<tr><td>${r.wall}</td><td>${pill(r.status)}</td><td>${r.severity.toFixed(0)}</td><td>${r.z.toFixed(1)}</td></tr>`).join('');
 const hist=S.walls.flatMap(w=>w.events.map(e=>({w:w.wall,...e}))).sort((a,b)=>b.t-a.t).slice(0,12).map(e=>`<tr><td>${e.t.toFixed(2)}h</td><td>${e.w}</td><td>${pill(e.state)}</td><td class="mu">#${e.track} ${e.label}</td></tr>`).join('');
 if(view==1){
  $('left').innerHTML=`<div class="card"><h2>Live camera · ${W.wall} ${pill(W.status)}</h2><img src="/img/${W.wall}/camera.jpg?${ts}">${warn}
  <div class="mu">baseline: ${W.baseline||'-'} · camera shift: ${(W.info.shift_px||0).toFixed(1)}px · occluded: ${((W.info.occluded||0)*100).toFixed(0)}% · t=${W.t.toFixed(2)}h</div></div>`;
  $('right').innerHTML=`<div class="card"><h2>Wall ranking</h2><table><tr><th>wall</th><th>status</th><th>sev</th><th>z</th></tr>${rank}</table></div><div class="card"><h2>Event history</h2><table>${hist||'<tr><td class="mu">no events</td></tr>'}</table></div>`;
 }else{
  const cards=tracks.map(t=>{const src=t.source,p=t.pipes&&t.pipes[0];
   return `<div class="card"><h2>Region #${t.id} · ${pill(t.state)} · ${t.label}</h2>
   <div>position <b>${t.x_cm.toFixed(0)} / ${t.y_cm.toFixed(0)} cm</b> (from left / top) · area ${t.area_cm2.toFixed(0)} cm²<br>
   distances: L ${t.left_cm.toFixed(0)} · R ${t.right_cm.toFixed(0)} · T ${t.top_cm.toFixed(0)} · B ${t.bottom_cm.toFixed(0)} cm<br>
   relative severity <b>${t.severity.toFixed(0)}</b>/100 · radius growth ${t.growth_cm_h.toFixed(1)} cm/h<br><span class="mu">why: ${t.reason}; class probs ${JSON.stringify(t.probs)}</span></div>
   ${src?`<div>estimated source: <b>${src.x_cm.toFixed(0)}, ${src.y_cm.toFixed(0)} cm</b> ± ${src.sx_cm.toFixed(0)}/${src.sy_cm.toFixed(0)} cm <span class="mu">(${src.method}) — a wet-region origin, not a certain break point</span></div>`:''}
   ${p?`<div>most probable pipe: <b>${p.pipe}</b> (${(p.prob*100).toFixed(0)}%) · inspect at ${p.inspect_x_cm.toFixed(0)}, ${p.inspect_y_cm.toFixed(0)} cm · ${p.arc_length_cm.toFixed(0)} cm along pipe</div>`:''}
   ${(t.assets||[]).map(a=>`<div class="${a.electrical?'warn':''}">${a.electrical?'⚡ ':''}${a.name}: ${a.status}${a.t_min_h!=null?` · ~${a.t_min_h.toFixed(1)}–${a.t_max_h.toFixed(1)} h`:''}</div>`).join('')}
   <canvas id="pl${t.id}"></canvas><div style="margin-top:6px"><button onclick="fb('${W.wall}',${t.id},'verified')">✔ Verify leak</button> <button onclick="fb('${W.wall}',${t.id},'dismissed')">✖ Dismiss</button></div></div>`}).join('');
  $('left').innerHTML=`<div class="card"><h2>Evidence card · ${W.wall} (baseline · now · score · result)</h2><img src="/img/${W.wall}/evidence.jpg?${ts}"></div>`;
  $('right').innerHTML=(cards||'<div class="card mu">No active regions on this wall.</div>')+`<div class="card"><h2>Wall ranking</h2><table><tr><th>wall</th><th>status</th><th>sev</th><th>z</th></tr>${rank}</table></div><div class="card"><h2>Event history</h2><table>${hist}</table></div><div class="card mu">Thresholds tuned by feedback: ×${W.cfg_thr_scale.toFixed(2)}</div>`;
  tracks.forEach(t=>plot('pl'+t.id,[{pts:t.history.map(h=>[h.t,h.area])},{pts:t.history.map(h=>[h.t,h.score*Math.max(...t.history.map(q=>q.area),1)])}],'area cm² (blue) / score (orange)'));
 }}
poll();
</script></body></html>"""


class Dashboard:
    def __init__(self, fleet, port=8000, feedback_csv="out/feedback.csv"):
        self.fleet, self.port, self.csv = fleet, port, feedback_csv
        self.lock = threading.Lock()
        self.snaps = {}
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def _send(self, code, body, ctype):
                self.send_response(code); self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

            def do_GET(self):
                p = self.path.split("?")[0]
                if p == "/":
                    return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                if p == "/api/state":
                    with outer.lock:
                        d = {"walls": list(outer.snaps.values()), "ranking": outer.fleet.ranking}
                    return self._send(200, json.dumps(d, default=float).encode(), "application/json")
                if p.startswith("/img/"):
                    _, _, wall, kind = p.split("/")
                    m = outer.fleet.mon.get(wall)
                    img = None if m is None else getattr(m, "images", {}).get(kind.split(".")[0])
                    if img is None:
                        return self._send(404, b"", "text/plain")
                    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    return self._send(200, buf.tobytes(), "image/jpeg")
                self._send(404, b"", "text/plain")

            def do_POST(self):
                if self.path == "/api/feedback":
                    d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                    with outer.lock:
                        ok = outer.fleet.feedback(d["wall"], int(d["track"]), d["verdict"], outer.csv)
                    return self._send(200, json.dumps({"ok": ok}).encode(), "application/json")
                self._send(404, b"", "text/plain")

        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), H)

    def update(self, snaps):
        with self.lock:
            self.snaps = dict(snaps)

    def start(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        print(f"Dashboard: http://localhost:{self.port}")
