"""Stdlib HTTP(S) server for the live scanner (same style as the existing dashboard; no new web dependency).
Only fixed routes are served - no static file serving, no client-controlled paths."""
import json, re, ssl, subprocess, sys, threading, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import numpy as np
from .scanner import ScanService, ApiError, ScannerConfig, ROOT, CATEGORIES

PAGE = ROOT / "web" / "scanner.html"
SID = r"([0-9a-f]{12})"


def _default(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o))


def make_handler(svc):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a): pass

        def _send(self, code, body, ctype="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body, default=_default).encode()
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store"); self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Permissions-Policy", "camera=(self)"); self.end_headers(); self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > svc.cfg.max_frame_bytes + 1000:
                self.rfile.read(min(n, 10_000_000)) if n < 10_000_000 else None
                raise ApiError(413, "payload too large")
            return self.rfile.read(n) if n else b""

        def _json(self):
            try:
                d = json.loads(self._body() or b"{}")
                if not isinstance(d, dict): raise ValueError
                return d
            except ValueError:
                raise ApiError(400, "invalid JSON body")

        def _auth(self, q):
            if svc.cfg.token and self.headers.get("X-EchoVein-Token") != svc.cfg.token and (q.get("token", [""])[0] != svc.cfg.token):
                raise ApiError(401, "missing or wrong token")

        def _route(self, method):
            u = urlparse(self.path); p = u.path; q = parse_qs(u.query)
            if method == "GET" and p in ("/", "/scanner"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if not p.startswith("/api/"):
                raise ApiError(404, "not found")
            self._auth(q)
            if method == "GET" and p == "/api/health":
                return self._send(200, {"ok": True, "model_versions": svc.versions, "config": {"proc_width": svc.cfg.proc_width, "max_jobs": svc.cfg.max_jobs}, "categories": CATEGORIES})
            if p == "/api/sessions":
                if method == "POST": return self._send(200, svc.create_session(self._json()))
                if method == "GET": return self._send(200, {"sessions": svc.store.list_sessions()})
            m = re.fullmatch(r"/api/sessions/" + SID + r"(?:/([a-z_]+))?", p)
            if not m:
                raise ApiError(404, "not found")
            sid, act = m.group(1), m.group(2)
            if act is None and method == "GET": return self._send(200, svc.session_detail(sid))
            if act is None and method == "DELETE": return self._send(200, svc.delete(sid))
            if method == "POST":
                if act == "frame":
                    sim = q.get("t", [None])[0]
                    return self._send(200, svc.process_frame(sid, self._body(), q.get("images", ["0"])[0] == "1", float(sim) if sim else None))
                if act == "detect_wall": return self._send(200, svc.detect_wall(sid, self._body()))
                if act == "auto_calibrate":
                    return self._send(200, svc.auto_calibrate(sid, self._body(), float(q.get("marker_cm", ["15"])[0]), q.get("apply", ["1"])[0] == "1", q.get("select_area", ["0"])[0] == "1"))
                if act == "calibrate": return self._send(200, svc.calibrate(sid, self._json()))
                if act == "scale_check": return self._send(200, svc.scale_check(sid, self._json()))
                if act == "reference":
                    return self._send(200, svc.add_reference(sid, q.get("category", ["dry"])[0], q.get("notes", [""])[0], self._body(), q.get("force", ["0"])[0] == "1"))
                if act == "thermal": return self._send(200, svc.set_thermal(sid, self._json()))
                if act == "layout": return self._send(200, svc.set_layout(sid, self._json()))
                if act == "snapshot": return self._send(200, svc.snapshot(sid))
                if act == "feedback":
                    d = self._json(); return self._send(200, svc.feedback(sid, int(d.get("track", -1)), str(d.get("verdict", ""))))
            if method == "GET":
                if act == "results": return self._send(200, svc.results(sid))
                if act == "report":
                    return self._send(200, svc.report_html(sid).encode(), "text/html; charset=utf-8")
            raise ApiError(404, "not found")

        def _wrap(self, method):
            try:
                self._route(method)
            except ApiError as e:
                self._send(e.status, {"error": e.msg})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                traceback.print_exc(file=sys.stderr)
                try: self._send(500, {"error": "internal error"})
                except Exception: pass

        def do_GET(self): self._wrap("GET")
        def do_POST(self): self._wrap("POST")
        def do_DELETE(self): self._wrap("DELETE")
    return H


def make_cert(cert, key):
    cert, key = Path(cert), Path(key); cert.parent.mkdir(parents=True, exist_ok=True)
    if not (cert.exists() and key.exists()):
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out", str(cert), "-days", "365",
                        "-subj", "/CN=echovein.local"], check=True, capture_output=True)


def serve(cfg=None, host="127.0.0.1", port=8443, https=False, cert="data/certs/cert.pem", key="data/certs/key.pem", background=False):
    svc = ScanService(cfg or ScannerConfig())
    httpd = ThreadingHTTPServer((host, port), make_handler(svc)); httpd.daemon_threads = True
    if https:
        make_cert(cert, key)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); ctx.load_cert_chain(cert, key)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    if background:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return svc, httpd
