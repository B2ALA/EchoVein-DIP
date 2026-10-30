"""EchoVein live scanner.
  python scan.py                         # http://localhost:8443  (laptop webcam works on localhost)
  python scan.py --https --host 0.0.0.0  # phone on same Wi-Fi: open https://<laptop-ip>:8443 (accept the self-signed cert warning)
  python scan.py --token SECRET          # require a token for the API when not on a trusted network"""
import argparse, socket
from evdip.scanner import ScannerConfig
from evdip.scanner_server import serve

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8443)
ap.add_argument("--https", action="store_true"); ap.add_argument("--token", default="")
ap.add_argument("--data-dir", default="data"); ap.add_argument("--proc-width", type=int, default=640)
ap.add_argument("--max-jobs", type=int, default=2); ap.add_argument("--retention-days", type=int, default=30)
a = ap.parse_args()
cfg = ScannerConfig(data_dir=a.data_dir, proc_width=a.proc_width, max_jobs=a.max_jobs, token=a.token, retention_days=a.retention_days)
svc, httpd = serve(cfg, a.host, a.port, a.https)
n = svc.store.purge_older_than(cfg.retention_days)
proto = "https" if a.https else "http"
try:
    ip = socket.gethostbyname(socket.gethostname())
except OSError:
    ip = "<laptop-ip>"
print(f"EchoVein scanner: {proto}://localhost:{a.port}" + (f"   phone: {proto}://{ip}:{a.port}" if a.host != "127.0.0.1" else ""))
if a.host != "127.0.0.1" and not a.https:
    print("WARNING: phones usually block camera access on plain-HTTP LAN addresses - use --https or a tunnel (see SCANNER.md).")
if a.token:
    print(f"API token required. Open the page as {proto}://localhost:{a.port}/?token=<token>")
if n: print(f"retention: deleted {n} old session(s)")
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    print("stopped")
