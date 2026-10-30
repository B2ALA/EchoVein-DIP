"""Local scan storage: SQLite metadata + image/thermal files under data/. Server-generated ids only
(no client string ever becomes a path). Supports retention and full deletion."""
import json, os, re, shutil, sqlite3, threading, time, uuid
from pathlib import Path

WALL_RE = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")
ID_RE = re.compile(r"^[0-9a-f]{12}$")


def _j(s):
    try:
        return json.loads(s) if s else {}
    except Exception:
        return {}                      # corrupted record -> empty, never crash


class Store:
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.db = str(self.root / "echovein.sqlite3"); self.lock = threading.RLock()
        c = self._c()
        try:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, wall_id TEXT, created REAL, camera_type TEXT, meta TEXT, calib TEXT);
            CREATE TABLE IF NOT EXISTS refs(id TEXT PRIMARY KEY, session_id TEXT, wall_id TEXT, category TEXT, created REAL, path TEXT, meta TEXT);
            CREATE TABLE IF NOT EXISTS results(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, created REAL, summary TEXT);
            CREATE TABLE IF NOT EXISTS snapshots(id TEXT PRIMARY KEY, session_id TEXT, created REAL, files TEXT, result TEXT);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, wall_id TEXT, created REAL, x REAL, y REAL, state TEXT, label TEXT);
            """)
        finally:
            c.close()

    def _c(self):
        c = sqlite3.connect(self.db, timeout=10); c.row_factory = sqlite3.Row; return c

    def _q(self, sql, args=(), one=False, write=False):
        with self.lock:
            c = self._c()
            try:
                cur = c.execute(sql, args)
                if write:
                    c.commit(); return cur.lastrowid
                rows = cur.fetchall()
                return (rows[0] if rows else None) if one else rows
            finally:
                c.close()

    def sdir(self, sid):
        assert ID_RE.match(sid); d = self.root / "sessions" / sid; d.mkdir(parents=True, exist_ok=True); return d

    # sessions
    def create_session(self, wall_id, camera_type, meta):
        if not WALL_RE.match(wall_id or ""):
            raise ValueError("wall_id must be 1-40 chars: letters, digits, _ or -")
        sid = uuid.uuid4().hex[:12]
        self._q("INSERT INTO sessions VALUES(?,?,?,?,?,?)", (sid, wall_id, time.time(), camera_type[:20], json.dumps(meta), ""), write=True)
        self.sdir(sid); return sid

    def get_session(self, sid):
        r = self._q("SELECT * FROM sessions WHERE id=?", (sid,), one=True) if ID_RE.match(sid or "") else None
        return None if r is None else {"id": r["id"], "wall_id": r["wall_id"], "created": r["created"], "camera_type": r["camera_type"],
                                       "meta": _j(r["meta"]), "calib": _j(r["calib"]) or None}

    def list_sessions(self, limit=50):
        return [{"id": r["id"], "wall_id": r["wall_id"], "created": r["created"], "camera_type": r["camera_type"], "meta": _j(r["meta"]),
                 "calibrated": bool(r["calib"])} for r in self._q("SELECT * FROM sessions ORDER BY created DESC LIMIT ?", (limit,))]

    def set_calib(self, sid, calib):
        self._q("UPDATE sessions SET calib=? WHERE id=?", (json.dumps(calib), sid), write=True)

    def set_meta(self, sid, meta):
        self._q("UPDATE sessions SET meta=? WHERE id=?", (json.dumps(meta), sid), write=True)

    # references
    def add_ref(self, sid, wall_id, category, jpeg_bytes, meta):
        rid = uuid.uuid4().hex[:12]; d = self.sdir(sid) / "refs"; d.mkdir(exist_ok=True)
        p = d / f"{rid}.jpg"; p.write_bytes(jpeg_bytes)
        self._q("INSERT INTO refs VALUES(?,?,?,?,?,?,?)", (rid, sid, wall_id, category, time.time(), str(p), json.dumps(meta)), write=True)
        return rid

    def list_refs(self, sid=None, wall_id=None):
        if sid:
            rows = self._q("SELECT * FROM refs WHERE session_id=? ORDER BY created", (sid,))
        else:
            rows = self._q("SELECT * FROM refs WHERE wall_id=? ORDER BY created DESC LIMIT 100", (wall_id,))
        return [{"id": r["id"], "session_id": r["session_id"], "wall_id": r["wall_id"], "category": r["category"],
                 "created": r["created"], "path": r["path"], "meta": _j(r["meta"])} for r in rows]

    # results / snapshots / events
    def add_result(self, sid, summary):
        self._q("INSERT INTO results(session_id,created,summary) VALUES(?,?,?)", (sid, time.time(), json.dumps(summary, default=float)), write=True)

    def list_results(self, sid, limit=100):
        return [{"created": r["created"], **_j(r["summary"])} for r in self._q("SELECT * FROM results WHERE session_id=? ORDER BY id DESC LIMIT ?", (sid, limit))]

    def add_snapshot(self, sid, files_bytes, result, keep=50):
        nid = uuid.uuid4().hex[:12]; d = self.sdir(sid) / "snaps"; d.mkdir(exist_ok=True); files = {}
        for k, (ext, b) in files_bytes.items():
            p = d / f"{nid}_{k}.{ext}"; p.write_bytes(b); files[k] = str(p)
        self._q("INSERT INTO snapshots VALUES(?,?,?,?,?)", (nid, sid, time.time(), json.dumps(files), json.dumps(result, default=float)), write=True)
        rows = self._q("SELECT id, files FROM snapshots WHERE session_id=? ORDER BY created DESC", (sid,))
        for r in rows[keep:]:                         # retention: keep newest N
            for f in _j(r["files"]).values():
                try: os.remove(f)
                except OSError: pass
            self._q("DELETE FROM snapshots WHERE id=?", (r["id"],), write=True)
        return nid

    def last_snapshot(self, sid):
        r = self._q("SELECT * FROM snapshots WHERE session_id=? ORDER BY created DESC LIMIT 1", (sid,), one=True)
        return None if r is None else {"id": r["id"], "created": r["created"], "files": _j(r["files"]), "result": _j(r["result"])}

    def list_snapshots(self, sid):
        return [{"id": r["id"], "created": r["created"]} for r in self._q("SELECT id, created FROM snapshots WHERE session_id=? ORDER BY created DESC", (sid,))]

    def add_event(self, sid, wall_id, x, y, state, label):
        self._q("INSERT INTO events(session_id,wall_id,created,x,y,state,label) VALUES(?,?,?,?,?,?,?)", (sid, wall_id, time.time(), x, y, state, label), write=True)

    def recurring_count(self, wall_id, x, y, radius=15.0, exclude_session=None):
        rows = self._q("SELECT DISTINCT session_id, x, y FROM events WHERE wall_id=?", (wall_id,))
        return len({r["session_id"] for r in rows if r["session_id"] != exclude_session and ((r["x"] - x) ** 2 + (r["y"] - y) ** 2) ** 0.5 <= radius})

    # deletion / retention
    def delete_session(self, sid):
        if not self.get_session(sid):
            return False
        for t in ("refs", "results", "snapshots", "events"):
            self._q(f"DELETE FROM {t} WHERE session_id=?", (sid,), write=True)
        self._q("DELETE FROM sessions WHERE id=?", (sid,), write=True)
        shutil.rmtree(self.root / "sessions" / sid, ignore_errors=True); return True

    def purge_older_than(self, days):
        cut = time.time() - days * 86400
        old = [r["id"] for r in self._q("SELECT id FROM sessions WHERE created<?", (cut,))]
        return sum(self.delete_session(s) for s in old)
