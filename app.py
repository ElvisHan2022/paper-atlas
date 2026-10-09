"""The search UI: a tiny local web server around pipeline.py. Standard library only.

    python app.py            # then open http://127.0.0.1:8000

The browser posts a query, gets a job id back, and polls /api/jobs/<id> about three times
a second for progress. The pipeline runs in a background thread, so the page stays live.
"""
import argparse
import json
import subprocess
import sys
import threading
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

import config
import llm
import pipeline

PAGE = config.ROOT / "templates" / "app.html"
jobs = {}                 # job id -> dict the browser polls
jobs_lock = threading.Lock()


def running_job():
    with jobs_lock:
        return next((j for j in jobs.values() if j["status"] == "running"), None)


def new_job(query, source_names=None, lens="balanced"):
    """Start a search, or return the one already running.

    One search at a time: they share the local models and the database, and parallel
    searches mostly just slow each other down.
    """
    busy = running_job()
    if busy:
        return busy["id"], busy["query"]
    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "query": query, "sources": source_names, "lens": lens,
           "status": "running", "stage": 0, "percent": 0.0, "message": "Starting",
           "error": None, "result": None}
    with jobs_lock:
        jobs[job_id] = job
    threading.Thread(target=run_job, args=(job,), daemon=True).start()
    return job_id, None


def run_job(job):
    def report(stage, fraction, message):
        with jobs_lock:
            job["stage"] = stage
            job["percent"] = round(pipeline.overall_percent(stage, fraction), 1)
            job["message"] = message

    try:
        result = pipeline.run(job["query"], report, job["sources"], job["lens"])
        with jobs_lock:
            job.update(status="done", percent=100.0, stage=len(pipeline.STAGES),
                       message="Done", result=result)
    except pipeline.PipelineError as e:
        with jobs_lock:
            job.update(status="error", error=str(e))
    except Exception as e:  # unexpected: log it, and still tell the browser
        traceback.print_exc()
        with jobs_lock:
            job.update(status="error", error=f"Something went wrong: {e}")


def recent_runs(limit=6):
    if not config.RUNS_DIR.exists():
        return []
    out = []
    for path in sorted(config.RUNS_DIR.glob("*.json"), reverse=True)[:limit]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            out.append({"file": path.name, "query": data["query"],
                        "created_at": data["created_at"]})
        except (OSError, ValueError, KeyError):
            continue
    return out


class Handler(BaseHTTPRequestHandler):
    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/jobs/"):
            with jobs_lock:
                job = jobs.get(self.path.rsplit("/", 1)[-1])
                snapshot = dict(job) if job else None
            if snapshot is None:
                self.send_json({"error": "unknown job"}, 404)
            else:
                snapshot["stages"] = pipeline.STAGES
                self.send_json(snapshot)
        elif self.path == "/api/options":
            self.send_json({
                "sources": config.SOURCES,
                "lenses": {k: {"label": v["label"], "summary": v["summary"]}
                           for k, v in config.LENSES.items()},
                "has_llm": llm.has_key(),
                "key_status": llm.KEY_STATUS,
                "key_hint": llm.masked_key(),
                "key_problem": llm.key_problem(),
                "env_path": str(config.ROOT / ".env"),
                "version": VERSION,
            })
        elif self.path == "/api/runs":
            self.send_json(recent_runs())
        elif self.path.startswith("/api/runs/"):
            name = unquote(self.path.rsplit("/", 1)[-1])
            path = config.RUNS_DIR / name
            # Only plain file names inside RUNS_DIR; no "../" tricks.
            if "/" in name or "\\" in name or not name.endswith(".json") or not path.exists():
                self.send_json({"error": "not found"}, 404)
            else:
                self.send_json(json.loads(path.read_text(encoding="utf-8")))
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/api/search":
            self.send_json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            body = {}
        query = str(body.get("query", ""))
        if len(query.strip()) < 3:
            self.send_json({"error": "Type a topic or a few keywords."}, 400)
            return
        chosen = [s for s in body.get("sources") or [] if s in config.SOURCES] or None
        lens = body.get("lens") if body.get("lens") in config.LENSES else "balanced"
        job_id, already = new_job(query[:300], chosen, lens)
        self.send_json({"job_id": job_id, "already_running": already})

    def log_message(self, fmt, *args):
        pass  # polling would flood the terminal


def report_key():
    """Check the API key once at startup and say plainly what's wrong, if anything."""
    env = config.ROOT / ".env"
    if env.exists():
        print(f".env file: {env}")
    else:
        wrong = [f.name for f in config.ROOT.glob(".env*") if f.name != ".env.example"]
        print(f".env file: NOT FOUND. Expected it at {env}"
              + (f" (found {', '.join(wrong)} instead; rename it to .env)" if wrong else ""))
    status = llm.check_key()
    messages = {
        "ok": f"Anthropic API key: OK ({llm.masked_key()})",
        "missing": (f"Anthropic API key: not found. Add a line ANTHROPIC_API_KEY=<your key> to {env}. "
                    "Searches will use the local models only."),
        "rejected": (f"Anthropic API key: REJECTED ({llm.masked_key()}). "
                     + (llm.key_problem() or "Make a new key at console.anthropic.com.")
                     + f" Paste it into {env} and restart. Until then, searches use the local "
                     "models only."),
        "unreachable": "Anthropic API key: couldn't check it (no connection?). Will try anyway.",
    }
    print(messages[status])


class Server(ThreadingHTTPServer):
    # HTTPServer turns on address reuse. On Windows that lets a second copy of the app bind
    # the same port while the first is still running, and the browser may keep talking to
    # the old copy. Turning it off makes a second copy fail loudly instead.
    allow_reuse_address = sys.platform != "win32"


def app_version():
    """Short git commit of the running code, shown on the page so stale copies are obvious."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


VERSION = app_version()


def main(host, port, open_browser):
    try:
        server = Server((host, port), Handler)
    except OSError:
        print(f"Port {port} is already in use, probably by another paper-atlas window that is "
              "still running. Close that window (or press Ctrl+C in it) and try again, or run "
              f"python app.py --port {port + 1}")
        sys.exit(1)
    url = f"http://{host}:{port}"
    print(f"paper-atlas {VERSION} is running at {url}  (Ctrl+C to stop)")
    threading.Thread(target=report_key, daemon=True).start()
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default=config.WEB_HOST)
    ap.add_argument("--port", type=int, default=config.WEB_PORT)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    main(args.host, args.port, not args.no_browser)
