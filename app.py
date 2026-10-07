"""The search UI: a tiny local web server around pipeline.py. Standard library only.

    python app.py            # then open http://127.0.0.1:8000

The browser posts a query, gets a job id back, and polls /api/jobs/<id> about three times
a second for progress. The pipeline runs in a background thread, so the page stays live.
"""
import argparse
import json
import threading
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

import config
import pipeline

PAGE = config.ROOT / "templates" / "app.html"
jobs = {}                 # job id -> dict the browser polls
jobs_lock = threading.Lock()


def new_job(query):
    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "query": query, "status": "running", "stage": 0,
           "percent": 0.0, "message": "Starting", "error": None, "result": None}
    with jobs_lock:
        jobs[job_id] = job
    threading.Thread(target=run_job, args=(job,), daemon=True).start()
    return job_id


def run_job(job):
    def report(stage, fraction, message):
        with jobs_lock:
            job["stage"] = stage
            job["percent"] = round(pipeline.overall_percent(stage, fraction), 1)
            job["message"] = message

    try:
        result = pipeline.run(job["query"], report)
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
            query = json.loads(self.rfile.read(length) or b"{}").get("query", "")
        except ValueError:
            query = ""
        if len(query.strip()) < 3:
            self.send_json({"error": "Type a topic or a few keywords."}, 400)
            return
        self.send_json({"job_id": new_job(query[:300])})

    def log_message(self, fmt, *args):
        pass  # polling would flood the terminal


def main(host, port, open_browser):
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"paper-atlas is running at {url}  (Ctrl+C to stop)")
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
