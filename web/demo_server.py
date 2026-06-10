from __future__ import annotations

import json
import mimetypes
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from duomon.benchmarking.benchmark_control import write_initial_control
from web.server.config import (
    DEFAULT_TEAM_P1,
    DEFAULT_TEAM_P3,
    DEMO_TEAM_ROOT,
    _coerce_team_text,
    _dataset_has_training_scope,
    _fixed_team_env,
    _generate_random_ally_teams,
    _looks_like_exported_set,
    _normalize_team_text,
    _path_has_records,
    _payload_enabled,
    _read_text_file,
    _split_exported_team,
    _team_namespace,
    _write_demo_team,
)
from web.server.progress import (
    _bounded_int,
    _canonical_opponent,
    _initial_progress,
    _refresh_progress_timing,
    _set_progress,
    _update_progress_from_log,
)
from web.server.results import (
    _latest_artifacts,
    _battle_trace,
    _battle_results,
    _filter_battles,
    _find_replay_html,
    _rewrite_replay_html,
    _summary_for_rows,
)
from web.server.requests import (
    _benchmark_env_from_payload,
    _train_env_from_payload,
)
from web.server.static import (
    CONFIG_JS,
    PLAY_ROOT,
    _local_config_js,
    _rewrite_showdown_index,
    _safe_file,
)


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = Path(__file__).resolve().parent

HOST = os.environ.get("DUOMON_DASHBOARD_HOST", "127.0.0.1")
PORT = int(os.environ.get("DUOMON_DASHBOARD_PORT", "8765"))
PYTHON_EXE = sys.executable.replace("pythonw.exe", "python.exe")
NODE_EXE = os.environ.get("NODE_EXE", "node")

JOBS: Dict[str, Dict[str, Any]] = {}
SHOWDOWN_PROCESS: Optional[subprocess.Popen[str]] = None
LOCK = threading.RLock()


def _json_response(handler: SimpleHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(handler: SimpleHTTPRequestHandler) -> Dict[str, Any]:
    try:
        length = int(handler.headers.get("Content-Length", "0") or "0")
    except ValueError:
        length = 0
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        data = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _append_log(job_id: str, line: str) -> None:
    with LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        log = job.setdefault("log", [])
        log.append(line.rstrip())
        if len(log) > 800:
            del log[: len(log) - 800]
        _update_progress_from_log(job, line.rstrip())


def _job_snapshot(job: Dict[str, Any], include_log: bool = True) -> Dict[str, Any]:
    snapshot = {key: value for key, value in job.items() if not key.startswith("_")}
    _refresh_progress_timing(snapshot)
    log = list(snapshot.get("log") or [])
    snapshot["log_length"] = len(log)
    snapshot["last_log_line"] = log[-1] if log else ""
    snapshot["log"] = log[-240:] if include_log else []
    return snapshot


def _run_job(job_id: str, command: list[str], env: Optional[Dict[str, str]] = None) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    control_path = str(merged_env.get("DUOMON_JOB_CONTROL_FILE") or "")
    if control_path:
        write_initial_control(control_path)
    with LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "running",
            "command": command,
            "started_at": time.time(),
            "finished_at": None,
            "returncode": None,
            "log": [],
            "paused": False,
            "stop_requested": False,
            "control_path": control_path,
            "progress": _initial_progress(job_id, env),
        }
    try:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with LOCK:
            current = JOBS.get(job_id)
            if current is not None:
                current["_process"] = process
        assert process.stdout is not None
        for line in process.stdout:
            _append_log(job_id, line)
        returncode = process.wait()
        with LOCK:
            job = JOBS[job_id]
            stop_requested = bool(job.get("stop_requested"))
            job["status"] = "stopped" if stop_requested else ("ok" if returncode == 0 else "failed")
            job["returncode"] = returncode
            job["finished_at"] = time.time()
            job["paused"] = False
            job.pop("_process", None)
            if returncode == 0 and not stop_requested:
                progress = job.get("progress") or {}
                _set_progress(
                    job, int(progress.get("total") or 1), f"{job_id.title()} finished"
                )
                if progress.get("kind") == "benchmark":
                    for opponent, opponent_state in (
                        progress.get("opponent_statuses") or {}
                    ).items():
                        if int(opponent_state.get("current") or 0) >= int(
                            opponent_state.get("total") or 0
                        ):
                            opponent_state["status"] = "done"
                            completed = list(progress.get("completed_opponents") or [])
                            if opponent not in completed:
                                completed.append(opponent)
                            progress["completed_opponents"] = completed
    except Exception as exc:
        _append_log(job_id, f"ERROR: {exc}")
        with LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["returncode"] = -1
            JOBS[job_id]["finished_at"] = time.time()
            JOBS[job_id]["paused"] = False
            JOBS[job_id].pop("_process", None)


def _start_job(
    job_id: str, command: list[str], env: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    with LOCK:
        existing = JOBS.get(job_id)
        if existing and existing.get("status") == "running":
            return _job_snapshot(existing)
    thread = threading.Thread(target=_run_job, args=(job_id, command, env), daemon=True)
    thread.start()
    time.sleep(0.05)
    with LOCK:
        return _job_snapshot(JOBS[job_id])


def _write_job_control(job: Dict[str, Any]) -> None:
    path = str(job.get("control_path") or "")
    if not path:
        return
    Path(path).write_text(
        json.dumps(
            {
                "paused": bool(job.get("paused")),
                "stop_requested": bool(job.get("stop_requested")),
                "updated_at": time.time(),
            }
        ),
        encoding="utf-8",
    )


def _control_benchmark(action: str) -> tuple[int, Dict[str, Any]]:
    with LOCK:
        job = JOBS.get("benchmark")
        if not job or job.get("status") != "running":
            return 409, {"error": "No benchmark is currently running."}
        if action == "pause":
            job["paused"] = True
            _write_job_control(job)
            _append_log("benchmark", "[control] action=pause_requested")
        elif action == "resume":
            job["paused"] = False
            _write_job_control(job)
            _append_log("benchmark", "[control] action=resume_requested")
        elif action == "stop":
            job["stop_requested"] = True
            job["paused"] = False
            _write_job_control(job)
            _append_log("benchmark", "[control] action=stop_requested")
            process = job.get("_process")
            if process is not None and process.poll() is None:
                process.terminate()
        else:
            return 404, {"error": f"Unknown benchmark control action: {action}"}
        return 202, {"job": _job_snapshot(job)}


def _showdown_running() -> bool:
    if SHOWDOWN_PROCESS is not None and SHOWDOWN_PROCESS.poll() is None:
        return True
    try:
        with socket.create_connection(("127.0.0.1", 8000), timeout=0.25):
            return True
    except OSError:
        return False


def _start_showdown() -> Dict[str, Any]:
    global SHOWDOWN_PROCESS
    with LOCK:
        if _showdown_running():
            return {
                "status": "running",
                "pid": SHOWDOWN_PROCESS.pid if SHOWDOWN_PROCESS is not None else None,
            }
        npm = "npm.cmd" if os.name == "nt" else "npm"
        SHOWDOWN_PROCESS = subprocess.Popen(
            [npm, "start", "--", "--no-security"],
            cwd=str(ROOT / "showdown-server"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return {"status": "starting", "pid": SHOWDOWN_PROCESS.pid}


class DemoHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class DemoHandler(SimpleHTTPRequestHandler):
    server_version = "DuoMonDemo/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            query = parse_qs(parsed.query)
            include_battles = _payload_enabled(query.get("battles", ["0"])[0], False)
            battles = _battle_results() if include_battles else []
            with LOCK:
                jobs = {job_id: _job_snapshot(job) for job_id, job in JOBS.items()}
                showdown_status = {
                    "running": _showdown_running(),
                    "pid": SHOWDOWN_PROCESS.pid if SHOWDOWN_PROCESS is not None else None,
                }
            summary = (
                _summary_for_rows(battles, include_turn_metrics=False) if include_battles else None
            )
            payload = {
                "jobs": jobs,
                "battles": battles,
                "summary": summary,
                "showdown": showdown_status,
                "server_time": time.time(),
            }
            _json_response(self, 200, payload)
            return
        if parsed.path == "/api/team":
            p1_text = _read_text_file(DEMO_TEAM_ROOT / "ally_p1.txt")
            p3_text = _read_text_file(DEMO_TEAM_ROOT / "ally_p3.txt")
            if not _looks_like_exported_set(p1_text):
                p1_text = _read_text_file(DEFAULT_TEAM_P1)
            if not _looks_like_exported_set(p3_text):
                p3_text = _read_text_file(DEFAULT_TEAM_P3)
            _json_response(
                self,
                200,
                {
                    "fixed_ally_team_enabled": True,
                    "mirror_opponent_team_enabled": False,
                    "ally_p1_team": p1_text,
                    "ally_p3_team": p3_text,
                },
            )
            return
        if parsed.path == "/api/battles":
            query = parse_qs(parsed.query)
            include_turn_metrics = _payload_enabled(query.get("turn_metrics", ["1"])[0], True)
            battles = _battle_results()
            _json_response(
                self,
                200,
                {
                    "battles": _filter_battles(battles, query),
                    "summary": _summary_for_rows(
                        battles, include_turn_metrics=include_turn_metrics
                    ),
                },
            )
            return
        if parsed.path == "/api/artifacts":
            _json_response(self, 200, {"artifacts": _latest_artifacts()})
            return
        if parsed.path.startswith("/api/battle_trace/"):
            tag = unquote(parsed.path.rsplit("/", 1)[-1])
            _json_response(self, 200, {"trace": _battle_trace(tag)})
            return
        if parsed.path.startswith("/api/replay/"):
            tag = unquote(parsed.path.rsplit("/", 1)[-1])
            replay_path = _find_replay_html(tag)
            if replay_path is None:
                self.send_error(404)
                return
            body = _rewrite_replay_html(replay_path.read_text(encoding="utf-8")).encode("utf-8")
            self._serve_file(replay_path, body, "text/html; charset=utf-8")
            return
        if parsed.path in {"/", "/index.html"}:
            self._serve_file(WEB_ROOT / "index.html")
            return
        if parsed.path.startswith("/showdown/"):
            self._serve_showdown(parsed.path)
            return
        target = _safe_file(WEB_ROOT, parsed.path)
        if target is not None:
            self._serve_file(target)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            payload = _read_json_body(self)
            if parsed.path == "/api/team/random":
                generated = _generate_random_ally_teams()
                if _payload_enabled(payload.get("persist"), True):
                    _write_demo_team("ally_p1.txt", generated["ally_p1_team"], DEFAULT_TEAM_P1)
                    _write_demo_team("ally_p3.txt", generated["ally_p3_team"], DEFAULT_TEAM_P3)
                _json_response(self, 200, generated)
                return
            if parsed.path == "/api/team/default":
                p1_text = _read_text_file(DEFAULT_TEAM_P1)
                p3_text = _read_text_file(DEFAULT_TEAM_P3)
                _write_demo_team("ally_p1.txt", p1_text, DEFAULT_TEAM_P1)
                _write_demo_team("ally_p3.txt", p3_text, DEFAULT_TEAM_P3)
                _json_response(
                    self,
                    200,
                    {
                        "fixed_ally_team_enabled": True,
                        "mirror_opponent_team_enabled": False,
                        "ally_p1_team": p1_text,
                        "ally_p3_team": p3_text,
                        "source": "curated default",
                    },
                )
                return
            if parsed.path == "/api/train":
                env, train_opponent, dataset_path = _train_env_from_payload(payload)
                if not _path_has_records(dataset_path):
                    _json_response(
                        self,
                        409,
                        {
                            "error": (
                                "No CTDE candidate dataset is available for the current training scope. "
                                "Run a benchmark or collection with these ally-team options first, then train the CTDE reranker."
                            ),
                            "dataset": dataset_path,
                        },
                    )
                    return
                if not _dataset_has_training_scope(dataset_path, train_opponent):
                    _json_response(
                        self,
                        409,
                        {
                            "error": (
                                "The CTDE dataset exists, but it does not contain examples for the selected model target. "
                                "Run a benchmark that includes SimpleHeuristics and/or Abyssal for this ally-team scope before training."
                            ),
                            "dataset": dataset_path,
                        },
                    )
                    return
                job = _start_job(
                    "train",
                    [PYTHON_EXE, "scripts/training/train_ctde.py", "--opponent", train_opponent],
                    env,
                )
                _json_response(self, 202, {"job": job})
                return
            if parsed.path == "/api/benchmark":
                _start_showdown()
                env = _benchmark_env_from_payload(payload)
                env["DUOMON_JOB_CONTROL_FILE"] = str(ROOT / "outputs" / "benchmark_control.json")
                job = _start_job("benchmark", [PYTHON_EXE, "-m", "duomon"], env)
                _json_response(self, 202, {"job": job})
                return
            if parsed.path in {
                "/api/benchmark/pause",
                "/api/benchmark/resume",
                "/api/benchmark/stop",
            }:
                action = parsed.path.rsplit("/", 1)[-1]
                status, response = _control_benchmark(action)
                _json_response(self, status, response)
                return
            if parsed.path == "/api/showdown/start":
                _json_response(self, 202, _start_showdown())
                return
            self.send_error(404)
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})

    def _serve_file(
        self, path: Path, body: Optional[bytes] = None, content_type: Optional[str] = None
    ) -> None:
        data = body if body is not None else path.read_bytes()
        ctype = content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_showdown(self, path: str) -> None:
        if path == "/showdown/blank/" or path.startswith("/showdown/blank/"):
            self._serve_file(
                WEB_ROOT / "index.html",
                b"<!doctype html><title>DuoMon</title>",
                "text/html; charset=utf-8",
            )
            return
        prefix = "/showdown/play.pokemonshowdown.com/"
        if path == "/showdown/" or path == prefix:
            path = prefix + "index.html"
        if not path.startswith(prefix):
            self.send_error(404)
            return
        relative = path[len(prefix) :]
        if relative.startswith("config/config.js"):
            self._serve_file(CONFIG_JS, _local_config_js(), "application/javascript; charset=utf-8")
            return
        target = _safe_file(PLAY_ROOT, relative)
        if target is None:
            self.send_error(404)
            return
        if target.name == "index.html":
            body = _rewrite_showdown_index(target.read_text(encoding="utf-8")).encode("utf-8")
            self._serve_file(target, body, "text/html; charset=utf-8")
            return
        self._serve_file(target)


def main() -> None:
    os.chdir(WEB_ROOT)
    if sys.stdout is None or sys.stderr is None:
        log = open(WEB_ROOT / "demo_runtime.log", "a", encoding="utf-8")
        sys.stdout = log
        sys.stderr = log
    server = DemoHTTPServer((HOST, PORT), DemoHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
