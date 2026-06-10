from __future__ import annotations

import re
import time
from typing import Any, Dict, Optional

from duomon.benchmarking.benchmark_opponents import OPPONENT_DESCRIPTIONS

OPPONENTS = ("random", "maxpower", "simpleheuristics", "abyssal")
DEFAULT_BENCHMARK_BATTLES = 200


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def _canonical_opponent(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    aliases = {
        "random": "random",
        "maxpower": "maxpower",
        "simple": "simpleheuristics",
        "simpleheuristics": "simpleheuristics",
        "abyssal": "abyssal",
    }
    return aliases.get(normalized, normalized)


def _initial_progress(job_id: str, env: Optional[Dict[str, str]]) -> Dict[str, Any]:
    if job_id == "benchmark":
        battles = _bounded_int(
            (env or {}).get("DUOMON_BATTLES_PER_OPPONENT"),
            DEFAULT_BENCHMARK_BATTLES,
            1,
            2000,
        )
        opponents = [
            _canonical_opponent(item)
            for item in str((env or {}).get("DUOMON_BENCHMARK_SUITE") or ",".join(OPPONENTS)).split(
                ","
            )
            if _canonical_opponent(item) in OPPONENTS
        ] or list(OPPONENTS)
        opponent_statuses = {
            opponent: {
                "opponent_kind": opponent,
                "label": opponent,
                "description": OPPONENT_DESCRIPTIONS.get(opponent, "Custom benchmark opponent."),
                "status": "pending",
                "current": 0,
                "total": battles,
                "wins": 0,
                "losses": 0,
                "errors": 0,
                "winrate_finished": 0.0,
            }
            for opponent in opponents
        }
        return {
            "kind": "benchmark",
            "current": 0,
            "total": battles * len(opponents),
            "percent": 0.0,
            "label": "Starting benchmark",
            "opponents": opponents,
            "phase_total": battles,
            "active_opponent": None,
            "completed_opponents": [],
            "opponent_statuses": opponent_statuses,
            "elapsed_seconds": 0.0,
            "eta_seconds": None,
        }
    if job_id == "train":
        target = str((env or {}).get("DUOMON_DASHBOARD_TRAIN_TARGET") or "both").lower()
        total = 2 if target == "both" else 1
        return {
            "kind": "train",
            "current": 0,
            "total": total,
            "percent": 0.0,
            "label": "Starting CTDE training",
            "target": target,
            "completed": [],
            "elapsed_seconds": 0.0,
            "eta_seconds": None,
        }
    return {
        "kind": job_id,
        "current": 0,
        "total": 1,
        "percent": 0.0,
        "label": "Starting",
    }


def _refresh_progress_timing(job: Dict[str, Any]) -> None:
    progress = job.get("progress") or {}
    started_at = float(job.get("started_at") or time.time())
    finished_at = job.get("finished_at")
    end_time = float(finished_at or time.time())
    elapsed = max(0.0, end_time - started_at)
    current = int(progress.get("current") or 0)
    total = int(progress.get("total") or 0)
    progress["elapsed_seconds"] = round(elapsed, 1)
    if job.get("status") == "running" and current > 0 and total > current and elapsed > 0:
        progress["eta_seconds"] = round((total - current) / (current / elapsed), 1)
    elif total and current >= total:
        progress["eta_seconds"] = 0.0
    else:
        progress["eta_seconds"] = None


def _set_progress(job: Dict[str, Any], current: int, label: str) -> None:
    progress = job.setdefault("progress", {"current": 0, "total": 1, "percent": 0.0, "label": ""})
    total = max(1, int(progress.get("total") or 1))
    current = max(0, min(total, int(current)))
    progress["current"] = current
    progress["percent"] = round(100.0 * current / total, 1)
    progress["label"] = label
    _refresh_progress_timing(job)


def _opponent_progress(progress: Dict[str, Any], opponent: str) -> Dict[str, Any]:
    statuses = progress.setdefault("opponent_statuses", {})
    phase_total = int(progress.get("phase_total") or 0)
    return statuses.setdefault(
        opponent,
        {
            "opponent_kind": opponent,
            "label": opponent,
            "description": OPPONENT_DESCRIPTIONS.get(opponent, "Custom benchmark opponent."),
            "status": "pending",
            "current": 0,
            "total": phase_total,
            "wins": 0,
            "losses": 0,
            "errors": 0,
            "winrate_finished": 0.0,
        },
    )


def _update_progress_from_log(job: Dict[str, Any], line: str) -> None:
    progress = job.get("progress") or {}
    if progress.get("kind") == "benchmark":
        match = re.search(r"\[benchmark\]\s+progress=(\d+)/(\d+)\s+opponent=([^\s]+)", line)
        if match:
            done = int(match.group(1))
            phase_total = int(match.group(2))
            opponent = _canonical_opponent(match.group(3))
            result_match = re.search(r"\bresult=([^\s]+)", line)
            result = str(result_match.group(1) if result_match else "").lower()
            opponents = list(progress.get("opponents") or [])
            try:
                opponent_index = opponents.index(opponent)
            except ValueError:
                opponent_index = 0
            current = opponent_index * phase_total + done
            progress["active_opponent"] = opponent
            progress["phase_total"] = phase_total
            opponent_state = _opponent_progress(progress, opponent)
            opponent_state["status"] = "running"
            opponent_state["current"] = max(int(opponent_state.get("current") or 0), done)
            opponent_state["total"] = phase_total
            if result == "win":
                opponent_state["wins"] = int(opponent_state.get("wins") or 0) + 1
            elif result == "loss":
                opponent_state["losses"] = int(opponent_state.get("losses") or 0) + 1
            elif result and result not in {"draw", "tie"}:
                opponent_state["errors"] = int(opponent_state.get("errors") or 0) + 1
            counted = max(
                1,
                int(opponent_state.get("wins") or 0)
                + int(opponent_state.get("losses") or 0)
                + int(opponent_state.get("errors") or 0),
            )
            opponent_state["winrate_finished"] = round(
                100.0 * int(opponent_state.get("wins") or 0) / counted, 1
            )
            if done >= phase_total:
                opponent_state["status"] = "done"
                completed = list(progress.get("completed_opponents") or [])
                if opponent not in completed:
                    completed.append(opponent)
                progress["completed_opponents"] = completed
            _set_progress(job, current, f"Benchmarking {match.group(3)} ({done}/{phase_total})")
            return
        match = re.search(r"\[phase\]\s+action=start\s+opponent=([^\s]+)\s+battles=(\d+)", line)
        if match:
            opponent = _canonical_opponent(match.group(1))
            opponents = list(progress.get("opponents") or [])
            try:
                current = opponents.index(opponent) * int(
                    progress.get("phase_total") or match.group(2)
                )
            except ValueError:
                current = int(progress.get("current") or 0)
            progress["active_opponent"] = opponent
            opponent_state = _opponent_progress(progress, opponent)
            opponent_state["status"] = "running"
            opponent_state["total"] = int(match.group(2))
            _set_progress(job, current, f"Starting {match.group(1)} benchmark")
            return
    if progress.get("kind") == "train":
        match = re.search(r"\[training\]\s+action=start\s+opponent=([^\s]+)", line)
        if match:
            _set_progress(
                job,
                int(progress.get("current") or 0),
                f"Training {match.group(1)} model",
            )
            return
        match = re.search(r"\[training\]\s+action=finished\s+opponent=([^\s]+)", line)
        if match:
            completed = set(progress.get("completed") or [])
            completed.add(str(match.group(1)).lower())
            progress["completed"] = sorted(completed)
            _set_progress(job, len(completed), f"Finished {match.group(1)} model")
