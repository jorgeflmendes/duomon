from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from ..config import AgentConfig, ensure_parent_dir
from ..heuristic import json_safe


TRACKED_PACKAGES = ("numpy", "poke-env", "torch", "websockets")


def artifact_run_dir(config: AgentConfig) -> str:
    root = str(getattr(config, "artifact_root", "artifacts") or "artifacts")
    run_name = str(getattr(config, "run_name", "default") or "default")
    run_id = str(getattr(config, "run_id", "default") or "default")
    return os.path.join(root, "experiments", run_name, run_id)


def _safe_run_git(args: Sequence[str], cwd: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def dependency_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {"python": sys.version.split()[0]}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def write_run_metadata(
    config: AgentConfig,
    run_kind: str,
    benchmark_suite: Iterable[str] = (),
    extra: Dict[str, Any] | None = None,
) -> str:
    path = os.path.join(artifact_run_dir(config), "metadata.json")
    cwd = os.getcwd()
    payload = {
        "schema": "duomon.run_metadata.v1",
        "run_kind": run_kind,
        "run_name": str(getattr(config, "run_name", "")),
        "run_id": str(getattr(config, "run_id", "")),
        "created_at": time.time(),
        "platform": platform.platform(),
        "git": {
            "commit": _safe_run_git(["rev-parse", "HEAD"], cwd),
            "dirty_status": _safe_run_git(["status", "--short"], cwd),
        },
        "dependencies": dependency_versions(),
        "benchmark_suite": list(benchmark_suite),
        "config": json_safe(asdict(config)),
    }
    if extra:
        payload["extra"] = json_safe(extra)
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def cleanup_replay_shards(paths: Iterable[str], keep: bool = False) -> int:
    if keep:
        return 0
    removed = 0
    parents: set[Path] = set()
    for raw_path in paths:
        path = Path(str(raw_path or ""))
        if not path.exists() or not path.is_file():
            continue
        try:
            parents.add(path.parent)
            path.unlink()
            removed += 1
        except OSError:
            continue
    for parent in sorted(parents, key=lambda item: len(str(item)), reverse=True):
        try:
            parent.rmdir()
        except OSError:
            pass
    return removed


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []

    def iterator() -> Iterable[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    return iterator()


def _battle_rows_by_tag(replay_path: str) -> Dict[str, List[Dict[str, Any]]]:
    rows: Dict[str, List[Dict[str, Any]]] = {}
    for row in _read_jsonl(replay_path):
        tag = str(row.get("battle_tag") or "")
        if tag:
            rows.setdefault(tag, []).append(row)
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _max_turn(rows: Sequence[Dict[str, Any]]) -> int:
    turns: List[int] = []
    for row in rows:
        try:
            turns.append(int(row.get("turn") or 0))
        except Exception:
            continue
    return max(turns, default=0)


def _avg_communication(rows: Sequence[Dict[str, Any]]) -> float:
    values: List[float] = []
    for row in rows:
        coordination = row.get("coordination") or {}
        if isinstance(coordination, dict):
            values.append(_safe_float(coordination.get("communication_gain"), 0.0))
    return sum(values) / len(values) if values else 0.0


def _selection_reasons(result: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> List[str]:
    reasons: List[str] = []
    if result.get("error") or not bool(result.get("finished", False)):
        reasons.append("unexpected_failure")
    if bool(result.get("p1_won", False)):
        reasons.append("win")
    turn_count = _max_turn(rows)
    if turn_count >= 35:
        reasons.append("close_match")
    communication_score = _avg_communication(rows)
    if communication_score >= 0.25:
        reasons.append("communication_success")
    if communication_score <= -0.25:
        reasons.append("communication_failure")
    return reasons


def _battle_metadata(
    config: AgentConfig,
    result: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    selected: bool,
    reasons: Sequence[str],
) -> Dict[str, Any]:
    won = bool(result.get("p1_won", False))
    lost = bool(result.get("p1_lost", False))
    return {
        "schema": "duomon.battle_metadata.v1",
        "battle_id": str(result.get("battle_tag") or ""),
        "run_id": str(getattr(config, "run_id", "")),
        "timestamp": time.time(),
        "seed": int(getattr(config, "seed", 0) or 0),
        "agents": [str(result.get("p1") or "p1"), str(result.get("p3") or "p3")],
        "opponent": str(result.get("opponent_kind") or ""),
        "result": "win" if won else "loss" if lost else "draw_or_unfinished",
        "reward": 1.0 if won else -1.0 if lost else 0.0,
        "turns": _max_turn(rows),
        "cooperation_score": _avg_communication(rows),
        "communication_score": _avg_communication(rows),
        "novelty_score": 0.0,
        "benchmark_id": str(result.get("benchmark_type") or ""),
        "checkpoint_id": "",
        "saved_full_trace": bool(selected),
        "selection_reasons": list(reasons),
        "result_record": json_safe(result),
    }


def _trace_path(config: AgentConfig, battle_id: str, reasons: Sequence[str]) -> str:
    reason = str(reasons[0] if reasons else "samples")
    safe_reason = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in reason)
    safe_id = "".join(ch if ch.isalnum() or ch in "_-." else "_" for ch in battle_id)
    suffix = ".jsonl.gz" if bool(getattr(config, "compress_selected_battles", True)) else ".jsonl"
    return os.path.join(
        artifact_run_dir(config),
        "battles",
        "selected",
        safe_reason,
        f"{safe_id}{suffix}",
    )


def _write_trace(path: str, rows: Sequence[Dict[str, Any]], compress: bool) -> None:
    ensure_parent_dir(path)
    if compress:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(json_safe(row), separators=(",", ":")) + "\n")
        return
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), separators=(",", ":")) + "\n")


def write_battle_artifacts(
    config: AgentConfig,
    results: Sequence[Dict[str, Any]],
    aggregate_replay_path: str,
    benchmark_id: str,
) -> Dict[str, Any]:
    rows_by_tag = _battle_rows_by_tag(aggregate_replay_path)
    summary_path = os.path.join(
        artifact_run_dir(config), "battles", "summaries", f"{benchmark_id}_metadata.jsonl"
    )
    manifest_path = os.path.join(artifact_run_dir(config), "battles", "manifest.jsonl")
    max_full = max(0, int(getattr(config, "max_full_battles_per_run", 500) or 0))
    selected_count = 0
    total_count = 0
    retained_tags: set[str] = set()
    ensure_parent_dir(summary_path)
    ensure_parent_dir(manifest_path)
    with (
        open(summary_path, "w", encoding="utf-8") as summary_handle,
        open(manifest_path, "a", encoding="utf-8") as manifest_handle,
    ):
        for result in results:
            battle_id = str(result.get("battle_tag") or "")
            rows = rows_by_tag.get(battle_id, [])
            reasons = _selection_reasons(result, rows)
            selected = bool(reasons) and (max_full <= 0 or selected_count < max_full)
            metadata_row = _battle_metadata(config, result, rows, selected, reasons)
            summary_handle.write(json.dumps(json_safe(metadata_row), separators=(",", ":")) + "\n")
            manifest_handle.write(json.dumps(json_safe(metadata_row), separators=(",", ":")) + "\n")
            total_count += 1
            if selected and rows:
                retained_tags.add(battle_id)
                path = _trace_path(config, battle_id, reasons)
                _write_trace(
                    path,
                    rows,
                    compress=bool(getattr(config, "compress_selected_battles", True)),
                )
                selected_count += 1
    if str(getattr(config, "replay_retention", "all") or "all").lower() == "selected_only":
        retained_rows: List[Dict[str, Any]] = []
        for battle_id in sorted(retained_tags):
            retained_rows.extend(rows_by_tag.get(battle_id, []))
        with open(aggregate_replay_path, "w", encoding="utf-8") as handle:
            for row in retained_rows:
                handle.write(json.dumps(json_safe(row), separators=(",", ":")) + "\n")
    return {
        "summary_path": summary_path,
        "manifest_path": manifest_path,
        "total_battles": total_count,
        "selected_full_traces": selected_count,
        "retained_aggregate_replay_tags": len(retained_tags)
        if str(getattr(config, "replay_retention", "all") or "all").lower() == "selected_only"
        else "all",
    }


def checkpoint_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_checkpoint(
    config: Any,
    checkpoint_path: str,
    checkpoint_name: str,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return {"registered": False, "reason": "missing_checkpoint"}
    root = Path(str(config.output_dir)) / "checkpoints"
    latest_dir = root / "latest"
    best_dir = root / "best"
    latest_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    latest_path = latest_dir / Path(checkpoint_path).name
    shutil.copy2(checkpoint_path, latest_path)
    digest = checkpoint_digest(checkpoint_path)
    manifest_path = root / "manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    manifest.setdefault("schema", "duomon.checkpoint_manifest.v1")
    manifest.setdefault("best", {})
    manifest["latest"] = {
        "name": checkpoint_name,
        "path": str(latest_path.as_posix()),
        "source_path": checkpoint_path,
        "sha256": digest,
        "metrics": json_safe(metrics),
        "updated_at": time.time(),
    }
    for metric_name, raw_value in metrics.items():
        value = _safe_float(raw_value, 0.0)
        current = manifest["best"].get(metric_name)
        if current is None or value > _safe_float(current.get("value"), float("-inf")):
            best_path = best_dir / f"best_{metric_name}_{Path(checkpoint_path).name}"
            shutil.copy2(checkpoint_path, best_path)
            manifest["best"][metric_name] = {
                "name": checkpoint_name,
                "value": value,
                "path": str(best_path.as_posix()),
                "sha256": digest,
                "updated_at": time.time(),
            }
    manifest_path.write_text(json.dumps(json_safe(manifest), indent=2), encoding="utf-8")
    return {"registered": True, "manifest_path": str(manifest_path)}


__all__ = [
    "cleanup_replay_shards",
    "artifact_run_dir",
    "dependency_versions",
    "register_checkpoint",
    "write_battle_artifacts",
    "write_run_metadata",
]
