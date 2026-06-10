from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from duomon.benchmarking.benchmark_metric_compute import (
    compute_benchmark_metrics,
    compute_metrics_by_opponent,
    load_turn_rows_for_results,
)
from duomon.core.jsonl import iter_jsonl
from web.server.progress import OPPONENTS, OPPONENT_DESCRIPTIONS, _bounded_int

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs"
ARTIFACT_ROOT = Path(os.environ.get("DUOMON_ARTIFACT_ROOT", os.environ.get("DUOMON_ARTIFACT_DIR", ROOT / "artifacts")))
SHOWDOWN_REPLAY_ROOT = OUTPUT_ROOT / "showdown_replays"

SUMMARY_CACHE: Dict[str, Any] = {"key": None, "summary": None}
RESULT_CACHE: Dict[str, Any] = {"key": None, "rows": None}
TRACE_CACHE: Dict[str, Any] = {"key": None, "trace": None}
LOCK = threading.RLock()


def _safe_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _latest_artifact_run_dir() -> Optional[Path]:
    root = ARTIFACT_ROOT if ARTIFACT_ROOT.is_absolute() else ROOT / ARTIFACT_ROOT
    candidates = [
        path
        for path in root.glob("experiments/*/*")
        if path.is_dir() and ((path / "metadata.json").exists() or (path / "profile").exists())
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _bytes_to_mib(value: Any) -> float:
    try:
        return round(float(value) / (1024.0 * 1024.0), 2)
    except Exception:
        return 0.0


def _latest_artifacts() -> Dict[str, Any]:
    run_dir = _latest_artifact_run_dir()
    if run_dir is None:
        return {"available": False}
    metadata_path = run_dir / "metadata.json"
    profile_candidates = sorted(
        (run_dir / "profile").glob("*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    manifest_path = run_dir / "battles" / "manifest.jsonl"
    profile_path = profile_candidates[0] if profile_candidates else None
    metadata = _safe_json(metadata_path) if metadata_path.exists() else {}
    profile = _safe_json(profile_path) if profile_path is not None else {}
    records = profile.get("records", []) if isinstance(profile, dict) else []
    battle_records = [
        record for record in records if isinstance(record, dict) and "battles" in record
    ]
    total_battles = sum(int(record.get("battles") or 0) for record in battle_records)
    total_elapsed = sum(float(record.get("elapsed_seconds") or 0.0) for record in battle_records)
    selected_traces = list((run_dir / "battles" / "selected").glob("**/*.jsonl.gz"))
    manifest_rows = 0
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest_rows = sum(1 for line in handle if line.strip())
        except OSError:
            manifest_rows = 0
    return {
        "available": True,
        "run_dir": str(run_dir),
        "metadata_path": str(metadata_path) if metadata_path.exists() else "",
        "profile_path": str(profile_path) if profile_path is not None else "",
        "manifest_path": str(manifest_path) if manifest_path.exists() else "",
        "run_name": metadata.get("run_name", run_dir.parent.name),
        "run_id": metadata.get("run_id", run_dir.name),
        "git_commit": (metadata.get("git") or {}).get("commit", "")[:12],
        "dirty": bool((metadata.get("git") or {}).get("dirty_status", "")),
        "seed": (metadata.get("config") or {}).get("seed"),
        "eval_battles_per_opponent": (metadata.get("extra") or {}).get(
            "eval_battles_per_opponent"
        ),
        "parallelism": (metadata.get("extra") or {}).get("parallelism"),
        "communication_enabled": (metadata.get("extra") or {}).get("communication_enabled"),
        "profile": {
            "elapsed_seconds": profile.get("elapsed_seconds"),
            "phase_count": len(records) if isinstance(records, list) else 0,
            "total_battles": total_battles,
            "battles_per_second": round(total_battles / total_elapsed, 4)
            if total_elapsed > 0
            else 0.0,
            "storage_mib": {
                key: _bytes_to_mib(value)
                for key, value in (profile.get("storage_bytes") or {}).items()
            },
        },
        "battles": {
            "metadata_rows": manifest_rows,
            "selected_traces": len(selected_traces),
        },
    }


def _latest_metrics_snapshot() -> tuple[Optional[Path], Dict[str, Any]]:
    candidates = list(OUTPUT_ROOT.glob("**/benchmark_metrics_summary.json"))
    candidates = [path for path in candidates if path.exists()]
    if not candidates:
        return None, {}
    path = max(candidates, key=lambda item: item.stat().st_mtime)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return path, {}
    return path, payload if isinstance(payload, dict) else {}


def _row_won(row: Dict[str, Any]) -> bool:
    return bool(row.get("p1_won", row.get("won", False)))


def _row_lost(row: Dict[str, Any]) -> bool:
    return bool(row.get("p1_lost", row.get("lost", False)))


def _turn_rows_for_results(
    rows: list[Dict[str, Any]], turn_rows: list[Dict[str, Any]]
) -> list[Dict[str, Any]]:
    tags = {str(row.get("battle_tag") or "") for row in rows if row.get("battle_tag")}
    if not tags:
        return []
    return [row for row in turn_rows if str(row.get("battle_tag") or "") in tags]


def _metrics_by_result_split(
    rows: list[Dict[str, Any]], turn_rows: list[Dict[str, Any]]
) -> Dict[str, Any]:
    splits = {
        "total": {
            "label": "Total",
            "description": "All finished benchmark battles in the selected result set.",
            "rows": rows,
        },
        "wins": {
            "label": "Wins",
            "description": "Only battles won by the allied DuoMon side.",
            "rows": [row for row in rows if _row_won(row)],
        },
        "losses": {
            "label": "Losses",
            "description": "Only battles lost by the allied DuoMon side.",
            "rows": [row for row in rows if _row_lost(row)],
        },
    }
    result: Dict[str, Any] = {}
    for key, split in splits.items():
        split_rows = list(split["rows"])
        split_turn_rows = _turn_rows_for_results(split_rows, turn_rows)
        summary = compute_benchmark_metrics(split_rows, split_turn_rows)
        result[key] = {
            "label": split["label"],
            "description": split["description"],
            "battle_count": len(split_rows),
            "turn_row_count": len(split_turn_rows),
            "metrics": summary.get("metrics", {}),
            "raw": summary.get("raw", {}),
        }
    return result


def _rows_for_opponent(rows: list[Dict[str, Any]], opponent: str) -> list[Dict[str, Any]]:
    return [row for row in rows if str(row.get("opponent_kind") or "").lower() == opponent]


def _rewrite_replay_html(text: str) -> str:
    text = text.replace("https://play.pokemonshowdown.com/", "/showdown/play.pokemonshowdown.com/")
    text = text.replace("http://pokemonshowdown.com/users/", "#")
    text = text.replace("https://github.com/hsahovic/poke-env", "#")
    text = text.replace("https://github.com/smogon/pokemon-showdown", "#")
    text = text.replace("https://github.com/hsahovic/poke-env/issues", "#")
    footer_start = text.find('<p style="text-align:center"> Replay created by')
    if footer_start >= 0:
        footer_end = text.find("</p>", footer_start)
        if footer_end >= 0:
            text = text[:footer_start] + text[footer_end + 4 :]
    bridge = """
<script>
(function () {
  const logData = document.querySelector(".battle-log-data")?.textContent || "";
  const tagMatch = logData.match(/>\\s*(battle-[^\\s|]+)/);
  const battleTag = tagMatch ? tagMatch[1] : (document.title || "");
  const maxTurn = Math.max(1, ...Array.from(logData.matchAll(/^\\|turn\\|(\\d+)/gm), item => Number(item[1] || 0)));
  let lastPayload = "";
  let subscribedBattle = null;

  function readObjectTurn() {
    const candidates = [
      window.Replays && window.Replays.battle,
      window.battle,
      window.room && window.room.battle,
      window.app && window.app.battle,
      window.Battle && window.Battle.currentBattle,
    ];
    for (const candidate of candidates) {
      const value = Number(candidate && (candidate.turn || candidate.curTurn || candidate.currentTurn));
      if (Number.isFinite(value) && value > 0) return Math.min(value, maxTurn);
    }
    return 0;
  }

  function readVisibleLogTurn() {
    const logs = Array.from(document.querySelectorAll(".battle-log:not(.battle-log-inline)"));
    const text = logs.map(item => item.innerText || item.textContent || "").join("\\n");
    const turns = Array.from(text.matchAll(/\\bTurn\\s+(\\d+)\\b/gi), item => Number(item[1] || 0)).filter(Boolean);
    if (turns.length) return Math.min(Math.max(...turns), maxTurn);
    return 1;
  }

  function publishTurn() {
    const turn = readObjectTurn() || readVisibleLogTurn();
    const payload = {
      type: "duomon:replay-turn",
      battle_tag: battleTag,
      turn: Math.max(1, Math.min(turn || 1, maxTurn)),
      max_turn: maxTurn,
    };
    const encoded = JSON.stringify(payload);
    if (encoded === lastPayload) return;
    lastPayload = encoded;
    if (window.parent && window.parent !== window) {
      window.parent.postMessage(payload, "*");
    }
  }

  function attachBattleSubscription() {
    const battle = window.Replays && window.Replays.battle;
    if (!battle || battle === subscribedBattle || typeof battle.subscribe !== "function") return;
    subscribedBattle = battle;
    battle.subscribe(() => setTimeout(publishTurn, 20));
    publishTurn();
  }

  document.addEventListener("click", () => setTimeout(publishTurn, 80), true);
  document.addEventListener("keyup", () => setTimeout(publishTurn, 80), true);
  new MutationObserver(() => setTimeout(publishTurn, 40)).observe(document.documentElement, {
    childList: true,
    subtree: true,
    characterData: true,
  });
  window.addEventListener("load", () => setTimeout(() => {
    attachBattleSubscription();
    publishTurn();
  }, 250));
  setInterval(() => {
    attachBattleSubscription();
    publishTurn();
  }, 250);
  publishTurn();
})();
</script>
"""
    if "duomon:replay-turn" not in text:
        text += bridge
    return text


def _replay_html_by_tag(replay_paths: Optional[list[Path]] = None) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    if replay_paths is None:
        replay_paths = (
            list(SHOWDOWN_REPLAY_ROOT.glob("*.html")) if SHOWDOWN_REPLAY_ROOT.exists() else []
        )
        replay_paths.extend(OUTPUT_ROOT.glob("runs/*/showdown_replays/*.html"))
    for path in sorted(replay_paths, key=lambda item: item.stat().st_mtime, reverse=True):
        stem = path.stem
        if "battle-" not in stem:
            continue
        tag = stem[stem.find("battle-") :]
        mapping.setdefault(tag, path)
    return mapping


def _find_replay_html(tag: str) -> Optional[Path]:
    tag = str(tag or "").strip()
    if not tag:
        return None
    candidates: list[Path] = []
    if SHOWDOWN_REPLAY_ROOT.exists():
        candidates.extend(SHOWDOWN_REPLAY_ROOT.glob(f"*{tag}.html"))
    candidates.extend(OUTPUT_ROOT.glob(f"runs/*/showdown_replays/*{tag}.html"))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _path_fingerprint(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
    items: list[tuple[str, int, int]] = []
    for path in paths:
        try:
            stat = path.stat()
            items.append((str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size)))
        except OSError:
            items.append((str(path), 0, 0))
    return tuple(sorted(items))


def _battle_results() -> list[Dict[str, Any]]:
    result_paths = list(OUTPUT_ROOT.glob("multi_independent_vs_*_multi_eval_vs_*_results.jsonl"))
    result_paths.extend(
        OUTPUT_ROOT.glob("runs/*/multi_independent_vs_*_multi_eval_vs_*_results.jsonl")
    )
    grouped_paths: Dict[Path, list[Path]] = {}
    latest_dir: Optional[Path] = None
    for path in result_paths:
        grouped_paths.setdefault(path.parent, []).append(path)
    if grouped_paths:
        latest_dir = max(
            grouped_paths,
            key=lambda directory: max(path.stat().st_mtime for path in grouped_paths[directory]),
        )
        result_paths = grouped_paths[latest_dir]
    result_paths = sorted(result_paths, key=lambda item: item.stat().st_mtime, reverse=True)
    snapshot_path, snapshot = _latest_metrics_snapshot()
    snapshot_results = snapshot.get("results", []) if isinstance(snapshot, dict) else []
    if isinstance(snapshot_results, list) and snapshot_results:
        latest_result_mtime = max(
            (path.stat().st_mtime for path in result_paths),
            default=0.0,
        )
        snapshot_mtime = snapshot_path.stat().st_mtime if snapshot_path else 0.0
        if snapshot_mtime >= latest_result_mtime:
            result_paths = []
    if latest_dir is not None and latest_dir != OUTPUT_ROOT:
        replay_paths = list((latest_dir / "showdown_replays").glob("*.html"))
    else:
        replay_paths = (
            list(SHOWDOWN_REPLAY_ROOT.glob("*.html")) if SHOWDOWN_REPLAY_ROOT.exists() else []
        )
    replay_paths = list(dict.fromkeys(replay_paths))
    cache_key = (_path_fingerprint(result_paths), _path_fingerprint(replay_paths))
    with LOCK:
        if RESULT_CACHE.get("key") == cache_key and RESULT_CACHE.get("rows") is not None:
            return list(RESULT_CACHE["rows"])

    replay_map = _replay_html_by_tag(replay_paths)
    rows: list[Dict[str, Any]] = []
    snapshot_records = snapshot_results if not result_paths else []
    sources = (
        [(snapshot_path or OUTPUT_ROOT / "benchmark_metrics_summary.json", snapshot_records)]
        if snapshot_records
        else [(path, list(iter_jsonl(path) or [])) for path in result_paths]
    )
    for result_path, records in sources:
        for record in records:
            tag = str(record.get("battle_tag") or "")
            if not tag:
                continue
            html_path = replay_map.get(tag)
            rows.append(
                {
                    "battle_tag": tag,
                    "opponent_kind": record.get("opponent_kind")
                    or str(record.get("benchmark_type") or "").replace("vs_", ""),
                    "benchmark_type": record.get("benchmark_type", ""),
                    "battle_idx": record.get("battle_idx"),
                    "finished": bool(record.get("finished", False)),
                    "won": bool(record.get("p1_won", False)),
                    "p1_won": bool(record.get("p1_won", False)),
                    "lost": bool(record.get("p1_lost", False)),
                    "p1_lost": bool(record.get("p1_lost", False)),
                    "error": record.get("error", ""),
                    "p1": record.get("p1", ""),
                    "p2": record.get("p2", ""),
                    "p3": record.get("p3", ""),
                    "p4": record.get("p4", ""),
                    "elapsed_seconds": record.get("elapsed_seconds"),
                    "fixed_ally_team_enabled": record.get(
                        "fixed_ally_team_enabled",
                        "fixedallies" in tag,
                    ),
                    "fixed_ally_team_hash": record.get("fixed_ally_team_hash", ""),
                    "data_namespace": record.get("data_namespace", ""),
                    "result_path": str(result_path),
                    "replay_log_path": record.get("replay_path", ""),
                    "replay_available": html_path is not None,
                    "replay_url": f"/api/replay/{tag}" if html_path is not None else "",
                }
            )
    rows = rows[:1000]
    with LOCK:
        RESULT_CACHE["key"] = cache_key
        RESULT_CACHE["rows"] = list(rows)
    return rows


def _summary_for_rows(
    rows: list[Dict[str, Any]], include_turn_metrics: bool = True
) -> Dict[str, Any]:
    snapshot_path, snapshot = _latest_metrics_snapshot()
    if include_turn_metrics and snapshot.get("incremental") and snapshot.get("overall"):
        snapshot_count = int(snapshot.get("result_count", 0) or 0)
        if snapshot_count >= len(rows):
            snapshot_cache_key = ("snapshot", str(snapshot_path), snapshot.get("updated_at"))
            with LOCK:
                if (
                    SUMMARY_CACHE.get("key") == snapshot_cache_key
                    and SUMMARY_CACHE.get("summary") is not None
                ):
                    return SUMMARY_CACHE["summary"]
            turn_rows = load_turn_rows_for_results(rows, str(ROOT))
            by_opponent = {}
            for opponent in OPPONENTS:
                metrics_payload = (
                    snapshot.get("opponents", {}).get(opponent, {})
                    if isinstance(snapshot.get("opponents"), dict)
                    else {}
                )
                raw = metrics_payload.get("raw", {}) if isinstance(metrics_payload, dict) else {}
                by_opponent[opponent] = {
                    "opponent_kind": opponent,
                    "description": OPPONENT_DESCRIPTIONS.get(
                        opponent, "Custom benchmark opponent."
                    ),
                    "total": int(raw.get("total", 0) or 0),
                    "finished": int(raw.get("finished", 0) or 0),
                    "wins": int(raw.get("wins", 0) or 0),
                    "losses": int(raw.get("losses", 0) or 0),
                    "errors": int(raw.get("errors", 0) or 0),
                    "winrate_all": float(raw.get("winrate_all", 0.0) or 0.0),
                    "winrate_finished": float(raw.get("winrate_finished", 0.0) or 0.0),
                    "metrics": metrics_payload.get("metrics", {})
                    if isinstance(metrics_payload, dict)
                    else {},
                    "metric_raw": raw,
                    "result_splits": _metrics_by_result_split(
                        _rows_for_opponent(rows, opponent), turn_rows
                    ),
                }
            overall_metrics = snapshot.get("overall", {})
            overall_raw = (
                overall_metrics.get("raw", {}) if isinstance(overall_metrics, dict) else {}
            )
            summary_payload = {
                "opponents": by_opponent,
                "overall": {
                    "opponent_kind": "overall",
                    "description": "Aggregate result across all selected benchmark opponents.",
                    "total": int(overall_raw.get("total", 0) or 0),
                    "finished": int(overall_raw.get("finished", 0) or 0),
                    "wins": int(overall_raw.get("wins", 0) or 0),
                    "losses": int(overall_raw.get("losses", 0) or 0),
                    "errors": int(overall_raw.get("errors", 0) or 0),
                    "winrate_all": float(overall_raw.get("winrate_all", 0.0) or 0.0),
                    "winrate_finished": float(overall_raw.get("winrate_finished", 0.0) or 0.0),
                    "metrics": overall_metrics.get("metrics", {})
                    if isinstance(overall_metrics, dict)
                    else {},
                    "metric_raw": overall_raw,
                },
                "metrics": overall_metrics.get("metrics", {})
                if isinstance(overall_metrics, dict)
                else {},
                "metric_raw": overall_raw,
                "metric_definitions": overall_metrics.get("definitions", {})
                if isinstance(overall_metrics, dict)
                else {},
                "result_splits": _metrics_by_result_split(rows, turn_rows),
                "incremental": True,
                "snapshot_path": str(snapshot_path) if snapshot_path else "",
            }
            with LOCK:
                SUMMARY_CACHE["key"] = snapshot_cache_key
                SUMMARY_CACHE["summary"] = summary_payload
            return summary_payload

    def fingerprint(path_value: Any) -> tuple[str, int, int]:
        path_text = str(path_value or "")
        if not path_text:
            return ("", 0, 0)
        path = Path(path_text)
        if not path.is_absolute():
            path = ROOT / path
        try:
            stat = path.stat()
            return (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
        except OSError:
            return (str(path), 0, 0)

    cache_key = (
        include_turn_metrics,
        tuple(sorted({fingerprint(row.get("result_path")) for row in rows})),
        tuple(sorted({fingerprint(row.get("replay_log_path")) for row in rows}))
        if include_turn_metrics
        else (),
        len(rows),
    )
    with LOCK:
        if SUMMARY_CACHE.get("key") == cache_key and SUMMARY_CACHE.get("summary") is not None:
            return SUMMARY_CACHE["summary"]

    turn_rows = load_turn_rows_for_results(rows, str(ROOT)) if include_turn_metrics else []
    by_opponent: Dict[str, Dict[str, Any]] = {}
    for opponent in OPPONENTS:
        by_opponent[opponent] = {
            "opponent_kind": opponent,
            "description": OPPONENT_DESCRIPTIONS.get(opponent, "Custom benchmark opponent."),
            "total": 0,
            "finished": 0,
            "wins": 0,
            "losses": 0,
            "errors": 0,
            "winrate_all": 0.0,
            "winrate_finished": 0.0,
        }

    for row in rows:
        opponent = str(row.get("opponent_kind") or "unknown")
        stats = by_opponent.setdefault(
            opponent,
            {
                "opponent_kind": opponent,
                "description": OPPONENT_DESCRIPTIONS.get(opponent, "Custom benchmark opponent."),
                "total": 0,
                "finished": 0,
                "wins": 0,
                "losses": 0,
                "errors": 0,
                "winrate_all": 0.0,
                "winrate_finished": 0.0,
            },
        )
        stats["total"] += 1
        stats["finished"] += 1 if row.get("finished") else 0
        stats["wins"] += 1 if row.get("won") else 0
        stats["losses"] += 1 if row.get("lost") else 0
        stats["errors"] += 1 if row.get("error") else 0

    overall = {
        "opponent_kind": "overall",
        "description": "Aggregate result across all selected benchmark opponents.",
        "total": 0,
        "finished": 0,
        "wins": 0,
        "losses": 0,
        "errors": 0,
        "winrate_all": 0.0,
        "winrate_finished": 0.0,
    }
    for stats in by_opponent.values():
        total = int(stats["total"])
        finished = int(stats["finished"])
        wins = int(stats["wins"])
        stats["winrate_all"] = (100.0 * wins / total) if total else 0.0
        stats["winrate_finished"] = (100.0 * wins / finished) if finished else 0.0
        for key in ("total", "finished", "wins", "losses", "errors"):
            overall[key] += int(stats[key])

    if overall["total"]:
        overall["winrate_all"] = 100.0 * overall["wins"] / overall["total"]
    if overall["finished"]:
        overall["winrate_finished"] = 100.0 * overall["wins"] / overall["finished"]

    metrics = compute_benchmark_metrics(rows, turn_rows)
    opponent_metrics = compute_metrics_by_opponent(rows, turn_rows)
    for opponent, stats in by_opponent.items():
        if opponent in opponent_metrics:
            stats["metrics"] = opponent_metrics[opponent].get("metrics", {})
            stats["metric_raw"] = opponent_metrics[opponent].get("raw", {})
        stats["result_splits"] = _metrics_by_result_split(
            _rows_for_opponent(rows, opponent), turn_rows
        )
    overall["metrics"] = metrics.get("metrics", {})
    overall["metric_raw"] = metrics.get("raw", {})
    summary_payload = {
        "opponents": by_opponent,
        "overall": overall,
        "metrics": metrics.get("metrics", {}),
        "metric_raw": metrics.get("raw", {}),
        "metric_definitions": metrics.get("definitions", {}),
        "result_splits": _metrics_by_result_split(rows, turn_rows),
    }
    with LOCK:
        SUMMARY_CACHE["key"] = cache_key
        SUMMARY_CACHE["summary"] = summary_payload
    return summary_payload


def _filter_battles(
    rows: list[Dict[str, Any]], query: Dict[str, list[str]]
) -> list[Dict[str, Any]]:
    opponent = (query.get("opponent", ["all"])[0] or "all").lower()
    result_filter = (query.get("result", ["all"])[0] or "all").lower()
    limit = _bounded_int(query.get("limit", [1000])[0], 1000, 1, 2000)

    if opponent != "all":
        rows = [row for row in rows if str(row.get("opponent_kind") or "").lower() == opponent]
    if result_filter == "win":
        rows = [row for row in rows if row.get("won")]
    elif result_filter == "loss":
        rows = [row for row in rows if row.get("lost")]
    elif result_filter == "finished":
        rows = [row for row in rows if row.get("finished")]
    elif result_filter == "error":
        rows = [row for row in rows if row.get("error")]
    return rows[:limit]


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_value(value: Any) -> Any:
    numeric = _as_float(value)
    if numeric is None:
        return value
    return round(numeric, 4)


def _compact_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "label": candidate.get("label") or candidate.get("recommended_label") or "",
        "signature": candidate.get("signature") or candidate.get("recommended_action") or "",
        "move_id": candidate.get("move_id") or candidate.get("recommended_move") or "",
        "kind": candidate.get("kind", ""),
        "score": _round_value(candidate.get("score", candidate.get("adjusted_score"))),
        "base_score": _round_value(candidate.get("base_score")),
        "adjustment": _round_value(candidate.get("adjustment")),
        "target": candidate.get("target") or candidate.get("target_species") or "",
        "target_slot": candidate.get("target_slot"),
        "damage_sum": _round_value(candidate.get("damage_sum", candidate.get("self_damage"))),
        "ko_sum": _round_value(candidate.get("ko_sum")),
        "accuracy": _round_value(candidate.get("accuracy")),
        "priority": _round_value(candidate.get("priority")),
        "protect": bool(candidate.get("protect", False)),
        "tera": bool(candidate.get("tera", False)),
        "support": bool(candidate.get("support", False)),
    }


def _compact_proposal(proposal: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "strategy": proposal.get("strategy", ""),
        "speech_act": proposal.get("speech_act", ""),
        "target_species": proposal.get("target_species", ""),
        "target_slot": proposal.get("target_slot"),
        "recommended_label": proposal.get("recommended_label", ""),
        "recommended_action": proposal.get("recommended_action", ""),
        "recommended_move": proposal.get("recommended_move", ""),
        "confidence": _round_value(proposal.get("confidence")),
        "self_damage": _round_value(proposal.get("self_damage")),
        "solo_ko": bool(proposal.get("solo_ko", False)),
        "solo_claim": bool(proposal.get("solo_claim", False)),
        "partner_required": bool(proposal.get("partner_required", False)),
        "communication_role": proposal.get("communication_role", ""),
        "rationale": proposal.get("rationale", ""),
    }


def _compact_message(message: Dict[str, Any]) -> Dict[str, Any]:
    content = message.get("content") if isinstance(message.get("content"), dict) else {}
    selected = content.get("selected") if isinstance(content.get("selected"), dict) else {}
    shared = (
        content.get("shared_joint_plan")
        if isinstance(content.get("shared_joint_plan"), dict)
        else {}
    )
    accepted = (
        content.get("accepted_partner_plan")
        if isinstance(content.get("accepted_partner_plan"), dict)
        else {}
    )
    team_plan = content.get("team_plan") if isinstance(content.get("team_plan"), dict) else {}
    return {
        "agent": message.get("agent", ""),
        "role": message.get("role", ""),
        "speech_act": message.get("speech_act", ""),
        "top_strategy": message.get("top_strategy", ""),
        "rationale": message.get("rationale", []),
        "proposals": [
            _compact_proposal(item)
            for item in (message.get("proposals") or [])[:5]
            if isinstance(item, dict)
        ],
        "top_candidates": [
            _compact_candidate(item)
            for item in (message.get("top_candidates") or [])[:6]
            if isinstance(item, dict)
        ],
        "commitment": {
            "decision": content.get("decision", ""),
            "reason": content.get("reason", ""),
            "selected": _compact_candidate(selected) if selected else {},
            "accepted_partner_plan": _compact_proposal(accepted) if accepted else {},
            "team_plan": _compact_proposal(team_plan) if team_plan else {},
            "shared_joint_plan": {
                "reason": shared.get("reason", ""),
                "pair_score": _round_value(shared.get("pair_score")),
                "pair_bonus": _round_value(shared.get("pair_bonus")),
                "local_label": shared.get("local_label", ""),
                "partner_label": shared.get("partner_label", ""),
                "local_agent": shared.get("local_agent", ""),
                "partner_agent": shared.get("partner_agent", ""),
                "local_signature": shared.get("local_signature", ""),
                "partner_signature": shared.get("partner_signature", ""),
            },
            "communication_diagnostics": content.get("communication_diagnostics", {}),
        }
        if content
        else {},
    }


def _compact_decision_diagnostics(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    candidate_summary = diagnostics.get("candidate_gate_summary") or {}
    return {
        "selected_signature": diagnostics.get("selected_signature", ""),
        "selected_label": diagnostics.get("selected_label", ""),
        "selected_move_id": diagnostics.get("selected_move_id", ""),
        "selected_score": _round_value(diagnostics.get("selected_score")),
        "risk_context": diagnostics.get("risk_context", {}),
        "selected_gates": diagnostics.get("selected_gates", {}),
        "top_candidates": [
            _compact_candidate(item)
            for item in (candidate_summary.get("top_candidates") or [])[:8]
            if isinstance(item, dict)
        ],
        "rejected_tera_candidates": [
            _compact_candidate(item)
            for item in (candidate_summary.get("rejected_tera_candidates") or [])[:5]
            if isinstance(item, dict)
        ],
    }


def _trace_paths_for_battle(tag: str) -> list[Path]:
    rows = [row for row in _battle_results() if str(row.get("battle_tag") or "") == tag]
    paths: list[Path] = []
    for row in rows:
        raw_path = str(row.get("replay_log_path") or row.get("replay_path") or "")
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            paths.append(path)
    return list(dict.fromkeys(paths))


def _battle_trace(tag: str) -> Dict[str, Any]:
    tag = str(tag or "").strip()
    if not tag:
        return {"battle_tag": "", "available": False, "turns": []}
    paths = _trace_paths_for_battle(tag)
    cache_key = (tag, _path_fingerprint(paths))
    with LOCK:
        if TRACE_CACHE.get("key") == cache_key and TRACE_CACHE.get("trace") is not None:
            return TRACE_CACHE["trace"]
    turn_map: Dict[int, Dict[str, Any]] = {}
    raw_count = 0
    for path in paths:
        for record in iter_jsonl(path) or []:
            if str(record.get("battle_tag") or "") != tag or "coordination" not in record:
                continue
            raw_count += 1
            turn = int(record.get("turn") or 0)
            messages = [item for item in (record.get("messages") or []) if isinstance(item, dict)]
            own_agent = str((messages[0] if messages else {}).get("agent") or "")
            coordination = record.get("coordination") or {}
            diagnostics = record.get("decision_diagnostics") or {}
            turn_payload = turn_map.setdefault(
                turn,
                {
                    "turn": turn,
                    "agents": [],
                },
            )
            turn_payload["agents"].append(
                {
                    "agent": own_agent or "unknown-agent",
                    "action": record.get("action", ""),
                    "value": _round_value(record.get("value")),
                    "mode": record.get("mode", ""),
                    "selected": {
                        "label": diagnostics.get("selected_label", ""),
                        "signature": diagnostics.get("selected_signature", ""),
                        "move_id": diagnostics.get("selected_move_id", ""),
                        "score": _round_value(diagnostics.get("selected_score")),
                    },
                    "protocol": {
                        "used": coordination.get("protocol_used", ""),
                        "reason": coordination.get("protocol_reason", ""),
                        "resolved_by": coordination.get("resolved_by", ""),
                        "veto_reason": coordination.get("veto_reason", ""),
                        "communication_gain": _round_value(
                            coordination.get("communication_gain")
                        ),
                        "message_agreement": bool(coordination.get("message_agreement", False)),
                        "message_conflict": bool(coordination.get("message_conflict", False)),
                        "plan_consistency": bool(coordination.get("plan_consistency", False)),
                    },
                    "active": {
                        "allies": record.get("my_active", []),
                        "opponents": record.get("opp_active", []),
                    },
                    "coordination": {
                        key: coordination.get(key)
                        for key in [
                            "damage_sum",
                            "ko_prob_sum",
                            "partner_damage_risk",
                            "focus_fire",
                            "split_targets",
                            "double_switch",
                            "support_without_attack",
                            "threat_response",
                            "saved_partner_attempt",
                            "role_left",
                            "role_right",
                            "teammate_alignment",
                        ]
                    },
                    "messages": [_compact_message(message) for message in messages],
                    "decision_diagnostics": _compact_decision_diagnostics(diagnostics),
                    "raw": {
                        "coordination": coordination,
                        "decision_diagnostics": diagnostics,
                        "messages": messages,
                    },
                }
            )
    turns = [turn_map[key] for key in sorted(turn_map)]
    trace_payload = {
        "battle_tag": tag,
        "available": bool(turns),
        "source_paths": [str(path) for path in paths],
        "raw_record_count": raw_count,
        "turn_count": len(turns),
        "turns": turns,
    }
    with LOCK:
        TRACE_CACHE["key"] = cache_key
        TRACE_CACHE["trace"] = trace_payload
    return trace_payload


__all__ = [
    "_battle_trace",
    "_battle_results",
    "_filter_battles",
    "_find_replay_html",
    "_rewrite_replay_html",
    "_summary_for_rows",
]
