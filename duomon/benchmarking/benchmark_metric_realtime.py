from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from ..heuristic import json_safe
from .benchmark_metric_compute import compute_benchmark_metrics, compute_metrics_by_opponent
from .benchmark_metrics_context import iter_jsonl


def _result_turn_rows(result: Dict[str, Any], root_dir: str = ".") -> List[Dict[str, Any]]:
    tag = str(result.get("battle_tag") or "")
    raw_path = str(result.get("replay_log_path") or result.get("replay_path") or "")
    if not tag or not raw_path:
        return []
    path = raw_path if os.path.isabs(raw_path) else os.path.join(root_dir, raw_path)
    rows: List[Dict[str, Any]] = []
    for record in iter_jsonl(path) or []:
        if str(record.get("battle_tag") or "") == tag and "coordination" in record:
            rows.append(record)
    return rows


class BenchmarkMetricsSnapshot:

    def __init__(self, path: str, root_dir: str = ".") -> None:
        self.path = path
        self.root_dir = root_dir
        self.results: List[Dict[str, Any]] = []
        self.turn_rows: List[Dict[str, Any]] = []
        self._result_keys: set[str] = set()
        self._turn_keys: set[str] = set()

    def add_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        key = self._result_key(result)
        if key not in self._result_keys:
            self._result_keys.add(key)
            self.results.append(dict(result))
        for row in _result_turn_rows(result, self.root_dir):
            turn_key = self._turn_key(row)
            if turn_key not in self._turn_keys:
                self._turn_keys.add(turn_key)
                self.turn_rows.append(row)
        return self.write()

    def write(self) -> Dict[str, Any]:
        metrics = compute_benchmark_metrics(self.results, self.turn_rows)
        by_opponent = compute_metrics_by_opponent(self.results, self.turn_rows)
        payload = {
            "overall": metrics,
            "opponents": by_opponent,
            "results": list(self.results),
            "result_count": len(self.results),
            "turn_row_count": len(self.turn_rows),
            "updated_at": time.time(),
            "incremental": True,
        }
        directory = os.path.dirname(os.path.abspath(self.path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, indent=2)
        return payload

    @staticmethod
    def _result_key(result: Dict[str, Any]) -> str:
        tag = str(result.get("battle_tag") or "")
        if tag:
            return tag
        return "|".join(
            str(result.get(key, "")) for key in ("label", "opponent_kind", "battle_idx", "error")
        )

    @staticmethod
    def _turn_key(row: Dict[str, Any]) -> str:
        return "|".join(
            str(row.get(key, ""))
            for key in ("battle_tag", "benchmark_type", "turn", "mode", "action")
        )


__all__ = ["BenchmarkMetricsSnapshot"]
