from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from ..config import ensure_parent_dir
from ..heuristic import json_safe


def directory_size_bytes(path: str) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    for item in root.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


class RunProfiler:

    def __init__(self, enabled: bool, output_dir: str, run_kind: str) -> None:
        self.enabled = bool(enabled)
        self.output_dir = output_dir
        self.run_kind = run_kind
        self.started_at = time.time()
        self.records: List[Dict[str, Any]] = []
        self.counters: Dict[str, float] = {}
        self.storage: Dict[str, int] = {}

    @classmethod
    def from_config(cls, config: Any, run_kind: str) -> "RunProfiler":
        artifact_root = str(getattr(config, "artifact_root", "artifacts") or "artifacts")
        run_name = str(getattr(config, "run_name", "default") or "default")
        run_id = str(getattr(config, "run_id", "default") or "default")
        output_dir = os.path.join(artifact_root, "experiments", run_name, run_id, "profile")
        return cls(bool(getattr(config, "profiling_enabled", False)), output_dir, run_kind)

    def record_phase(self, name: str, elapsed_seconds: float, **fields: Any) -> None:
        if not self.enabled:
            return
        record = {
            "name": name,
            "elapsed_seconds": round(float(elapsed_seconds), 6),
            "timestamp": time.time(),
        }
        record.update(fields)
        self.records.append(record)

    def increment(self, name: str, amount: float = 1.0) -> None:
        if not self.enabled:
            return
        self.counters[name] = self.counters.get(name, 0.0) + float(amount)

    def record_storage(self, label: str, path: str) -> None:
        if not self.enabled:
            return
        self.storage[label] = directory_size_bytes(path)

    def payload(self) -> Dict[str, Any]:
        elapsed = max(0.0, time.time() - self.started_at)
        return {
            "schema": "duomon.profile.v1",
            "run_kind": self.run_kind,
            "started_at": self.started_at,
            "elapsed_seconds": round(elapsed, 6),
            "records": self.records,
            "counters": self.counters,
            "storage_bytes": self.storage,
        }

    def write(self, filename: str = "profile.json") -> str:
        if not self.enabled:
            return ""
        path = os.path.join(self.output_dir, filename)
        ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(json_safe(self.payload()), handle, indent=2, sort_keys=True)
        return path


__all__ = ["RunProfiler", "directory_size_bytes"]
