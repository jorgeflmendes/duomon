from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Iterable, List


def iter_jsonl(path: str | os.PathLike[str], *, strict: bool = False) -> Iterable[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception as exc:
                if strict:
                    raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc
                continue
            if isinstance(record, dict):
                yield record


def load_jsonl(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path) or [])


def count_lines(path: str | os.PathLike[str]) -> int:
    if not path or not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def tag_is_validation(tag: str, validation_fraction: float, seed: int = 42) -> bool:
    try:
        fraction = max(0.0, min(0.95, float(validation_fraction)))
    except (TypeError, ValueError):
        fraction = 0.0
    if not tag or fraction <= 0.0:
        return False
    digest = hashlib.sha256(f"{int(seed)}:{tag}".encode("utf-8")).hexdigest()
    value = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    return value < fraction


__all__ = ["count_lines", "iter_jsonl", "load_jsonl", "tag_is_validation"]
