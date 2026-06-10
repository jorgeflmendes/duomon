from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict


BASE_DIR = Path(__file__).resolve().parents[2]


def ensure_repo_on_path() -> None:
    root = str(BASE_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)


def _default_path(env_name: str, *parts: str) -> str:
    return os.environ.get(env_name, str(BASE_DIR.joinpath(*parts)))


def default_ctde_paths() -> Dict[str, str]:
    out_dir = _default_path("DUOMON_CTDE_OUT_DIR", "learned_weights", "outcome_ctde_100_each")
    return {
        "dataset_path": _default_path(
            "DUOMON_CTDE_JOINT_DATASET_PATH",
            "learned_weights",
            "ctde_joint_examples.jsonl",
        ),
        "outcomes_path": _default_path(
            "DUOMON_CTDE_OUTCOMES_PATH", "learned_weights", "ctde_outcomes.jsonl"
        ),
        "results_dir": _default_path("DUOMON_CTDE_RESULTS_DIR", "outputs"),
        "out_dir": out_dir,
        "split_path": os.environ.get(
            "DUOMON_CTDE_SPLIT_PATH", str(Path(out_dir) / "ctde_split.json")
        ),
    }
