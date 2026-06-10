from __future__ import annotations

import os
from typing import Any, Dict

from web.server.config import (
    TRAINING_ROOT,
    _fixed_team_env,
    _payload_enabled,
    _reset_ctde_training_inputs,
)
from web.server.progress import DEFAULT_BENCHMARK_BATTLES, OPPONENTS, _bounded_int


DEFAULT_BENCHMARK_PARALLELISM = 32
MAX_BENCHMARK_PARALLELISM = 32


def _requested_train_opponent(payload: Dict[str, Any]) -> str:
    train_opponent = str(payload.get("opponent") or "both").lower()
    return train_opponent if train_opponent in {"simple", "abyssal", "both"} else "both"


def _ctde_dataset_path(env: Dict[str, str]) -> str:
    return env.get(
        "DUOMON_CTDE_JOINT_DATASET_PATH",
        os.environ.get(
            "DUOMON_CTDE_JOINT_DATASET_PATH",
            str(TRAINING_ROOT / "ctde_joint_examples.jsonl"),
        ),
    )


def _train_env_from_payload(payload: Dict[str, Any]) -> tuple[Dict[str, str], str, str]:
    env = _fixed_team_env(payload)
    env.setdefault("DUOMON_RUN_NAME", "web_training")
    train_epochs = payload.get("epochs") or os.environ.get("DUOMON_DASHBOARD_TRAIN_EPOCHS")
    train_batch_size = payload.get("batch_size") or os.environ.get("DUOMON_DASHBOARD_TRAIN_BATCH_SIZE")
    train_lr = payload.get("learning_rate")
    train_dropout = payload.get("dropout")
    train_early_stop = payload.get("early_stopping_patience")
    train_mode = str(payload.get("train_mode") or "standard").strip().lower()
    train_opponent = _requested_train_opponent(payload)
    env["DUOMON_DASHBOARD_TRAIN_TARGET"] = train_opponent

    if train_epochs:
        env["DUOMON_CTDE_TRAIN_EPOCHS"] = str(_bounded_int(train_epochs, 48, 1, 500))
    elif train_mode == "smart":
        env["DUOMON_CTDE_TRAIN_EPOCHS"] = "160"
    if train_batch_size:
        env["DUOMON_CTDE_BATCH_SIZE"] = str(_bounded_int(train_batch_size, 512, 16, 8192))
    if train_lr:
        try:
            env["DUOMON_CTDE_LR"] = str(max(1e-5, min(0.1, float(train_lr))))
        except (TypeError, ValueError):
            pass
    is_smart = train_mode == "smart"
    if train_dropout:
        try:
            env["DUOMON_CTDE_DROPOUT"] = str(max(0.0, min(0.80, float(train_dropout))))
        except (TypeError, ValueError):
            pass
    elif is_smart:
        env["DUOMON_CTDE_DROPOUT"] = "0.10"
    if train_early_stop:
        patience = _bounded_int(train_early_stop, 0, 0, 100)
        env["DUOMON_CTDE_EARLY_STOPPING_PATIENCE"] = str(patience)
        if patience > 0:
            env.setdefault("DUOMON_CTDE_EARLY_STOPPING_MIN_DELTA", "0.0001")
            env.setdefault("DUOMON_CTDE_LR_SCHEDULER_PATIENCE", str(max(1, patience // 3)))
            env.setdefault("DUOMON_CTDE_LR_SCHEDULER_FACTOR", "0.5")
    elif is_smart:
        env.update(
            {
                "DUOMON_CTDE_EARLY_STOPPING_PATIENCE": "16",
                "DUOMON_CTDE_EARLY_STOPPING_MIN_DELTA": "0.0001",
                "DUOMON_CTDE_LR_SCHEDULER_PATIENCE": "6",
                "DUOMON_CTDE_LR_SCHEDULER_FACTOR": "0.5",
            }
        )
    return env, train_opponent, _ctde_dataset_path(env)


def _benchmark_opponents_from_payload(payload: Dict[str, Any]) -> list[str]:
    requested_opponents = payload.get("opponents")
    if isinstance(requested_opponents, str):
        requested_opponents = requested_opponents.split(",")
    if not isinstance(requested_opponents, list):
        requested_opponents = list(OPPONENTS)
    return [
        str(item).strip().lower()
        for item in requested_opponents
        if str(item).strip().lower() in OPPONENTS
    ] or list(OPPONENTS)


def _benchmark_profile_from_payload(payload: Dict[str, Any]) -> str:
    requested_profile = str(payload.get("profile") or "ctde_mlp").strip().lower()
    return (
        requested_profile if requested_profile in {"baseline_heuristic", "ctde_mlp"} else "ctde_mlp"
    )


def _benchmark_env_from_payload(payload: Dict[str, Any]) -> Dict[str, str]:
    opponents = _benchmark_opponents_from_payload(payload)
    fixed_env = _fixed_team_env(payload)
    env = {
        "DUOMON_PROFILE": _benchmark_profile_from_payload(payload),
        "DUOMON_COMMUNICATION_ENABLED": (
            "1" if _payload_enabled(payload.get("communication_enabled"), True) else "0"
        ),
        "DUOMON_BENCHMARK_ONLINE_LEARNING": "0",
        "DUOMON_USE_ONLINE_LEARNING": "0",
        "DUOMON_BATTLES_PER_OPPONENT": str(
            _bounded_int(payload.get("battles"), DEFAULT_BENCHMARK_BATTLES, 1, 2000)
        ),
        "DUOMON_PARALLEL_BATTLES": str(
            _bounded_int(
                payload.get("parallelism"),
                DEFAULT_BENCHMARK_PARALLELISM,
                1,
                MAX_BENCHMARK_PARALLELISM,
            )
        ),
        "DUOMON_BENCHMARK_SUITE": ",".join(opponents),
        "DUOMON_BENCHMARK_LOG_CTDE_EXAMPLES": "1",
        "DUOMON_PROFILING_ENABLED": "1",
        "DUOMON_RUN_NAME": "web_benchmark",
    }
    env.update(fixed_env)
    out_dir = fixed_env.get("DUOMON_CTDE_OUT_DIR")
    if out_dir:
        env.setdefault(
            "DUOMON_SIMPLE_CTDE_JOINT_RERANKER_PATH",
            os.path.join(out_dir, "ctde_joint_reranker_mlp_simple.json"),
        )
        env.setdefault(
            "DUOMON_ABYSSAL_CTDE_JOINT_RERANKER_PATH",
            os.path.join(out_dir, "ctde_joint_reranker_mlp_abyssal.json"),
        )
    _reset_ctde_training_inputs(env)
    return env


__all__ = [
    "_benchmark_env_from_payload",
    "_benchmark_opponents_from_payload",
    "_benchmark_profile_from_payload",
    "_ctde_dataset_path",
    "_requested_train_opponent",
    "_train_env_from_payload",
]
