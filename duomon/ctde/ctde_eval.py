from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

import numpy as np

from ..core.jsonl import iter_jsonl as _iter_ctde_records
from .ctde_data import _add_rates, _candidate_features, load_benchmark_outcomes
from .ctde_features import CTDE_FEATURE_NAMES
from .ctde_models import CTDEJointReranker


def evaluate_ctde_joint_reranker(
    dataset_path: str,
    model_path: str,
    results_dir: str = "outputs",
    outcomes_path: str = "",
) -> Dict[str, Any]:
    outcomes = load_benchmark_outcomes(results_dir, outcomes_path)
    model = CTDEJointReranker(model_path)
    return evaluate_ctde_joint_reranker_records(_iter_ctde_records(dataset_path), outcomes, model)


def evaluate_ctde_joint_reranker_records(
    records: Iterable[Dict[str, Any]],
    outcomes: Dict[str, bool],
    scorer: Any,
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "rows": 0,
        "rows_with_outcome": 0,
        "chosen_top1": 0,
        "winning_rows": 0,
        "winning_chosen_top1": 0,
        "losing_rows": 0,
        "losing_chosen_top1": 0,
        "by_benchmark": {},
    }
    for record in records:
        stats["rows"] += 1
        tag = str(record.get("battle_tag") or "")
        outcome = outcomes.get(tag)
        if outcome is None:
            continue
        candidates = record.get("candidates", [])
        chosen_signature = str(record.get("chosen_pair_signature") or "")
        if not candidates or not chosen_signature:
            continue
        benchmark = str(record.get("benchmark_type") or "unknown")
        scored = []
        for candidate in candidates:
            features = _candidate_features(candidate, benchmark)
            if len(features) != len(CTDE_FEATURE_NAMES):
                continue
            scored.append((_score_ctde_features(scorer, features), candidate))
        if not scored:
            continue
        scored.sort(key=lambda item: item[0], reverse=True)
        top_signature = str(scored[0][1].get("pair_signature") or "")
        chosen_top1 = top_signature == chosen_signature
        outcome_aligned = bool(chosen_top1) if outcome else not bool(chosen_top1)
        bucket = stats["by_benchmark"].setdefault(
            benchmark,
            {
                "rows": 0,
                "chosen_top1": 0,
                "outcome_aligned_top1": 0,
                "winning_rows": 0,
                "winning_chosen_top1": 0,
                "losing_rows": 0,
                "losing_chosen_top1": 0,
            },
        )
        for target in (stats, bucket):
            target["rows_with_outcome" if target is stats else "rows"] = (
                target.get("rows_with_outcome" if target is stats else "rows", 0) + 1
            )
            target["chosen_top1"] = target.get("chosen_top1", 0) + int(chosen_top1)
            target["outcome_aligned_top1"] = target.get("outcome_aligned_top1", 0) + int(
                outcome_aligned
            )
            if outcome:
                target["winning_rows"] = target.get("winning_rows", 0) + 1
                target["winning_chosen_top1"] = target.get("winning_chosen_top1", 0) + int(
                    chosen_top1
                )
            else:
                target["losing_rows"] = target.get("losing_rows", 0) + 1
                target["losing_chosen_top1"] = target.get("losing_chosen_top1", 0) + int(
                    chosen_top1
                )

    _add_rates(stats)
    for bucket in stats["by_benchmark"].values():
        _add_rates(bucket)
    return stats


def _score_ctde_features(scorer: Any, features: Sequence[float]) -> float:
    predict = getattr(scorer, "predict", None)
    if callable(predict):
        return float(predict(features))
    weights = np.array(scorer, dtype=np.float32)
    if len(features) != len(weights):
        return 0.0
    return float(np.dot(weights, np.array(features, dtype=np.float32)))


__all__ = [
    "_score_ctde_features",
    "evaluate_ctde_joint_reranker",
    "evaluate_ctde_joint_reranker_records",
]
