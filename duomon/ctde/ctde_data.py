from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import ensure_parent_dir
from ..core.jsonl import iter_jsonl as _iter_ctde_records
from ..core.jsonl import tag_is_validation as _tag_is_validation
from .ctde_features import CTDE_FEATURE_NAMES, ctde_runtime_features_from_details


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    n_pos = len(pos)
    n_neg = len(neg)
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    pos_rank_sum = ranks[:n_pos].sum()
    u = pos_rank_sum - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _collect_ctde_value_examples(
    dataset_path: str,
    outcomes: Dict[str, bool],
    max_candidates_per_record: int,
    benchmarks: Optional[set[str]] = None,
    validation_fraction: float = 0.20,
    split_seed: int = 42,
) -> Dict[str, Any]:
    inputs: List[np.ndarray] = []
    labels: List[float] = []
    sample_weights: List[float] = []
    validation_inputs: List[np.ndarray] = []
    validation_labels: List[float] = []
    validation_sample_weights: List[float] = []
    eval_records: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "total_records_scanned": 0,
        "skipped_no_outcome": 0,
        "skipped_benchmark": 0,
        "winning_records": 0,
        "losing_records": 0,
        "training_records": 0,
        "validation_records": 0,
        "validation_examples": 0,
        "validation_fraction": float(validation_fraction),
        "split_seed": int(split_seed),
        "examples": 0,
        "by_benchmark": {},
        "_train_tags": set(),
        "_validation_tags": set(),
    }
    cap = max(1, int(max_candidates_per_record))
    for record in _iter_ctde_records(dataset_path):
        stats["total_records_scanned"] += 1
        benchmark = str(record.get("benchmark_type") or "")
        if benchmarks and benchmark not in benchmarks:
            stats["skipped_benchmark"] += 1
            continue
        tag = str(record.get("battle_tag") or "")
        outcome = outcomes.get(tag)
        if outcome is None:
            stats["skipped_no_outcome"] += 1
            continue
        candidates = record.get("candidates", []) or []
        if not candidates:
            continue
        is_validation = _tag_is_validation(tag, validation_fraction, split_seed)
        ranked = sorted(
            candidates,
            key=lambda c: float(c.get("base_pair_score", c.get("pair_score", 0.0)) or 0.0),
            reverse=True,
        )[:cap]
        bucket = stats["by_benchmark"].setdefault(
            benchmark,
            {"records": 0, "winning_records": 0, "losing_records": 0, "examples": 0},
        )
        bucket["records"] += 1
        if outcome:
            stats["winning_records"] += 1
            bucket["winning_records"] += 1
        else:
            stats["losing_records"] += 1
            bucket["losing_records"] += 1
        if is_validation:
            stats["validation_records"] += 1
            stats["_validation_tags"].add(tag)
            eval_records.append(record)
            label = 1.0 if outcome else 0.0
            for candidate in ranked:
                features = _candidate_features(candidate, benchmark)
                if len(features) != len(CTDE_FEATURE_NAMES):
                    continue
                validation_inputs.append(features)
                validation_labels.append(label)
                validation_sample_weights.append(1.0)
                stats["validation_examples"] += 1
            continue
        stats["training_records"] += 1
        stats["_train_tags"].add(tag)
        label = 1.0 if outcome else 0.0
        for candidate in ranked:
            features = _candidate_features(candidate, benchmark)
            if len(features) != len(CTDE_FEATURE_NAMES):
                continue
            inputs.append(features)
            labels.append(label)
            sample_weights.append(1.0)
            bucket["examples"] += 1
    stats["examples"] = len(inputs)
    return {
        "inputs": inputs,
        "labels": labels,
        "sample_weights": sample_weights,
        "validation_inputs": validation_inputs,
        "validation_labels": validation_labels,
        "validation_sample_weights": validation_sample_weights,
        "eval_records": eval_records,
        "stats": stats,
    }


def _collect_ctde_pairwise_examples(
    dataset_path: str,
    outcomes: Dict[str, bool],
    objective: str,
    loss_weight: float,
    max_win_alternatives: int,
    max_loss_alternatives: int,
    benchmarks: Optional[set[str]] = None,
    validation_fraction: float = 0.20,
    split_seed: int = 42,
) -> Dict[str, Any]:
    positives: List[np.ndarray] = []
    negatives: List[np.ndarray] = []
    sample_weights: List[float] = []
    validation_positives: List[np.ndarray] = []
    validation_negatives: List[np.ndarray] = []
    validation_sample_weights: List[float] = []
    eval_records: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "total_records_scanned": 0,
        "skipped_no_outcome": 0,
        "skipped_benchmark": 0,
        "skipped_no_chosen": 0,
        "winning_records": 0,
        "losing_records": 0,
        "training_records": 0,
        "validation_records": 0,
        "validation_pair_examples": 0,
        "validation_fraction": float(validation_fraction),
        "split_seed": int(split_seed),
        "pair_examples": 0,
        "by_benchmark": {},
        "_train_tags": set(),
        "_validation_tags": set(),
    }

    for record in _iter_ctde_records(dataset_path):
        stats["total_records_scanned"] += 1
        benchmark = str(record.get("benchmark_type") or "")
        if benchmarks and benchmark not in benchmarks:
            stats["skipped_benchmark"] += 1
            continue
        tag = str(record.get("battle_tag") or "")
        outcome = outcomes.get(tag)
        if outcome is None:
            stats["skipped_no_outcome"] += 1
            continue
        candidates = record.get("candidates", [])
        chosen_signature = str(record.get("chosen_pair_signature") or "")
        chosen = next(
            (c for c in candidates if str(c.get("pair_signature") or "") == chosen_signature),
            None,
        )
        if chosen is None or len(candidates) < 2:
            stats["skipped_no_chosen"] += 1
            continue
        chosen_features = _candidate_features(chosen, benchmark)
        alternatives = [
            candidate
            for candidate in candidates
            if str(candidate.get("pair_signature") or "") != chosen_signature
        ]
        ranked_alternatives = []
        for candidate in alternatives:
            features = _candidate_features(candidate, benchmark)
            if len(features) != len(CTDE_FEATURE_NAMES):
                continue
            ranked_alternatives.append((_candidate_base_score(candidate), features))
        if not ranked_alternatives:
            continue
        ranked_alternatives.sort(key=lambda item: item[0], reverse=True)
        bucket = stats["by_benchmark"].setdefault(
            benchmark,
            {
                "records": 0,
                "winning_records": 0,
                "losing_records": 0,
                "pair_examples": 0,
            },
        )
        bucket["records"] += 1
        is_validation = _tag_is_validation(tag, validation_fraction, split_seed)
        if outcome:
            stats["winning_records"] += 1
            bucket["winning_records"] += 1
            if is_validation:
                stats["validation_records"] += 1
                stats["_validation_tags"].add(tag)
                eval_records.append(record)
                for _score, alt_features in ranked_alternatives[
                    : max(1, int(max_win_alternatives))
                ]:
                    validation_positives.append(chosen_features)
                    validation_negatives.append(alt_features)
                    validation_sample_weights.append(1.0)
                    stats["validation_pair_examples"] += 1
                continue
            stats["training_records"] += 1
            stats["_train_tags"].add(tag)
            for _score, alt_features in ranked_alternatives[: max(1, int(max_win_alternatives))]:
                positives.append(chosen_features)
                negatives.append(alt_features)
                sample_weights.append(1.0)
                bucket["pair_examples"] += 1
        elif objective == "outcome_margin":
            stats["losing_records"] += 1
            bucket["losing_records"] += 1
            if is_validation:
                stats["validation_records"] += 1
                stats["_validation_tags"].add(tag)
                eval_records.append(record)
                for _score, alt_features in ranked_alternatives[
                    : max(1, int(max_loss_alternatives))
                ]:
                    validation_positives.append(alt_features)
                    validation_negatives.append(chosen_features)
                    validation_sample_weights.append(float(loss_weight))
                    stats["validation_pair_examples"] += 1
                continue
            stats["training_records"] += 1
            stats["_train_tags"].add(tag)
            for _score, alt_features in ranked_alternatives[: max(1, int(max_loss_alternatives))]:
                positives.append(alt_features)
                negatives.append(chosen_features)
                sample_weights.append(float(loss_weight))
                bucket["pair_examples"] += 1
    stats["pair_examples"] = len(positives)
    return {
        "positives": positives,
        "negatives": negatives,
        "sample_weights": sample_weights,
        "validation_positives": validation_positives,
        "validation_negatives": validation_negatives,
        "validation_sample_weights": validation_sample_weights,
        "eval_records": eval_records,
        "stats": stats,
    }


def load_benchmark_outcomes(
    results_dir: str = "outputs", outcomes_path: str = ""
) -> Dict[str, bool]:
    outcomes: Dict[str, bool] = {}

    def _record_outcome(record: Dict[str, Any]) -> None:
        tag = str(record.get("battle_tag") or "")
        won = record.get("p1_won")
        if not tag or record.get("finished") is not True or not isinstance(won, bool):
            return
        outcomes[tag] = won

    if outcomes_path:
        paths: List[str] = []
        if os.path.isdir(outcomes_path):
            paths = [
                os.path.join(outcomes_path, name)
                for name in os.listdir(outcomes_path)
                if name.endswith(".jsonl")
            ]
        elif os.path.exists(outcomes_path):
            paths = [outcomes_path]
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            record = json.loads(line)
                        except Exception:
                            continue
                        _record_outcome(record)
            except OSError:
                continue
        return outcomes
    if not results_dir or not os.path.isdir(results_dir):
        return outcomes
    for name in os.listdir(results_dir):
        if not name.endswith("_results.jsonl"):
            continue
        path = os.path.join(results_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
                    _record_outcome(record)
        except OSError:
            continue
    return outcomes


def _public_split_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    public = dict(stats)
    public.pop("_train_tags", None)
    public.pop("_validation_tags", None)
    public["by_benchmark"] = {
        key: dict(value) for key, value in dict(stats.get("by_benchmark", {})).items()
    }
    return public


def _write_ctde_split_manifest(
    split_path: str, stats: Dict[str, Any], dataset_path: str, outcomes_path: str
) -> None:
    if not split_path:
        return
    train_tags = sorted(str(tag) for tag in stats.get("_train_tags", set()))
    validation_tags = sorted(str(tag) for tag in stats.get("_validation_tags", set()))
    payload = {
        "dataset_path": dataset_path,
        "outcomes_path": outcomes_path,
        "validation_fraction": stats.get("validation_fraction", 0.0),
        "split_seed": stats.get("split_seed", 42),
        "train_battle_tags": train_tags,
        "validation_battle_tags": validation_tags,
        "train_battles": len(train_tags),
        "validation_battles": len(validation_tags),
    }
    ensure_parent_dir(split_path)
    with open(split_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _candidate_features(candidate: Dict[str, Any], benchmark_type: str = "") -> np.ndarray:
    raw = candidate.get("features", [])
    if isinstance(raw, list) and len(raw) == len(CTDE_FEATURE_NAMES):
        return np.array(raw, dtype=np.float32)
    details = _candidate_details(candidate, benchmark_type)
    return np.array(ctde_runtime_features_from_details(details), dtype=np.float32)


def _candidate_details(candidate: Dict[str, Any], benchmark_type: str = "") -> Dict[str, Any]:
    details = dict(candidate.get("details", {}) or {})
    for key in (
        "pair_score",
        "base_pair_score",
        "local_signature",
        "partner_signature",
        "local_label",
        "partner_label",
        "reason",
        "local_move_id",
        "partner_move_id",
        "local_target_slot",
        "partner_target_slot",
        "local_damage_target",
        "partner_damage_target",
        "local_damage_by_slot",
        "partner_damage_by_slot",
        "local_damage_total",
        "partner_damage_total",
        "target_slot",
        "target_damage",
        "ko_slots",
        "combined_damage_by_slot",
        "trade_value",
        "survival_value",
        "hard_benchmark_value",
        "split_target_value",
        "split_ko_count",
        "split_pressure_count",
    ):
        if key in candidate and key not in details:
            details[key] = candidate.get(key)
    if benchmark_type and "benchmark_type" not in details:
        details["benchmark_type"] = benchmark_type
    if "pair_score" not in details:
        details["pair_score"] = candidate.get("pair_score", 0.0)
    if "base_pair_score" not in details:
        details["base_pair_score"] = candidate.get(
            "base_pair_score", details.get("pair_score", 0.0)
        )
    return details


def _candidate_base_score(candidate: Dict[str, Any]) -> float:
    for key in ("base_pair_score", "pair_score"):
        try:
            return float(candidate.get(key, 0.0) or 0.0)
        except Exception:
            continue
    return 0.0


def _normalize_benchmark_filter(benchmarks: Any) -> Optional[set[str]]:
    if benchmarks is None:
        return None
    if isinstance(benchmarks, str):
        raw_items = [item.strip().lower() for item in benchmarks.split(",")]
    else:
        raw_items = [str(item).strip().lower() for item in benchmarks]
    normalized = set()
    for item in raw_items:
        if not item:
            continue
        normalized.add(item if item.startswith("vs_") else f"vs_{item}")
    return normalized or None


def _rate(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def _add_rates(stats: Dict[str, Any]) -> None:
    rows = int(stats.get("rows_with_outcome", stats.get("rows", 0)) or 0)
    win_rows = int(stats.get("winning_rows", 0) or 0)
    loss_rows = int(stats.get("losing_rows", 0) or 0)
    stats["chosen_top1_rate"] = _rate(int(stats.get("chosen_top1", 0) or 0), rows)
    stats["outcome_aligned_top1_rate"] = _rate(int(stats.get("outcome_aligned_top1", 0) or 0), rows)
    stats["winning_chosen_top1_rate"] = _rate(
        int(stats.get("winning_chosen_top1", 0) or 0), win_rows
    )
    stats["losing_chosen_top1_rate"] = _rate(
        int(stats.get("losing_chosen_top1", 0) or 0), loss_rows
    )


__all__ = [
    "_add_rates",
    "_binary_auc",
    "_candidate_base_score",
    "_candidate_details",
    "_candidate_features",
    "_collect_ctde_pairwise_examples",
    "_collect_ctde_value_examples",
    "_normalize_benchmark_filter",
    "_public_split_stats",
    "_write_ctde_split_manifest",
    "load_benchmark_outcomes",
]
