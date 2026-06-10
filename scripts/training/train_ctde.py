from __future__ import annotations

import argparse
import os
import time
from types import SimpleNamespace

from ctde_cli_common import default_ctde_paths, ensure_repo_on_path

ensure_repo_on_path()

from duomon.ctde import (
    train_ctde_joint_mlp_reranker,
    evaluate_ctde_joint_reranker,
)
from duomon.core.artifacts import register_checkpoint
from duomon.core.jsonl import count_lines



_PATHS = default_ctde_paths()
DATASET_PATH = _PATHS["dataset_path"]
OUTCOMES_PATH = _PATHS["outcomes_path"]
RESULTS_DIR = _PATHS["results_dir"]
OUT_DIR = _PATHS["out_dir"]
SPLIT_PATH = _PATHS["split_path"]
VALIDATION_FRACTION = float(os.environ.get("DUOMON_CTDE_VALIDATION_FRACTION", "0.20"))
TRAIN_RUN_ID = os.environ.get("DUOMON_RUN_ID", time.strftime("%Y%m%d_%H%M%S"))

MODEL_SIMPLE = os.path.join(OUT_DIR, "ctde_joint_reranker_mlp_simple.json")
MODEL_ABYSSAL = os.path.join(OUT_DIR, "ctde_joint_reranker_mlp_abyssal.json")



TRAIN_KWARGS = dict(
    epochs=int(os.environ.get("DUOMON_CTDE_TRAIN_EPOCHS", "48")),
    learning_rate=float(os.environ.get("DUOMON_CTDE_LR", "0.0012")),
    margin=0.15,
    objective=os.environ.get("DUOMON_CTDE_OBJECTIVE", "value_regression"),
    loss_weight=0.65,
    max_win_alternatives=12,
    max_loss_alternatives=6,
    hidden_sizes=(128, 64, 32),
    batch_size=int(os.environ.get("DUOMON_CTDE_BATCH_SIZE", "512")),
    weight_decay=8.0e-5,
    transform="compact_nonlinear",
    activation="tanh",
    seed=42,
    dropout=float(os.environ.get("DUOMON_CTDE_DROPOUT", "0.0")),
    early_stopping_patience=int(os.environ.get("DUOMON_CTDE_EARLY_STOPPING_PATIENCE", "0")),
    early_stopping_min_delta=float(os.environ.get("DUOMON_CTDE_EARLY_STOPPING_MIN_DELTA", "0.0001")),
    lr_scheduler_patience=int(os.environ.get("DUOMON_CTDE_LR_SCHEDULER_PATIENCE", "0")),
    lr_scheduler_factor=float(os.environ.get("DUOMON_CTDE_LR_SCHEDULER_FACTOR", "0.5")),
)


def _fmt(result: dict) -> str:
    lines = [
        f"[training] pair_examples={result.get('examples', 0):,}",
        f"[training] gradient_steps={result.get('updates', 0):,}",
    ]
    ps = result.get("pair_stats", {})
    if ps:
        lines += [
            f"[training] win_records={ps.get('winning_records', ps.get('successful_records', '?'))}",
            f"[training] loss_records={ps.get('losing_records', '?')}",
        ]
    ev = result.get("eval", {})
    if ev:
        lines += [
            f"[training] chosen_top1={ev.get('chosen_top1_rate', 0.0):.3f}",
            f"[training] outcome_aligned_top1={ev.get('outcome_aligned_top1_rate', 0.0):.3f}",
        ]
    return "\n".join(lines)


def _checkpoint_metrics(result: dict) -> dict:
    eval_stats = result.get("eval", {}) or {}
    metrics = {}
    for key in ("chosen_top1_rate", "outcome_aligned_top1_rate", "train_auc"):
        if key in eval_stats:
            metrics[key] = eval_stats[key]
    if "examples" in result:
        metrics["examples"] = result["examples"]
    return metrics


def _register_training_checkpoint(opponent: str, output_path: str, result: dict) -> None:
    artifact_root = os.environ.get(
        "DUOMON_ARTIFACT_ROOT", os.environ.get("DUOMON_ARTIFACT_DIR", "artifacts")
    )
    run_name = os.environ.get("DUOMON_RUN_NAME", "ctde_training")
    output_dir = os.path.join(
        artifact_root,
        "experiments",
        run_name,
        TRAIN_RUN_ID,
    )
    config = SimpleNamespace(output_dir=output_dir)
    registered = register_checkpoint(
        config,
        output_path,
        checkpoint_name=f"ctde_joint_reranker_{opponent}",
        metrics=_checkpoint_metrics(result),
    )
    if registered.get("registered"):
        print(f"[training] checkpoint_manifest={registered.get('manifest_path')}")


def train(opponent: str) -> None:
    assert opponent in {"simple", "abyssal"}
    benchmark_filter = "vs_simpleheuristics" if opponent == "simple" else "vs_abyssal"
    output_path = MODEL_SIMPLE if opponent == "simple" else MODEL_ABYSSAL

    total = count_lines(DATASET_PATH)
    print(f"\n[training] action=start opponent={opponent} benchmark={benchmark_filter}")
    print(f"[training] dataset={DATASET_PATH} records={total:,}")
    print(f"[training] outcomes={OUTCOMES_PATH}")
    print(f"[training] validation_fraction={VALIDATION_FRACTION:.2f} split={SPLIT_PATH}")
    print(f"[training] output={output_path}")
    print(
        f"[training] architecture={TRAIN_KWARGS['hidden_sizes']} activation={TRAIN_KWARGS['activation']}"
    )
    print(
        f"[training] epochs={TRAIN_KWARGS['epochs']} learning_rate={TRAIN_KWARGS['learning_rate']}"
    )
    if int(TRAIN_KWARGS.get("early_stopping_patience", 0) or 0) > 0:
        print(
            "[training] mode=smart_ctde "
            f"early_stopping_patience={TRAIN_KWARGS['early_stopping_patience']} "
            f"min_delta={TRAIN_KWARGS['early_stopping_min_delta']} "
            f"dropout={TRAIN_KWARGS['dropout']} "
            f"lr_scheduler_patience={TRAIN_KWARGS['lr_scheduler_patience']}"
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    result = train_ctde_joint_mlp_reranker(
        dataset_path=DATASET_PATH,
        output_path=output_path,
        results_dir=RESULTS_DIR,
        outcomes_path=OUTCOMES_PATH,
        benchmarks=benchmark_filter,
        validation_fraction=VALIDATION_FRACTION,
        split_path=SPLIT_PATH,
        allow_empty_overwrite=os.environ.get("DUOMON_CTDE_ALLOW_EMPTY_OVERWRITE", "0")
        .strip()
        .lower()
        in {"1", "true", "yes"},
        **TRAIN_KWARGS,
    )
    elapsed = time.time() - t0
    print(f"\n[training] action=finished opponent={opponent} elapsed_seconds={elapsed:.1f}")
    print(_fmt(result))
    if bool(result.get("output_written", False)):
        _register_training_checkpoint(opponent, output_path, result)
    examples = int(result.get("examples", 0) or 0)
    updates = int(result.get("updates", 0) or 0)
    if examples <= 0 or updates <= 0 or not bool(result.get("output_written", False)):
        print(
            "[training] action=failed "
            "reason=no_matching_training_examples "
            f"opponent={opponent} benchmark={benchmark_filter} "
            f"dataset={DATASET_PATH}"
        )
        raise SystemExit(2)


def evaluate(opponent: str) -> None:
    assert opponent in {"simple", "abyssal"}
    model_path = MODEL_SIMPLE if opponent == "simple" else MODEL_ABYSSAL
    print(f"\n[training] action=evaluate opponent={opponent} model={model_path}")
    result = evaluate_ctde_joint_reranker(DATASET_PATH, model_path, RESULTS_DIR, OUTCOMES_PATH)
    print(f"[training] rows_with_outcome={result.get('rows_with_outcome', 0):,}")
    print(f"[training] chosen_top1={result.get('chosen_top1_rate', 0.0):.3f}")
    print(f"[training] outcome_aligned={result.get('outcome_aligned_top1_rate', 0.0):.3f}")
    by_bm = result.get("by_benchmark", {})
    for k, v in by_bm.items():
        print(
            f"[training] benchmark={k} rows={v.get('rows', 0)} "
            f"chosen_top1={v.get('chosen_top1_rate', 0.0):.3f} "
            f"outcome_aligned={v.get('outcome_aligned_top1_rate', 0.0):.3f}"
        )


def main() -> None:
    global\
        DATASET_PATH,\
        OUTCOMES_PATH,\
        RESULTS_DIR,\
        OUT_DIR,\
        MODEL_SIMPLE,\
        MODEL_ABYSSAL,\
        SPLIT_PATH,\
        VALIDATION_FRACTION
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", choices=["simple", "abyssal", "both"], default="both")
    parser.add_argument(
        "--eval-only", action="store_true", help="Evaluate existing models, skip training"
    )
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--outcomes", default=OUTCOMES_PATH)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--split-path", default=SPLIT_PATH)
    parser.add_argument("--validation-fraction", type=float, default=VALIDATION_FRACTION)
    parser.add_argument("--allow-empty-overwrite", action="store_true")
    parser.add_argument("--dropout", type=float, default=TRAIN_KWARGS["dropout"])
    parser.add_argument(
        "--early-stopping-patience", type=int, default=TRAIN_KWARGS["early_stopping_patience"]
    )
    parser.add_argument(
        "--early-stopping-min-delta", type=float, default=TRAIN_KWARGS["early_stopping_min_delta"]
    )
    parser.add_argument(
        "--lr-scheduler-patience", type=int, default=TRAIN_KWARGS["lr_scheduler_patience"]
    )
    parser.add_argument(
        "--lr-scheduler-factor", type=float, default=TRAIN_KWARGS["lr_scheduler_factor"]
    )
    args = parser.parse_args()
    DATASET_PATH = args.dataset
    OUTCOMES_PATH = args.outcomes
    RESULTS_DIR = args.results_dir
    OUT_DIR = args.out_dir
    MODEL_SIMPLE = os.path.join(OUT_DIR, "ctde_joint_reranker_mlp_simple.json")
    MODEL_ABYSSAL = os.path.join(OUT_DIR, "ctde_joint_reranker_mlp_abyssal.json")
    SPLIT_PATH = args.split_path
    VALIDATION_FRACTION = args.validation_fraction
    if args.allow_empty_overwrite:
        os.environ["DUOMON_CTDE_ALLOW_EMPTY_OVERWRITE"] = "1"
    TRAIN_KWARGS["dropout"] = max(0.0, min(0.80, float(args.dropout)))
    TRAIN_KWARGS["early_stopping_patience"] = max(0, int(args.early_stopping_patience))
    TRAIN_KWARGS["early_stopping_min_delta"] = max(0.0, float(args.early_stopping_min_delta))
    TRAIN_KWARGS["lr_scheduler_patience"] = max(0, int(args.lr_scheduler_patience))
    TRAIN_KWARGS["lr_scheduler_factor"] = max(0.05, min(0.95, float(args.lr_scheduler_factor)))

    opponents = ["simple", "abyssal"] if args.opponent == "both" else [args.opponent]

    if args.eval_only:
        for opp in opponents:
            evaluate(opp)
    else:
        for opp in opponents:
            train(opp)
        print("\n[training] action=models_saved next_step=run_benchmark\n")


if __name__ == "__main__":
    main()
