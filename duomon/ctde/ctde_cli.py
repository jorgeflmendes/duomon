from __future__ import annotations

import json
import os

from .ctde_eval import evaluate_ctde_joint_reranker
from .ctde_train_pairwise import train_ctde_joint_mlp_reranker


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train/evaluate CTDE joint-action reranker.")
    parser.add_argument(
        "--dataset",
        default=os.path.join("training_artifacts", "ctde_joint_examples.jsonl"),
    )
    parser.add_argument(
        "--model",
        default=os.path.join("training_artifacts", "ctde_joint_reranker.json"),
    )
    parser.add_argument("--results-dir", default="outputs")
    parser.add_argument("--outcomes-path", default="")
    parser.add_argument("--split-path", default="")
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--allow-empty-overwrite", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument(
        "--objective",
        choices=["outcome_margin", "value_regression"],
        default="outcome_margin",
    )
    parser.add_argument("--loss-weight", type=float, default=0.55)
    parser.add_argument("--max-loss-alternatives", type=int, default=3)
    parser.add_argument("--max-win-alternatives", type=int, default=8)
    parser.add_argument("--hidden", default="96,48")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--transform", default="compact_nonlinear")
    parser.add_argument("--activation", choices=["tanh", "relu"], default="tanh")
    parser.add_argument("--benchmarks", default="")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--device", default=os.environ.get("DUOMON_CTDE_DEVICE", "auto"))
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if args.eval_only:
        result = evaluate_ctde_joint_reranker(
            args.dataset, args.model, args.results_dir, args.outcomes_path
        )
    else:
        hidden_sizes = tuple(
            int(item.strip()) for item in str(args.hidden or "").split(",") if item.strip()
        )
        result = train_ctde_joint_mlp_reranker(
            args.dataset,
            args.model,
            results_dir=args.results_dir,
            outcomes_path=args.outcomes_path,
            epochs=args.epochs,
            learning_rate=args.lr,
            margin=args.margin,
            objective=args.objective,
            loss_weight=args.loss_weight,
            max_win_alternatives=args.max_win_alternatives,
            max_loss_alternatives=args.max_loss_alternatives,
            hidden_sizes=hidden_sizes or (96, 48),
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            transform=args.transform,
            activation=args.activation,
            benchmarks=args.benchmarks,
            max_pairs=args.max_pairs,
            device=args.device,
            validation_fraction=args.validation_fraction,
            split_path=args.split_path,
            allow_empty_overwrite=args.allow_empty_overwrite,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _main()


__all__ = ["_main"]
