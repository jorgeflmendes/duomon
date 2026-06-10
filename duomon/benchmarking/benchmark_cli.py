from __future__ import annotations

from .benchmark_context import *
from .benchmark_reporting import *


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DuoMon multi-agent Pokemon Showdown benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m duomon --profile baseline_heuristic --opponents simpleheuristics,abyssal --battles 200
  python -m duomon --profile ctde_mlp --opponents simpleheuristics,abyssal --battles 200
  python -m duomon --profile league --league --battles 200

Known profiles: """
        + ", ".join(sorted(KNOWN_PROFILES)),
    )
    parser.add_argument(
        "--profile",
        default="",
        help="Named benchmark profile (baseline_heuristic, ctde_mlp, league)",
    )
    parser.add_argument(
        "--opponents",
        default="",
        help="Comma-separated opponent list: random,maxpower,typeaware,simpleheuristics,abyssal",
    )
    parser.add_argument(
        "--battles",
        type=int,
        default=0,
        help="Number of battles per opponent (overrides DUOMON_BATTLES_PER_OPPONENT)",
    )
    parser.add_argument(
        "--league",
        action="store_true",
        help="Run the full league suite: random,maxpower,typeaware,simpleheuristics,abyssal",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=0,
        help="Max concurrent battles (overrides DUOMON_PARALLEL_BATTLES)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (0 = use existing DUOMON_SEED or 42)",
    )
    parser.add_argument(
        "--profiling",
        action="store_true",
        help="Write lightweight timing, throughput, and storage profile JSON.",
    )
    parser.add_argument(
        "--artifact-root",
        default="",
        help="Root for experiment metadata, selected battle traces, and profiles.",
    )
    parser.add_argument(
        "--replay-retention",
        choices=["all", "selected_only"],
        default="",
        help="Keep all aggregate replay rows or only selected full battle traces.",
    )
    parser.add_argument(
        "--select-model",
        action="store_true",
        help=(
            "Interactively select a CTDE model for the current team. "
            "Shows trained models available for the team namespace; "
            "if none exist, offers to collect N battles and train one."
        ),
    )
    communication_group = parser.add_mutually_exclusive_group()
    communication_group.add_argument(
        "--communication",
        dest="communication_enabled",
        action="store_true",
        default=None,
        help="Enable structured inter-agent communication during execution.",
    )
    communication_group.add_argument(
        "--no-communication",
        dest="communication_enabled",
        action="store_false",
        help="Disable inter-agent communication during execution.",
    )
    args = parser.parse_args()



    if args.select_model:
        os.environ.setdefault("DUOMON_SELECT_MODEL", "1")
    if args.profile:
        os.environ.setdefault("DUOMON_PROFILE", args.profile)
    if args.league:
        os.environ.setdefault(
            "DUOMON_BENCHMARK_SUITE",
            ",".join(LEAGUE_OPPONENTS),
        )
    elif args.opponents:
        os.environ.setdefault("DUOMON_BENCHMARK_SUITE", args.opponents)
    if args.battles > 0:
        os.environ.setdefault("DUOMON_BATTLES_PER_OPPONENT", str(args.battles))
    if args.parallelism > 0:
        os.environ.setdefault("DUOMON_PARALLEL_BATTLES", str(args.parallelism))
    if args.seed > 0:
        os.environ.setdefault("DUOMON_SEED", str(args.seed))
    if args.profiling:
        os.environ.setdefault("DUOMON_PROFILING_ENABLED", "1")
    if args.artifact_root:
        os.environ.setdefault("DUOMON_ARTIFACT_ROOT", args.artifact_root)
    if args.replay_retention:
        os.environ.setdefault("DUOMON_REPLAY_RETENTION", args.replay_retention)
    if args.communication_enabled is not None:
        os.environ["DUOMON_COMMUNICATION_ENABLED"] = "1" if args.communication_enabled else "0"
    return args


def _ctde_hidden_sizes_from_env() -> Tuple[int, ...]:
    sizes = tuple(
        int(item.strip())
        for item in os.environ.get("DUOMON_CTDE_HIDDEN", "96,48").split(",")
        if item.strip()
    )
    return sizes or (96, 48)


def _train_ctde_mlp_from_env(config: AgentConfig) -> Dict[str, Any]:
    return train_ctde_joint_mlp_reranker(
        config.ctde_joint_dataset_path,
        config.ctde_joint_reranker_path,
        results_dir=config.output_dir,
        outcomes_path=config.ctde_outcomes_path,
        epochs=int(os.environ.get("DUOMON_CTDE_EPOCHS", "24")),
        learning_rate=float(os.environ.get("DUOMON_CTDE_LR", "0.0015")),
        margin=float(os.environ.get("DUOMON_CTDE_MARGIN", "0.15")),
        objective=os.environ.get("DUOMON_CTDE_OBJECTIVE", "outcome_margin"),
        loss_weight=float(os.environ.get("DUOMON_CTDE_LOSS_WEIGHT", "0.70")),
        max_win_alternatives=int(os.environ.get("DUOMON_CTDE_MAX_WIN_ALTERNATIVES", "8")),
        max_loss_alternatives=int(os.environ.get("DUOMON_CTDE_MAX_LOSS_ALTERNATIVES", "5")),
        hidden_sizes=_ctde_hidden_sizes_from_env(),
        batch_size=int(os.environ.get("DUOMON_CTDE_BATCH_SIZE", "512")),
        weight_decay=float(os.environ.get("DUOMON_CTDE_WEIGHT_DECAY", "0.0001")),
        transform=os.environ.get("DUOMON_CTDE_TRANSFORM", "compact_nonlinear"),
        activation=os.environ.get("DUOMON_CTDE_ACTIVATION", "tanh"),
        benchmarks=os.environ.get("DUOMON_CTDE_BENCHMARKS", ""),
        max_pairs=int(os.environ.get("DUOMON_CTDE_MAX_PAIRS", "0")),
        device=os.environ.get("DUOMON_CTDE_DEVICE", "auto"),
        validation_fraction=config.ctde_validation_fraction,
        split_path=config.ctde_split_path,
    )


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
