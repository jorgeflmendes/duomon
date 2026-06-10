from __future__ import annotations

from .benchmark_context import *
from .benchmark_phase_run import *


def summarize_phase_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    finished = sum(1 for r in results if r.get("finished"))
    wins = sum(1 for r in results if r.get("p1_won"))
    losses = sum(1 for r in results if r.get("p1_lost"))
    errors = sum(1 for r in results if r.get("error"))
    winrate_all = wins / total if total else 0.0
    winrate_finished = wins / finished if finished else 0.0
    return {
        "total": total,
        "finished": finished,
        "wins": wins,
        "losses": losses,
        "errors": errors,
        "winrate_all": winrate_all,
        "winrate_finished": winrate_finished,
    }


def print_phase_summary(opponent_kind: str, summary: Dict[str, Any]) -> None:
    print(
        f"[summary] opponent={_opponent_name(opponent_kind)} "
        f"wins={summary['wins']} total={summary['total']} "
        f"winrate_total={100.0 * summary['winrate_all']:.1f}% "
        f"winrate_finished={100.0 * summary['winrate_finished']:.1f}% "
        f"finished={summary['finished']} losses={summary['losses']} errors={summary['errors']}"
    )






_CTDE_SUBDIR = "outcome_ctde_100_each"


def _detect_available_ctde_models(config: AgentConfig) -> Dict[str, Dict]:
    subdir = os.path.join(config.training_dir, _CTDE_SUBDIR)
    mlp_simple = os.path.join(subdir, "ctde_joint_reranker_mlp_simple.json")
    mlp_abyssal = os.path.join(subdir, "ctde_joint_reranker_mlp_abyssal.json")
    return {
        "ctde_mlp": {
            "label": "CTDE MLP - per-opponent neural reranker",
            "available": os.path.exists(mlp_simple) and os.path.exists(mlp_abyssal),
            "paths": [mlp_simple, mlp_abyssal],
        },
    }


def _print_ctde_model_status(models: Dict[str, Dict]) -> None:
    for key, info in models.items():
        tick = "OK" if info["available"] else "--"
        status = "available" if info["available"] else "not trained"
        print(f"  [{tick}] {key:<22}  {info['label']}  [{status}]")
        for p in info["paths"]:
            rel = os.path.relpath(p)
            mark = "ok" if os.path.exists(p) else "!!"
            print(f"           {mark}  {rel}")


async def _run_ctde_collection_and_train(
    config: AgentConfig, model_type: str, n_battles: int
) -> bool:
    import shutil

    collect_opponents = ("simpleheuristics", "abyssal")
    print(
        f"\n[ctde-select] Collecting {n_battles} battles × {len(collect_opponents)} opponents "
        f"({n_battles * len(collect_opponents)} total)..."
    )
    reset_file(config.ctde_joint_dataset_path)
    reset_file(config.ctde_outcomes_path)

    for opponent_kind in collect_opponents:
        print(f"\n[ctde-select] Collection phase: vs {_opponent_name(opponent_kind)}")
        await run_multi_collection_phase(
            label=f"collect_vs_{opponent_kind}",
            base_config=config,
            n_battles=n_battles,
            opponent_kind=opponent_kind,
            online_learning=False,
        )

    subdir = os.path.join(config.training_dir, _CTDE_SUBDIR)
    os.makedirs(subdir, exist_ok=True)

    print(f"\n[ctde-select] Training {model_type}...")

    if model_type == "ctde_mlp":

        unified = os.path.join(subdir, "ctde_joint_reranker_mlp_unified.json")
        result = train_ctde_joint_mlp_reranker(
            config.ctde_joint_dataset_path,
            unified,
            results_dir=config.output_dir,
            outcomes_path=config.ctde_outcomes_path,
            epochs=24,
        )
        ok = isinstance(result, dict) and result.get("status") != "error"
        if ok and os.path.exists(unified):
            for suffix in ("simple", "abyssal"):
                dest = os.path.join(subdir, f"ctde_joint_reranker_mlp_{suffix}.json")
                shutil.copy2(unified, dest)
            print("[ctde-select] MLP trained and saved to both opponent paths.")
        else:
            print(f"[ctde-select] MLP training failed: {result}")
        return ok and os.path.exists(unified)

    return False


async def _interactive_ctde_select(config: AgentConfig) -> None:
    namespace = getattr(config, "data_namespace", "") or "(default)"
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  CTDE Model Selection  -  team: {namespace}")
    print(sep)

    models = _detect_available_ctde_models(config)
    _print_ctde_model_status(models)
    print()

    available = [k for k, v in models.items() if v["available"]]

    if available:
        print("Select a model:")
        for i, key in enumerate(available, 1):
            print(f"  [{i}] {key} - {models[key]['label']}")
        print("  [0] baseline_heuristic (no learned model)")
        print()
        try:
            raw = input("Your choice [1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = "1"
        idx = int(raw) if raw.isdigit() else 1
        selected = available[idx - 1] if 1 <= idx <= len(available) else "baseline_heuristic"
        os.environ["DUOMON_PROFILE"] = selected
        print(f"\n[ctde-select] Profile => {selected}")

    else:
        print("No CTDE models trained for this team yet.")
        print()
        train_options = list(models.keys())
        print("Options:")
        for i, key in enumerate(train_options, 1):
            print(f"  [{i}] Collect battles and train {key}")
        print("  [0] Continue without CTDE (baseline_heuristic)")
        print()
        try:
            raw = input("Your choice [0]: ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = "0"
        idx = int(raw) if raw.isdigit() else 0

        if 1 <= idx <= len(train_options):
            model_type = train_options[idx - 1]
            try:
                raw_n = input("Collection battles per opponent [100]: ").strip()
            except (EOFError, KeyboardInterrupt):
                raw_n = "100"
            n_battles = int(raw_n) if raw_n.isdigit() and int(raw_n) > 0 else 100
            success = await _run_ctde_collection_and_train(config, model_type, n_battles)
            if success:
                os.environ["DUOMON_PROFILE"] = model_type
                print(f"\n[ctde-select] Training complete. Profile => {model_type}")
            else:
                print("\n[ctde-select] Training failed - falling back to baseline_heuristic.")
                os.environ["DUOMON_PROFILE"] = "baseline_heuristic"
        else:
            os.environ["DUOMON_PROFILE"] = "baseline_heuristic"
            print("[ctde-select] Using baseline_heuristic.")

    print(f"{sep}\n")


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
