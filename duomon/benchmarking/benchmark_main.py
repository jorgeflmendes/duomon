from __future__ import annotations

from .benchmark_context import *
from .benchmark_cli import *
from .benchmark_collection import run_multi_collection_phase
from .benchmark_config_builder import *
from .benchmark_control import BenchmarkStopped, wait_if_benchmark_paused
from .benchmark_metric_realtime import BenchmarkMetricsSnapshot
from .benchmark_phase_env import _apply_post_profile_env_overrides
from .benchmark_phase_run import run_multi_phase
from .benchmark_reporting import print_phase_summary, summarize_phase_results
from .benchmark_setup import (
    _multi_battle_format,
    apply_data_hygiene_namespace,
    ensure_showdown_server,
)
from ..core.artifacts import write_run_metadata
from ..core.profiling import RunProfiler


async def main() -> None:
    _parse_cli_args()
    config = build_benchmark_config_from_env()
    data_namespace = apply_data_hygiene_namespace(config)
    profiler = RunProfiler.from_config(config, "benchmark")
    if os.environ.get("DUOMON_SELECT_MODEL", "").strip() == "1":
        await _interactive_ctde_select(config)
    os.environ.setdefault("DUOMON_PROFILE", "ctde_mlp")
    active_profile = _profile_apply(config, announce=False)
    _apply_post_profile_env_overrides(config)
    announce_profile_status(active_profile, config)
    config.battle_format = _multi_battle_format(config)
    if os.environ.get("DUOMON_TRAIN_CTDE", "").strip() == "1":
        result = _train_ctde_mlp_from_env(config)
        print(f"CTDE joint reranker trained: {result}")
        return

    ensure_showdown_server()




    collect_n = int(os.environ.get("DUOMON_COLLECT_BATTLES", "0") or "0")
    if collect_n > 0:
        set_global_seed(config.seed)
        reset_file(config.ctde_joint_dataset_path)
        reset_file(config.ctde_outcomes_path)
        collect_env = os.environ.get("DUOMON_COLLECT_OPPONENTS", "simpleheuristics,abyssal,maxpower")
        allowed_collect = {"maxpower", "simpleheuristics", "abyssal", "typeaware"}
        collect_opponents = tuple(
            item.strip().lower()
            for item in collect_env.split(",")
            if item.strip().lower() in allowed_collect
        ) or ("simpleheuristics", "abyssal", "maxpower")
        print(
            f"\n[collection] action=start battles_per_opponent={collect_n} opponents={','.join(collect_opponents)}"
        )
        collect_online_learning = os.environ.get(
            "DUOMON_COLLECT_ONLINE_LEARNING", "0"
        ).strip().lower() in {"1", "true", "yes"}
        for opponent_kind in collect_opponents:
            print("\n" + "-" * 72)
            print(
                f"[collection] action=start_phase opponent={_opponent_name(opponent_kind)} "
                f'battles={collect_n} description="{_opponent_description(opponent_kind)}"'
            )
            print("-" * 72)
            await run_multi_collection_phase(
                label=f"collect_vs_{opponent_kind}",
                base_config=config,
                n_battles=collect_n,
                opponent_kind=opponent_kind,
                online_learning=collect_online_learning,
            )
        ctde_stats = _train_ctde_mlp_from_env(config)
        print(f"\n[collection] action=train_ctde_joint_reranker result={ctde_stats}")
        return

    set_global_seed(config.seed)
    print('[benchmark] action=start name="Independent Multi Battle Agent" teams="p1+p3 vs p2+p4"')
    print(f"[benchmark] battle_format={config.battle_format}")
    print("[benchmark] allies=p1,p3 opponents=p2,p4 control=one_action_per_slot")
    if data_namespace:
        print(
            f"[benchmark] data_namespace={data_namespace} output_dir={config.output_dir} "
            f"training_dir={config.training_dir}"
        )
    if config.fixed_ally_team_enabled:
        print(
            "[benchmark] fixed_ally_team=on "
            f"p1_team={config.fixed_ally_team_p1_path or 'inline'} "
            f"p3_team={config.fixed_ally_team_p3_path or 'inline'} "
            f"team_hash={getattr(config, 'fixed_ally_team_hash', '')}"
        )
        if getattr(config, "mirror_opponent_team_enabled", False):
            print("[benchmark] opponent_team_mode=mirror_allies p2_team=p1 p4_team=p3")
    if active_profile not in {"", "control", "default", "none", "off"}:
        print(f"[benchmark] active_profile={active_profile}")
    print(
        "[benchmark] communication="
        + ("on" if bool(getattr(config, "communication_enabled", True)) else "off")
        + f" mode={getattr(config, 'communication_ablation_mode', 'normal')}"
    )
    if not config.benchmark_online_learning:
        print(f"[benchmark] online_learning=off parallelism={config.max_concurrent_battles}")
    _VALID_OPPONENTS = {
        "random",
        "maxpower",
        "typeaware",
        "simpleheuristics",
        "abyssal",
    }
    suite_env = os.environ.get("DUOMON_BENCHMARK_SUITE", "").strip()
    if suite_env:
        benchmark_suite = tuple(
            item.strip().lower()
            for item in suite_env.split(",")
            if item.strip().lower() in _VALID_OPPONENTS
        )
        if not benchmark_suite:
            benchmark_suite = ("random", "maxpower", "simpleheuristics", "abyssal")
    else:
        benchmark_suite = ("random", "maxpower", "simpleheuristics", "abyssal")
    metadata_path = write_run_metadata(
        config,
        "benchmark",
        benchmark_suite=benchmark_suite,
        extra={
            "eval_battles_per_opponent": config.eval_battles,
            "parallelism": config.max_concurrent_battles,
            "communication_enabled": bool(getattr(config, "communication_enabled", True)),
        },
    )

    print(
        "[benchmark] suite="
        + ", ".join(f"{config.eval_battles} vs {_opponent_name(name)}" for name in benchmark_suite)
    )
    print(f"[benchmark] metadata_path={metadata_path}")

    reset_file(config.metrics_path)
    metrics_logger = MetricsLogger(config.metrics_path)
    metrics_snapshot = BenchmarkMetricsSnapshot(
        output_path(config, "benchmark_metrics_summary.json"),
        root_dir=os.getcwd(),
    )
    all_results: List[Dict[str, Any]] = []

    phase_summaries: Dict[str, Dict[str, Any]] = {}

    def publish_result(record: Dict[str, Any]) -> None:
        metrics_logger.log_result(record)
        metrics_snapshot.add_result(record)

    try:
        for opponent_kind in benchmark_suite:
            await wait_if_benchmark_paused(f"benchmark:{opponent_kind}:phase")
            print("\n" + "-" * 72)
            print(
                f"[phase] action=start opponent={_opponent_name(opponent_kind)} "
                f'battles={config.eval_battles} description="{_opponent_description(opponent_kind)}"'
            )
            print("-" * 72)
            phase_started = time.perf_counter()
            results = await run_multi_phase(
                label=f"multi_eval_vs_{opponent_kind}",
                base_config=config,
                n_battles=config.eval_battles,
                opponent_kind=opponent_kind,
                on_result=publish_result,
            )
            phase_elapsed = time.perf_counter() - phase_started
            profiler.record_phase(
                f"benchmark_vs_{opponent_kind}",
                phase_elapsed,
                battles=len(results),
                requested_battles=int(config.eval_battles),
                battles_per_second=(len(results) / phase_elapsed if phase_elapsed > 0 else 0.0),
                opponent_kind=opponent_kind,
            )
            all_results.extend(results)

            summary = summarize_phase_results(results)
            phase_summaries[opponent_kind] = summary
            print_phase_summary(opponent_kind, summary)
            metrics_logger.log_result(
                {
                    "label": f"summary_vs_{opponent_kind}",
                    "opponent_kind": opponent_kind,
                    "summary": summary,
                }
            )
    except BenchmarkStopped:
        print("[benchmark] action=stopped", flush=True)
        return

    print("\n" + "=" * 72)
    print('[summary] title="Multi benchmark team win rate for p1+p3"')
    print("=" * 72)
    for opponent_kind in benchmark_suite:
        print_phase_summary(opponent_kind, phase_summaries[opponent_kind])
    overall = summarize_phase_results(all_results)
    print("-" * 72)
    print_phase_summary("overall", overall)
    turn_rows = metrics_snapshot.turn_rows or load_turn_rows_for_results(all_results)
    metrics_summary = compute_benchmark_metrics(all_results, turn_rows)
    metrics_by_opponent = compute_metrics_by_opponent(all_results, turn_rows)
    metrics_path = output_path(config, "benchmark_metrics_summary.json")
    ensure_parent_dir(metrics_path)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(
            json_safe(
                {
                    "overall": metrics_summary,
                    "opponents": metrics_by_opponent,
                    "results": all_results,
                    "result_count": len(all_results),
                    "turn_row_count": len(turn_rows),
                    "updated_at": time.time(),
                    "incremental": True,
                }
            ),
            handle,
            indent=2,
        )
    print('[summary] title="Coordination and efficiency metrics"')
    for key, metric in metrics_summary.get("metrics", {}).items():
        value = metric.get("value")
        value_text = (
            "n/a"
            if value is None or not metric.get("available")
            else f"{float(value):.2f}{metric.get('unit', '')}"
        )
        print(f'[metric] name={key} value={value_text} detail="{metric.get("detail", "")}"')
    print("=" * 72)
    print(f"[summary] metrics_path={config.metrics_path}")
    print(f"[summary] benchmark_metrics_path={metrics_path}")
    profiler.record_storage("output_dir", config.output_dir)
    profiler.record_storage("training_dir", config.training_dir)
    profile_path = profiler.write("benchmark_profile.json")
    if profile_path:
        print(f"[summary] profile_path={profile_path}")


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
