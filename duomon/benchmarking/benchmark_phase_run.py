from __future__ import annotations

from .benchmark_context import *
from .benchmark_battle import append_ctde_outcomes, run_multi_battle
from .benchmark_control import wait_if_benchmark_paused
from .benchmark_file_utils import append_file_if_exists
from .benchmark_opponent_pairs import make_opponent_pair
from .benchmark_phase_comm import *
from .benchmark_setup import _result_label
from ..core.artifacts import cleanup_replay_shards, write_battle_artifacts


def _role_model_path(model_path: str, role: str) -> str:
    root, ext = os.path.splitext(model_path)
    return f"{root}_{role}{ext or '.json'}"


async def _run_one_multi_eval_battle(
    battle_idx: int,
    label: str,
    base_config: AgentConfig,
    opponent_kind: str,
    benchmark_type: str,
    benchmark_model_path: str,
    replay_path: str,
) -> Dict[str, Any]:
    cfg = _eval_phase_config(
        base_config,
        opponent_kind,
        benchmark_type,
        benchmark_model_path,
        replay_path,
    )
    reset_file(replay_path)
    _showdown_replay_dir = output_path(base_config, "showdown_replays")
    os.makedirs(_showdown_replay_dir, exist_ok=True)
    p1 = make_multi_agent(cfg, "ally-p1", save_replays=_showdown_replay_dir)
    p3 = make_multi_agent(cfg, "ally-p3", save_replays=_showdown_replay_dir)
    p2, p4 = make_opponent_pair(cfg, opponent_kind)

    try:
        started = time.perf_counter()
        result = await run_multi_battle(
            p1,
            p2,
            p3,
            p4,
            timeout_seconds=cfg.per_battle_timeout_seconds,
            log_context=f"phase=benchmark opponent={_opponent_name(opponent_kind)} battle_index={battle_idx + 1}",
        )
        result["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        result["label"] = label
        result["benchmark_type"] = benchmark_type
        result["opponent_kind"] = opponent_kind
        result["replay_path"] = cfg.replay_path
        result["battle_idx"] = battle_idx + 1
        result["fixed_ally_team_enabled"] = bool(
            getattr(base_config, "fixed_ally_team_enabled", False)
        )
        result["fixed_ally_team_hash"] = getattr(base_config, "fixed_ally_team_hash", "")
        result["data_namespace"] = getattr(base_config, "data_namespace", "")
        return result
    except Exception as exc:
        return {
            "label": label,
            "benchmark_type": benchmark_type,
            "opponent_kind": opponent_kind,
            "replay_path": replay_path,
            "battle_idx": battle_idx + 1,
            "fixed_ally_team_enabled": bool(getattr(base_config, "fixed_ally_team_enabled", False)),
            "fixed_ally_team_hash": getattr(base_config, "fixed_ally_team_hash", ""),
            "data_namespace": getattr(base_config, "data_namespace", ""),
            "finished": False,
            "error": repr(exc),
        }


async def _run_multi_phase_parallel(
    label: str,
    base_config: AgentConfig,
    n_battles: int,
    opponent_kind: str,
    benchmark_type: str,
    benchmark_model_path: str,
    benchmark_replay_path: str,
    phase_result_path: str,
    parallelism: int,
    on_result: Any = None,
) -> List[Dict[str, Any]]:
    safe_label = re.sub(r"[^a-zA-Z0-9_]+", "_", label)
    replay_dir = output_path(
        base_config,
        os.path.join("parallel_replays", f"{benchmark_type}_{safe_label}"),
    )
    os.makedirs(replay_dir, exist_ok=True)
    semaphore = asyncio.Semaphore(max(1, int(parallelism)))
    results_by_idx: Dict[int, Dict[str, Any]] = {}

    async def runner(idx: int) -> Dict[str, Any]:
        async with semaphore:
            await asyncio.sleep((idx % max(1, int(parallelism))) * 0.20)
            await wait_if_benchmark_paused(f"benchmark:{opponent_kind}:battle_{idx + 1}")
            replay_path = os.path.join(replay_dir, f"battle_{idx + 1:04d}.jsonl")
            return await _run_one_multi_eval_battle(
                idx,
                label,
                base_config,
                opponent_kind,
                benchmark_type,
                benchmark_model_path,
                replay_path,
            )

    tasks = [asyncio.create_task(runner(idx)) for idx in range(int(n_battles))]
    for completed_no, task in enumerate(asyncio.as_completed(tasks), 1):
        result = await task
        idx = int(result.get("battle_idx", completed_no) or completed_no) - 1
        results_by_idx[idx] = result
        if callable(on_result):
            on_result(result)
        print(
            f"[benchmark] progress={completed_no}/{n_battles} opponent={_opponent_name(opponent_kind)} "
            f"battle_index={idx + 1} result={_result_label(result)} battle={result.get('battle_tag')}"
            + (f" error={result.get('error')}" if result.get("error") else "")
        )

    results = [results_by_idx[idx] for idx in sorted(results_by_idx)]
    reset_file(benchmark_replay_path)
    shard_paths: List[str] = []
    for result in results:
        shard_path = str(result.get("replay_path") or "")
        shard_paths.append(shard_path)
        append_file_if_exists(shard_path, benchmark_replay_path)
        result["replay_path"] = benchmark_replay_path
    removed = cleanup_replay_shards(
        shard_paths, keep=bool(getattr(base_config, "keep_parallel_replay_shards", False))
    )
    if removed:
        print(f"[benchmark] replay_shards_removed={removed} aggregate={benchmark_replay_path}")

    reset_file(phase_result_path)
    ensure_parent_dir(phase_result_path)
    with open(phase_result_path, "a", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(json_safe(result)) + "\n")
    artifact_summary = write_battle_artifacts(
        base_config,
        results,
        benchmark_replay_path,
        benchmark_id=f"{benchmark_type}_{safe_label}",
    )
    print(
        f"[benchmark] battle_metadata={artifact_summary['summary_path']} "
        f"selected_traces={artifact_summary['selected_full_traces']}"
    )
    written = append_ctde_outcomes(base_config, results, label, phase_result_path)
    if written:
        print(
            f"[benchmark] outcomes_written={written} path={getattr(base_config, 'ctde_outcomes_path', '')}"
        )
    return results


async def run_multi_phase(
    label: str,
    base_config: AgentConfig,
    n_battles: int,
    opponent_kind: str = "random",
    on_result: Any = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    benchmark_type = f"vs_{opponent_kind}"
    benchmark_model_path = training_path(
        base_config, f"multi_independent_{benchmark_type}_model.json"
    )
    benchmark_replay_path = output_path(
        base_config, f"multi_independent_{benchmark_type}_replays.jsonl"
    )
    phase_result_path = output_path(
        base_config,
        f"multi_independent_{benchmark_type}_{re.sub(r'[^a-zA-Z0-9_]+', '_', label)}_results.jsonl",
    )
    if base_config.benchmark_online_learning:
        reset_file(benchmark_model_path)
        reset_file(_role_model_path(benchmark_model_path, "p1"))
        reset_file(_role_model_path(benchmark_model_path, "p3"))
    reset_file(benchmark_replay_path)
    reset_file(phase_result_path)

    parallelism = (
        1
        if base_config.benchmark_online_learning
        else max(1, int(base_config.max_concurrent_battles or 1))
    )
    if parallelism > 1 and int(n_battles) > 1:
        print(
            f"[phase] mode=parallel opponent={_opponent_name(opponent_kind)} "
            f"battles={n_battles} parallelism={parallelism} online_learning=off"
        )
        return await _run_multi_phase_parallel(
            label,
            base_config,
            n_battles,
            opponent_kind,
            benchmark_type,
            benchmark_model_path,
            benchmark_replay_path,
            phase_result_path,
            min(parallelism, int(n_battles)),
            on_result=on_result,
        )

    for battle_idx in range(int(n_battles)):
        await wait_if_benchmark_paused(f"benchmark:{opponent_kind}:battle_{battle_idx + 1}")
        cfg = _eval_phase_config(
            base_config,
            opponent_kind,
            benchmark_type,
            benchmark_model_path,
            benchmark_replay_path,
        )

        if cfg.use_online_learning:
            p1_cfg = clone_config(cfg, model_path=_role_model_path(benchmark_model_path, "p1"))
            p3_cfg = clone_config(cfg, model_path=_role_model_path(benchmark_model_path, "p3"))
        else:
            p1_cfg = cfg
            p3_cfg = cfg

        _showdown_replay_dir = output_path(base_config, "showdown_replays")
        os.makedirs(_showdown_replay_dir, exist_ok=True)
        p1 = make_multi_agent(p1_cfg, "ally-p1", save_replays=_showdown_replay_dir)
        p3 = make_multi_agent(p3_cfg, "ally-p3", save_replays=_showdown_replay_dir)

        p2, p4 = make_opponent_pair(cfg, opponent_kind)

        try:
            started = time.perf_counter()
            result = await run_multi_battle(
                p1, p2, p3, p4, timeout_seconds=cfg.per_battle_timeout_seconds
            )
            result["elapsed_seconds"] = round(time.perf_counter() - started, 4)
            result["label"] = label
            result["benchmark_type"] = benchmark_type
            result["opponent_kind"] = opponent_kind
            result["replay_path"] = cfg.replay_path
            result["battle_idx"] = battle_idx + 1
            result["fixed_ally_team_enabled"] = bool(
                getattr(base_config, "fixed_ally_team_enabled", False)
            )
            result["fixed_ally_team_hash"] = getattr(base_config, "fixed_ally_team_hash", "")
            result["data_namespace"] = getattr(base_config, "data_namespace", "")
            results.append(result)
            if callable(on_result):
                on_result(result)
            print(
                f"[benchmark] progress={battle_idx + 1}/{n_battles} opponent={_opponent_name(opponent_kind)} "
                f"result={_result_label(result)} battle={result.get('battle_tag')}"
            )
        except Exception as exc:
            result = {
                "label": label,
                "benchmark_type": benchmark_type,
                "opponent_kind": opponent_kind,
                "replay_path": cfg.replay_path,
                "battle_idx": battle_idx + 1,
                "fixed_ally_team_enabled": bool(
                    getattr(base_config, "fixed_ally_team_enabled", False)
                ),
                "fixed_ally_team_hash": getattr(base_config, "fixed_ally_team_hash", ""),
                "data_namespace": getattr(base_config, "data_namespace", ""),
                "finished": False,
                "error": repr(exc),
            }
            results.append(result)
            if callable(on_result):
                on_result(result)
            print(
                f"[benchmark] progress={battle_idx + 1}/{n_battles} opponent={_opponent_name(opponent_kind)} "
                f"result=error error={exc}"
            )
        ensure_parent_dir(phase_result_path)
        with open(phase_result_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_safe(result)) + "\n")
        append_ctde_outcomes(base_config, [result], label, phase_result_path)
        await asyncio.sleep(base_config.inter_battle_sleep_seconds)
    artifact_summary = write_battle_artifacts(
        base_config,
        results,
        benchmark_replay_path,
        benchmark_id=f"{benchmark_type}_{re.sub(r'[^a-zA-Z0-9_]+', '_', label)}",
    )
    print(
        f"[benchmark] battle_metadata={artifact_summary['summary_path']} "
        f"selected_traces={artifact_summary['selected_full_traces']}"
    )
    return results


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
