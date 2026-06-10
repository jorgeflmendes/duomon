from __future__ import annotations

from .benchmark_context import *
from .benchmark_battle import *
from .benchmark_file_utils import append_file_if_exists
from .benchmark_opponent_pairs import make_opponent_pair
from .benchmark_phase_comm import _collection_phase_config
from .benchmark_setup import _result_label
from ..core.artifacts import cleanup_replay_shards, write_battle_artifacts


async def run_multi_collection_phase(
    label: str,
    base_config: AgentConfig,
    n_battles: int,
    opponent_kind: str,
    online_learning: bool = False,
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
    reset_file(benchmark_model_path)
    reset_file(benchmark_replay_path)
    reset_file(phase_result_path)

    parallelism = 1 if online_learning else max(1, int(base_config.max_concurrent_battles or 1))
    if parallelism > 1 and int(n_battles) > 1:
        print(
            f"[collection] mode=parallel opponent={_opponent_name(opponent_kind)} "
            f"battles={n_battles} parallelism={min(parallelism, int(n_battles))} online_learning=off"
        )
        replay_dir = output_path(
            base_config,
            os.path.join(
                "parallel_replays",
                f"{benchmark_type}_{re.sub(r'[^a-zA-Z0-9_]+', '_', label)}",
            ),
        )
        os.makedirs(replay_dir, exist_ok=True)
        semaphore = asyncio.Semaphore(min(parallelism, int(n_battles)))
        results_by_idx: Dict[int, Dict[str, Any]] = {}

        async def runner(idx: int) -> Dict[str, Any]:
            async with semaphore:
                await asyncio.sleep((idx % max(1, int(parallelism))) * 0.20)
                replay_path = os.path.join(replay_dir, f"battle_{idx + 1:04d}.jsonl")
                return await _run_one_multi_collection_battle(
                    idx,
                    label,
                    base_config,
                    opponent_kind,
                    benchmark_type,
                    benchmark_model_path,
                    replay_path,
                    online_learning=False,
                )

        tasks = [asyncio.create_task(runner(idx)) for idx in range(int(n_battles))]
        for completed_no, task in enumerate(asyncio.as_completed(tasks), 1):
            result = await task
            idx = int(result.get("battle_idx", completed_no) or completed_no) - 1
            results_by_idx[idx] = result
            print(
                f"[collection] progress={completed_no}/{n_battles} opponent={_opponent_name(opponent_kind)} "
                f"battle_index={idx + 1} result={_result_label(result)} battle={result.get('battle_tag')}"
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
            print(f"[collection] replay_shards_removed={removed} aggregate={benchmark_replay_path}")
        ensure_parent_dir(phase_result_path)
        with open(phase_result_path, "a", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(json_safe(result)) + "\n")
        artifact_summary = write_battle_artifacts(
            base_config,
            results,
            benchmark_replay_path,
            benchmark_id=f"{benchmark_type}_{re.sub(r'[^a-zA-Z0-9_]+', '_', label)}",
        )
        print(
            f"[collection] battle_metadata={artifact_summary['summary_path']} "
            f"selected_traces={artifact_summary['selected_full_traces']}"
        )
        written = append_ctde_outcomes(base_config, results, label, phase_result_path)
        print(
            f"[collection] outcomes_written={written} path={getattr(base_config, 'ctde_outcomes_path', '')}"
        )
        return results

    for battle_idx in range(int(n_battles)):
        result = await _run_one_multi_collection_battle(
            battle_idx,
            label,
            base_config,
            opponent_kind,
            benchmark_type,
            benchmark_model_path,
            benchmark_replay_path,
            online_learning=online_learning,
        )
        results.append(result)
        print(
            f"[collection] progress={battle_idx + 1}/{n_battles} opponent={_opponent_name(opponent_kind)} "
            f"result={_result_label(result)} battle={result.get('battle_tag')}"
        )
        ensure_parent_dir(phase_result_path)
        with open(phase_result_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_safe(results[-1])) + "\n")
        append_ctde_outcomes(base_config, [result], label, phase_result_path)
        await asyncio.sleep(base_config.inter_battle_sleep_seconds)
    artifact_summary = write_battle_artifacts(
        base_config,
        results,
        benchmark_replay_path,
        benchmark_id=f"{benchmark_type}_{re.sub(r'[^a-zA-Z0-9_]+', '_', label)}",
    )
    print(
        f"[collection] battle_metadata={artifact_summary['summary_path']} "
        f"selected_traces={artifact_summary['selected_full_traces']}"
    )
    return results


async def _run_one_multi_collection_battle(
    battle_idx: int,
    label: str,
    base_config: AgentConfig,
    opponent_kind: str,
    benchmark_type: str,
    benchmark_model_path: str,
    replay_path: str,
    online_learning: bool,
) -> Dict[str, Any]:
    cfg = _collection_phase_config(
        base_config,
        opponent_kind,
        benchmark_type,
        benchmark_model_path,
        replay_path,
        online_learning,
    )
    _showdown_replay_dir = output_path(base_config, "showdown_replays")
    os.makedirs(_showdown_replay_dir, exist_ok=True)
    p1 = make_multi_agent(cfg, "ally-p1", save_replays=_showdown_replay_dir)
    p3 = make_multi_agent(cfg, "ally-p3", save_replays=_showdown_replay_dir)
    p2, p4 = make_opponent_pair(cfg, opponent_kind)
    try:
        result = await run_multi_battle(
            p1,
            p2,
            p3,
            p4,
            timeout_seconds=cfg.per_battle_timeout_seconds,
            log_context=f"phase=collection opponent={_opponent_name(opponent_kind)} battle_index={battle_idx + 1}",
        )
        result["label"] = label
        result["benchmark_type"] = benchmark_type
        result["opponent_kind"] = opponent_kind
        result["replay_path"] = cfg.replay_path
        result["battle_idx"] = battle_idx + 1
        return result
    except Exception as exc:
        return {
            "label": label,
            "benchmark_type": benchmark_type,
            "opponent_kind": opponent_kind,
            "battle_idx": battle_idx + 1,
            "replay_path": replay_path,
            "finished": False,
            "error": repr(exc),
        }


__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
