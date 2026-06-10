from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict


class BenchmarkStopped(RuntimeError):
    pass


def _control_path() -> str:
    return os.environ.get("DUOMON_JOB_CONTROL_FILE", "").strip()


def _read_control_state() -> Dict[str, Any]:
    path = _control_path()
    if not path:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


async def wait_if_benchmark_paused(context: str = "benchmark") -> None:
    announced = False
    while True:
        control = _read_control_state()
        if bool(control.get("stop_requested")):
            print(f"[control] action=stop_requested context={context}", flush=True)
            raise BenchmarkStopped("Benchmark stop requested")
        if not bool(control.get("paused")):
            if announced:
                print(f"[control] action=resumed context={context}", flush=True)
            return
        if not announced:
            print(f"[control] action=paused context={context}", flush=True)
            announced = True
        await asyncio.sleep(0.5)


def write_initial_control(path: str) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"paused": False, "stop_requested": False, "updated_at": time.time()}),
        encoding="utf-8",
    )
