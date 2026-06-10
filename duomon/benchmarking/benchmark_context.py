from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Sequence, Tuple

from .benchmark_metric_compute import (
    compute_benchmark_metrics,
    compute_metrics_by_opponent,
    load_turn_rows_for_results,
)
from .benchmark_opponents import (
    opponent_description as _opponent_description,
    opponent_name as _opponent_name,
)
from ..config import (
    AgentConfig,
    ensure_parent_dir,
    logger,
    output_path,
    set_global_seed,
    training_path,
)
from ..heuristic import json_safe, safe_getattr
from ..ctde import train_ctde_joint_mlp_reranker
from ..agents.factories import (
    make_multi_abyssal_opponent,
    make_multi_agent,
    make_multi_maxpower_opponent,
    make_multi_random_opponent,
    make_multi_simpleheuristics_opponent,
    make_multi_typeaware_opponent,
    to_id_str,
)
from ..policy_core import MetricsLogger, clone_config, reset_file
from ..profiles import KNOWN_PROFILES, LEAGUE_OPPONENTS, announce_profile_status
from ..profiles import apply_runtime_profile as _profile_apply
from ..shared import Player

__all__ = [name for name in globals() if name != "annotations" and not name.startswith("__")]
