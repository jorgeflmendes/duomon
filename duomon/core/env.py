from __future__ import annotations

import os
from typing import Any


TRUTHY_VALUES = frozenset({"1", "true", "yes"})
ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})
DISABLED_VALUES = frozenset({"0", "false", "no"})


def env_text(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def env_flag(name: str, default: str = "0") -> bool:
    return env_text(name, default).strip().lower() in TRUTHY_VALUES


def env_enabled(name: str, default: str = "0") -> bool:
    return env_text(name, default).strip().lower() in ENABLED_VALUES


def env_not_disabled(name: str, default: str = "1") -> bool:
    return env_text(name, default).strip().lower() not in DISABLED_VALUES


def env_int(name: str, default: int | str) -> int:
    return int(env_text(name, str(default)))


def env_float(name: str, default: float | str) -> float:
    return float(env_text(name, str(default)))


def optional_bool(value: Any, default: bool) -> bool:
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return bool(default)
    return text in ENABLED_VALUES


__all__ = [
    "env_enabled",
    "env_flag",
    "env_float",
    "env_int",
    "env_not_disabled",
    "env_text",
    "optional_bool",
]
