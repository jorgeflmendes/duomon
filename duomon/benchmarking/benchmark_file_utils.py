from __future__ import annotations

import os

from ..config import ensure_parent_dir


def append_file_if_exists(source_path: str, dest_path: str) -> None:
    if not source_path or not os.path.exists(source_path):
        return
    ensure_parent_dir(dest_path)
    with (
        open(source_path, "r", encoding="utf-8") as src,
        open(dest_path, "a", encoding="utf-8") as dst,
    ):
        for line in src:
            dst.write(line)
