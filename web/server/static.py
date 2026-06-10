from __future__ import annotations

from pathlib import Path
from typing import Optional
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
CLIENT_ROOT = ROOT / "showdown-client"
PLAY_ROOT = CLIENT_ROOT / "play.pokemonshowdown.com"
CONFIG_JS = CLIENT_ROOT / "config" / "config.js"


def _safe_file(base: Path, request_path: str) -> Optional[Path]:
    relative = unquote(request_path).replace("\\", "/").lstrip("/")
    target = (base / relative).resolve()
    try:
        target.relative_to(base.resolve())
    except ValueError:
        return None
    return target if target.is_file() else None


def _rewrite_showdown_index(text: str) -> str:
    text = text.replace("//play.pokemonshowdown.com/", "/showdown/play.pokemonshowdown.com/")
    text = text.replace("//dex.pokemonshowdown.com/", "/showdown/blank/")
    text = text.replace("//replay.pokemonshowdown.com/", "/showdown/play.pokemonshowdown.com/")
    text = text.replace("//pokemonshowdown.com/", "/showdown/blank/")
    text = text.replace("http://smogon.com/forums/", "#")
    return text


def _local_config_js() -> bytes:
    text = CONFIG_JS.read_text(encoding="utf-8")
    start = text.find("Config.defaultserver = {")
    end = text.find("};", start)
    if start >= 0 and end >= 0:
        replacement = (
            "Config.defaultserver = {\n"
            "\tid: 'showdown',\n"
            "\thost: 'localhost',\n"
            "\tport: 8000,\n"
            "\thttpport: 8000,\n"
            "\taltport: 8000,\n"
            "\tregistered: false\n"
            "}"
        )
        text = text[:start] + replacement + text[end + 1 :]
    text = text.replace("'wikipedia.org'", "")
    text = text.replace(
        "client: 'play.pokemonshowdown.com'",
        "client: location.host + '/showdown/play.pokemonshowdown.com'",
    )
    text = text.replace("root: 'pokemonshowdown.com'", "root: location.host + '/showdown/blank'")
    text = text.replace("dex: 'dex.pokemonshowdown.com'", "dex: location.host + '/showdown/blank'")
    text = text.replace(
        "replays: 'replay.pokemonshowdown.com'",
        "replays: location.host + '/showdown/play.pokemonshowdown.com'",
    )
    text = text.replace(
        "users: 'pokemonshowdown.com/users'",
        "users: location.host + '/showdown/blank'",
    )
    text = text.replace(
        "teams: 'teams.pokemonshowdown.com'",
        "teams: location.host + '/showdown/blank'",
    )
    return text.encode("utf-8")


__all__ = ["CONFIG_JS", "PLAY_ROOT", "_local_config_js", "_rewrite_showdown_index", "_safe_file"]
