#!/usr/bin/env python3
"""Shared configuration helpers.

Project .yaml files are deliberately JSON-compatible YAML so the safety-critical
command path can parse them with the Python standard library.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ACT_TASK = (
    "First pick up the pink cube and place it in the center target area, then pick up "
    "the cyan cube and place it in the center target area."
)
CYAN_THEN_PINK_TASK = (
    "First pick up the cyan cube and place it in the center target area, then pick up "
    "the pink cube and place it in the center target area."
)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Config not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"{config_path} must remain JSON-compatible YAML: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Config root must be a mapping: {config_path}")
    return data


def require(mapping: dict[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise SystemExit(f"Missing required config key: {dotted_key}")
        value = value[part]
    if value is None or value == "":
        raise SystemExit(f"Unresolved required config value: {dotted_key}")
    return value


def bool_text(value: bool) -> str:
    return "true" if value else "false"
