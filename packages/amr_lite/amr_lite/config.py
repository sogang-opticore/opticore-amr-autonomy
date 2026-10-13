from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def deep_merge(base: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Return a recursive copy of *base* updated with *overrides*."""
    merged = deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    path = ROOT / "configs" / (name if name.endswith(".yaml") else f"{name}.yaml")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return deep_merge(config, {key: value for key, value in (overrides or {}).items() if value is not None})


def load_robot_profile(name: str) -> dict[str, Any]:
    """Named robot profile: ``{"name", "description", "overrides"}``."""
    path = ROOT / "configs" / "robot_profiles" / f"{name}.yaml"
    if not path.exists():
        available = sorted(item.stem for item in path.parent.glob("*.yaml"))
        raise FileNotFoundError(f"Unknown robot profile {name!r}; available: {available}")
    with path.open(encoding="utf-8") as handle:
        profile = yaml.safe_load(handle) or {}
    profile.setdefault("name", name)
    profile["overrides"] = profile.get("overrides") or {}
    return profile


def resolve_env_config(robot_profile: str | None = None, overrides: dict[str, Any] | None = None,
                       base: dict[str, Any] | None = None) -> dict[str, Any]:
    """``env.yaml`` (or *base*) + robot profile + experiment overrides.

    The profile name is written to ``robot_profile`` so checkpoints, bundles
    and evaluation outputs record which robot they were produced for.
    """
    config = deepcopy(base) if base is not None else load_config("env")
    if robot_profile:
        profile = load_robot_profile(robot_profile)
        config = deep_merge(config, profile["overrides"])
        config["robot_profile"] = profile["name"]
    return deep_merge(config, overrides)
