from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    path = ROOT / "configs" / (name if name.endswith(".yaml") else f"{name}.yaml")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = deepcopy(config)
    if overrides:
        config.update({key: value for key, value in overrides.items() if value is not None})
    return config

