from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from amr_lite.config import ROOT, deep_merge, load_config
from amr_lite.rl.ppo import train


def load_trainer_config(name_or_path: str) -> dict[str, Any]:
    """A config name under ``configs/`` or a JSON/YAML file (sweep trials)."""
    path = Path(name_or_path)
    if path.suffix in {".json", ".yaml", ".yml"} and path.exists():
        text = path.read_text(encoding="utf-8")
        return json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    return load_config(name_or_path)


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """``a.b.c=value`` pairs (YAML-typed values) as a nested override dict."""
    overrides: dict[str, Any] = {}
    for item in items or []:
        key, separator, raw = item.partition("=")
        if not separator or not key:
            raise ValueError(f"Expected key=value, got {item!r}")
        cursor = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = yaml.safe_load(raw)
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO v2 local planner policy")
    parser.add_argument("--config", default="ppo_v2", help="config name in configs/ or a YAML path")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE",
                        help="override, e.g. --set policy.mode=direct --set lagrangian.enabled=true")
    parser.add_argument("--output", default=str(ROOT / "artifacts/rl/ppo_v2"))
    parser.add_argument("--smoke", action="store_true", help="tiny budget, sync envs; checks the code path only")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-updates", type=int, help="pause after N updates (resume later with --resume)")
    args = parser.parse_args()
    config = deep_merge(load_trainer_config(args.config), parse_overrides(args.overrides))
    print(json.dumps(train(args.output, config, smoke=args.smoke, resume=args.resume,
                           max_updates=args.max_updates), indent=2, default=str))


if __name__ == "__main__":
    main()
