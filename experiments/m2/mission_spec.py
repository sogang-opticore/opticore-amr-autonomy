"""Resolve a single condition from the versioned warehouse mission manifest."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from scene import Actor, world_xy

MANIFEST = Path(__file__).with_name("missions.json")
MAP = Path(__file__).resolve().parents[2] / "packages/amr_lite/assets/maps/opticore_warehouse/warehouse_map.pgm"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(manifest=MANIFEST):
    spec = json.loads(manifest.read_text())
    if spec["version"] != 1 or spec["frame"] != "map" or spec["start_yaw_rad"] != 0:
        raise ValueError("M2 requires a v1 map-frame mission and zero spawn yaw")
    if not math.isfinite(spec["timeout_sec"]) or spec["timeout_sec"] <= 0:
        raise ValueError("invalid timeout")
    seen, conditions, start_poses = set(), set(), {}
    for case in spec["missions"]:
        key = case["id"]
        src, dst = (spec["stations"][case[n]] for n in ("from", "to"))
        if not all(len(p) == 2 and all(map(math.isfinite, p)) for p in (src, dst)) or src == dst:
            raise ValueError(f"invalid endpoints: {key}")
        split, family = case["split"], case["family"]
        if split not in ("train", "selection", "test") or family not in ("static","crossing","head_on","blockage"):
            raise ValueError(f"invalid split/family: {key}")
        actor = case["actor"]
        if (actor is None) != (family == "static"):
            raise ValueError(f"invalid actor for {key}")
        if actor is not None:
            Actor.from_dict(actor)
        signature = json.dumps([src, dst, actor], sort_keys=True)
        if key in seen or signature in conditions:
            raise ValueError(f"duplicate mission or scenario: {key}")
        for p in (src, dst):
            if start_poses.setdefault(tuple(p), split) != split:
                raise ValueError(f"station pose reused across splits: {key}")
        seen.add(key)
        conditions.add(signature)
    return spec


def resolve(case_id, manifest=MANIFEST):
    spec = load_cases(manifest)
    case = next((c for c in spec["missions"] if c["id"] == case_id), None)
    if case is None:
        raise ValueError(f"unknown mission {case_id}")
    mapping = spec["stations"]
    return {**case, "start": mapping[case["from"]], "goal": mapping[case["to"]],
            "spawn_world": world_xy(mapping[case["from"]]), "timeout_sec": spec["timeout_sec"],
            "map_sha256": spec["map_sha256"], "manifest_sha256": digest(manifest)}
