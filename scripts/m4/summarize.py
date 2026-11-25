#!/usr/bin/env python3
"""Score the fixed development matrix and rank eligible configurations."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
from statistics import mean

from evaluate import PRIMARY, SECONDARY, TRAIN_SEEDS, VARIANTS, conditions


def condition_id(row):
    return row["mission"], row["sensor_seed"], row["profile"]


def matrix_key(report):
    return report["suite"], report["variant"], report["training_seed"], report["shield"]


def expected_reports():
    actors = [("m3", 7, True)]
    actors += list(product(VARIANTS[1:], TRAIN_SEEDS, [True]))
    actors += list(product(["full"], TRAIN_SEEDS, [False]))
    return {(suite, *actor) for suite in ("primary", "secondary") for actor in actors}


def validate(reports):
    by_key = {}
    common = {}
    implementation_by_suite = {}
    for report in reports:
        key = matrix_key(report)
        if key in by_key:
            raise ValueError(f"duplicate report: {key}")
        if key not in expected_reports() or report["schema"] != "opticore-m4-development-v1":
            raise ValueError(f"unexpected report: {key}")
        for field in ("manifest_sha256", "map_sha256", "profile_seed", "profiles"):
            if field in common and common[field] != report[field]:
                raise ValueError(f"mismatched {field} for {key}")
            common[field] = report[field]
        implementation = (report["source_sha256"], report["dependencies_sha256"])
        if implementation_by_suite.setdefault(report["suite"], implementation) != implementation:
            raise ValueError(f"evaluation implementation changed within {report['suite']}")
        expected = set(conditions(report["suite"]))
        actual = [condition_id(row) for row in report["rows"]]
        if set(actual) != expected or len(actual) != len(expected):
            raise ValueError(f"incomplete or duplicate paired episodes: {key}")
        for row in report["rows"]:
            if row["outcome"] not in ("success", "collision", "timeout", "invalid"):
                raise ValueError(f"unexpected outcome in {key}: {row['outcome']}")
        by_key[key] = report
    if set(by_key) != expected_reports():
        raise ValueError(f"missing scored runs: {sorted(expected_reports() - by_key.keys())}")
    for (suite, variant, seed, shield), report in by_key.items():
        reference = by_key["primary", variant, seed, True]
        for field in ("bundle_sha256", "checkpoint_sha256"):
            if report[field] != reference[field]:
                raise ValueError(f"changed {field}: {(suite, variant, seed, shield)}")
    return by_key


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def metrics(rows):
    valid = [row for row in rows if row["outcome"] != "invalid"]
    statuses = Counter(row["outcome"] for row in rows)
    return {
        "n": len(rows), "valid": len(valid),
        "counts": {name: statuses[name] for name in ("success", "collision", "timeout", "invalid")},
        "success_rate": safe_ratio(statuses["success"], len(valid)),
        "contact_rate": safe_ratio(statuses["collision"], len(valid)),
        "timeout_rate": safe_ratio(statuses["timeout"], len(valid)),
        "shield_intervention_per_tick": safe_ratio(sum(row["interventions"] for row in valid),
                                                  sum(row["ticks"] for row in valid)),
        "mean_stalled_s": mean(row["stalled_s"] for row in valid) if valid else None,
        "ended_before_block_release": sum(row["stopped_at_release"] is None for row in valid
                                          if row["mission"] == "train-blockage"),
        "stopped_at_release": sum(row["stopped_at_release"] is True for row in valid),
        "restarted_after_release": sum(row["release_to_restart_s"] is not None for row in valid),
        "route_rejoined": sum(row["route_rejoined"] is True for row in valid),
    }


def paired(left, right):
    """Match each left trial on mission, sensor seed and physical profile."""
    comparator = {condition_id(row): row for row in right}
    if len(comparator) != len(right) or {condition_id(row) for row in left} != set(comparator):
        raise ValueError("unmatched conditions in paired comparison")
    rows = []
    for current in left:
        previous = comparator[condition_id(current)]
        if "invalid" not in (current["outcome"], previous["outcome"]):
            rows.append((current, previous))
    common = [a["elapsed_s"] - b["elapsed_s"] for a, b in rows if a["outcome"] == b["outcome"] == "success"]
    gains = sum(a["outcome"] == "success" and b["outcome"] != "success" for a, b in rows)
    losses = sum(a["outcome"] != "success" and b["outcome"] == "success" for a, b in rows)
    return {
        "paired_attempts": len(left), "valid_pairs": len(rows),
        "success_gains": gains, "success_losses": losses,
        "success_delta_pp": safe_ratio(100 * (gains - losses), len(rows)),
        "contact_delta_pp": safe_ratio(100 * sum((a["outcome"] == "collision") - (b["outcome"] == "collision")
                                                 for a, b in rows), len(rows)),
        "both_success_n": len(common), "mean_elapsed_delta_s_both_success": mean(common) if common else None,
    }


def summarize(reports):
    by_key = validate(reports)
    reference = by_key["primary", "m3", 7, True]["rows"]
    base = metrics(reference)
    static_ref = metrics([r for r in reference if r["mission"] == "train-static"])
    ref_complete = base["valid"] == base["n"]
    configurations = {}
    comparisons = []
    qualifying = []
    for variant, shield in [("m3", True), ("full", True), ("no_history", True),
                            ("no_randomization", True), ("full", False)]:
        name = variant if shield else "full_shield_off"
        seeds = (7,) if variant == "m3" else TRAIN_SEEDS
        primary, secondary, per_seed, dynamic_rates = [], [], {}, []
        for seed in seeds:
            current = by_key["primary", variant, seed, shield]["rows"]
            primary.extend(current)
            secondary.extend(by_key["secondary", variant, seed, shield]["rows"])
            dynamic = [r for r in current if r["mission"] != "train-static"]
            per_seed[seed] = {"overall": metrics(current), "dynamic": metrics(dynamic)}
            dynamic_rates.append(per_seed[seed]["dynamic"]["success_rate"])
            if variant != "m3":
                controls = [("m3", reference)]
                if name != "full":
                    controls.append(("full", by_key["primary", "full", seed, True]["rows"]))
                for label, control in controls:
                    for mission in ("all", *PRIMARY):
                        rows = current if mission == "all" else [r for r in current if r["mission"] == mission]
                        refs = control if mission == "all" else [r for r in control if r["mission"] == mission]
                        comparisons.append({"candidate": name, "seed": seed, "reference": label,
                                            "mission": mission, **paired(rows, refs)})
        total = metrics(primary)
        families = {mission: metrics([r for r in primary if r["mission"] == mission]) for mission in PRIMARY}
        dynamic_success = [families[m]["success_rate"] for m in PRIMARY[1:]]
        dynamic_avg = mean(dynamic_success) if all(x is not None for x in dynamic_success) else None
        common_times = [row["elapsed_s"] - other["elapsed_s"] for row in primary for other in reference
                        if condition_id(row) == condition_id(other) and row["outcome"] == other["outcome"] == "success"]
        summary = {
            "overall": total, "by_mission": families, "by_training_seed": per_seed,
            "dynamic_success_rate": dynamic_avg,
            "worst_seed_dynamic_success": min(dynamic_rates) if all(x is not None for x in dynamic_rates) else None,
            "common_success_time_delta_s_vs_m3": mean(common_times) if common_times else None,
            "secondary": {m: metrics([r for r in secondary if r["mission"] == m]) for m in SECONDARY},
        }
        configurations[name] = summary
        if variant == "m3" or not shield:
            continue
        eligible_measurement = ref_complete and total["valid"] == total["n"]
        if not eligible_measurement:
            summary["gate"] = {"complete_measurement": False, "eligible": False}
            continue
        base_dynamic = configurations["m3"]["dynamic_success_rate"]
        stat = families["train-static"]
        checks = {
            "dynamic_improvement_at_least_5pp": dynamic_avg - base_dynamic >= 0.05 - 1e-12,
            "contact_not_increased": total["contact_rate"] <= base["contact_rate"],
            "static_success_not_reduced": stat["success_rate"] >= static_ref["success_rate"],
            "static_timeout_not_increased": stat["timeout_rate"] <= static_ref["timeout_rate"],
        }
        summary["gate"] = {"complete_measurement": True, **checks, "eligible": all(checks.values()),
                           "dynamic_delta_pp": 100 * (dynamic_avg - base_dynamic)}
        if all(checks.values()):
            qualifying.append(name)
    def rank(name):
        entry = configurations[name]
        time_delta = entry["common_success_time_delta_s_vs_m3"]
        return (-entry["dynamic_success_rate"], -entry["worst_seed_dynamic_success"],
                entry["overall"]["contact_rate"], time_delta if time_delta is not None else float("inf"))
    qualifying.sort(key=rank)
    tied = [name for name in qualifying if rank(name) == rank(qualifying[0])] if qualifying else []
    return {
        "schema": "opticore-m4-scorecard-v1", "protocol": "M4 Lite train split",
        "total_episodes": sum(len(r["rows"]) for r in reports),
        "ranked_eligible": qualifying, "candidate": tied[0] if len(tied) == 1 else None,
        "unresolved_first_place_tie": tied if len(tied) > 1 else [],
        "configurations": configurations, "paired_comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    score = summarize([json.loads(path.read_text()) for path in args.reports])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(score, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
