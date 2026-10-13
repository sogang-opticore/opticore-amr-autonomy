import numpy as np
import pytest

from amr_lite.config import resolve_env_config
from amr_lite.planning import RuleBasedPlanner
from amr_lite.rl.evaluation import aggregate, protocol_episodes, run_episodes, selection_score
from amr_lite.rl.vector_env import (
    TRAINING_SEED_BASE,
    SlotSpec,
    SubprocVectorEnv,
    SyncVectorEnv,
    stage_for,
    training_episode_seed,
)

CURRICULUM = [
    {"name": "first", "until": 0.5, "scenarios": {"S0": 1, "S3": 1}},
    {"name": "second", "until": 1.0, "scenarios": {"D0": 1, "P-AISLE": 1}},
]


def _specs(count: int = 3) -> list[SlotSpec]:
    return [SlotSpec(resolve_env_config(), index, 11, curriculum=CURRICULUM) for index in range(count)]


def _rollout(vector, steps: int = 60):
    rng = np.random.default_rng(0)
    outputs = [vector.reset()]
    for step in range(steps):
        if step == steps // 2:
            vector.set_progress(0.75)
        observations, rewards, costs, terminated, truncated, infos = vector.step(
            rng.uniform(-1, 1, (vector.num_envs, 2)).astype(np.float32))
        outputs += [observations, rewards, costs, terminated, truncated,
                    [sorted((k, str(v)) for k, v in info.items() if k != "final_observation") for info in infos]]
    vector.close()
    return outputs


def test_subprocess_and_sync_vector_envs_are_identical():
    sync, subprocess = _rollout(SyncVectorEnv(_specs())), _rollout(SubprocVectorEnv(_specs()))
    assert len(sync) == len(subprocess)
    for left, right in zip(sync, subprocess):
        if isinstance(left, np.ndarray):
            assert np.array_equal(left, right)
        else:
            assert left == right


def test_curriculum_stages_and_training_seed_namespace():
    assert stage_for(CURRICULUM, 0.2)["name"] == "first"
    assert stage_for(CURRICULUM, 0.5)["name"] == "second"
    assert stage_for(CURRICULUM, 2.0)["name"] == "second"
    seeds = {training_episode_seed(7, slot, counter) for slot in range(4) for counter in range(50)}
    assert min(seeds) >= TRAINING_SEED_BASE and len(seeds) == 200
    vector = SyncVectorEnv(_specs(2))
    vector.reset()
    assert all(slot.current[0] in {"S0", "S3"} for slot in vector.slots)
    vector.set_progress(0.9)
    for slot in vector.slots:
        slot.reset()
    assert all(slot.current[0] in {"D0", "P-AISLE"} and slot.current[2] == "second" for slot in vector.slots)


def test_run_episodes_keeps_episode_order_across_workers():
    config = resolve_env_config()
    episodes = [("S0", 1), ("S3", 2), ("D0", 3), ("S1", 4), ("S0", 5)]
    planner = RuleBasedPlanner(config)
    single = run_episodes(planner, episodes, config, workers=1)
    parallel = run_episodes(planner, episodes, config, workers=3, backend="subprocess")
    assert [row["scenario_id"] for row in parallel] == [scenario for scenario, _ in episodes]
    assert [row["seed"] for row in parallel] == [seed for _, seed in episodes]
    for left, right in zip(single, parallel):
        assert left["elapsed_time"] == right["elapsed_time"] and left["success"] == right["success"]
        assert left["episode_return"] == pytest.approx(right["episode_return"])
    metrics = aggregate(parallel)
    assert metrics["episodes"] == 5 and 0.0 <= metrics["success_rate"] <= 1.0
    assert sum(metrics["failure_counts"].values()) == 5
    assert len(selection_score(metrics)) == 5


def test_protocol_episodes_expand_scenarios_and_seeds():
    protocol = {"seed": 20000, "episodes_per_scenario": 2, "scenarios": ["S0", "D1"]}
    assert protocol_episodes(protocol) == [("S0", 20000), ("S0", 20001), ("D1", 20000), ("D1", 20001)]
    assert protocol_episodes(protocol, 1) == [("S0", 20000), ("D1", 20000)]


def test_parallel_evaluator_matches_sequential_and_honours_robot_profile(tmp_path):
    from amr_lite.evaluation import evaluate

    timing = {"inference_ms", "simulation_steps_per_second"}
    config = {"seed": 3, "episodes_per_scenario": 1, "scenarios": ["S3", "D0"], "max_steps": 400,
              "shield_modes": [False, True], "bootstrap_resamples": 20, "robot_profile": "opticore_gazebo",
              "environment_overrides": {"shield_strategy": "scan"}}
    planner = RuleBasedPlanner(resolve_env_config("opticore_gazebo"))
    sequential, _ = evaluate({"rule": planner}, tmp_path / "seq", config)
    parallel, _ = evaluate({"rule": planner}, tmp_path / "par", {**config, "workers": 2})
    strip = lambda rows: [{k: v for k, v in row.items() if k not in timing} for row in rows]
    assert strip(sequential) == strip(parallel)
    assert {row["shield_strategy"] for row in sequential} == {"scan"}
    environment = __import__("json").loads((tmp_path / "seq" / "config.json").read_text())["environment"]
    assert environment["robot_profile"] == "opticore_gazebo" and environment["w_max"] == 1.5
