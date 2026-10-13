"""PPO v2: vectorized PPO with residual policies, a cost constraint and a curriculum.

Differences from the frozen M1 trainer (``amr_lite.learning.train_ppo``):

- lockstep vector environments sampling a staged scenario curriculum;
- ``direct`` or ``residual`` policies (``a = clip(rule + s * pi, -1, 1)``);
- time-limit truncation bootstraps ``gamma * V(final observation)``;
- optional cost critic with a Lagrange multiplier on the per-step safety
  cost (PPO-Lagrangian);
- learning-rate and entropy schedules, approximate-KL early stopping;
- periodic selection-split evaluation with the pre-registered score,
  best/final checkpoints and resumable state.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset

from amr_lite.config import load_config, resolve_env_config
from amr_lite.learning.dataset import load_dataset
from amr_lite.planning.rule_baseline import RuleBasedPlanner

from .evaluation import evaluate_protocol, selection_score
from .networks import PolicyNetwork
from .policy import PPOPolicy, combine_residual
from .vector_env import SlotSpec, make_vector_env


class RunningMeanStd:
    def __init__(self, mean: float = 0.0, var: float = 1.0, count: float = 1e-4):
        self.mean, self.var, self.count = mean, var, count

    @property
    def std(self) -> float:
        return math.sqrt(self.var + 1e-8)

    def update(self, values: np.ndarray) -> None:
        batch_mean, batch_var, batch_count = float(np.mean(values)), float(np.var(values)), values.size
        delta, total = batch_mean - self.mean, self.count + batch_count
        self.mean += delta * batch_count / total
        self.var = (self.var * self.count + batch_var * batch_count
                    + delta * delta * self.count * batch_count / total) / total
        self.count = total

    def as_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "var": self.var, "count": self.count}


def _schedule(start: float, end: float, progress: float) -> float:
    return start + (end - start) * min(max(progress, 0.0), 1.0)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def warm_start_actor(network: PolicyNetwork, datasets: str | list[str], settings: dict[str, Any],
                     seed: int) -> float:
    """Behaviour-clone the actor mean onto teacher actions (direct mode)."""
    data = load_dataset(datasets if isinstance(datasets, list) else [datasets])
    loader = DataLoader(TensorDataset(torch.from_numpy(data.observations), torch.from_numpy(data.actions)),
                        batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(network.actor_parameters(), lr=float(settings.get("learning_rate", 1e-3)))
    loss_value = math.nan
    for _ in range(int(settings.get("epochs", 15))):
        for observations, actions in loader:
            loss = torch.mean((torch.tanh(network.mean(observations)) - actions) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = float(loss.item())
    network.log_std.data.fill_(float(settings.get("log_std", -1.5)))
    return loss_value


def _gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_values: np.ndarray,
         gamma: float, lam: float) -> np.ndarray:
    advantages = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1])
    for step in reversed(range(rewards.shape[0])):
        following = last_values if step == rewards.shape[0] - 1 else values[step + 1]
        alive = 1.0 - dones[step]
        delta = rewards[step] + gamma * following * alive - values[step]
        running = delta + gamma * lam * alive * running
        advantages[step] = running
    return advantages


def _denormalize(values: np.ndarray, stats: RunningMeanStd, enabled: bool) -> np.ndarray:
    """Critic outputs are normalized returns when value normalization is on."""
    return values * stats.std + stats.mean if enabled else values


def _standardize(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / (values.std() + 1e-8)


def checkpoint_payload(network: PolicyNetwork, cfg: dict[str, Any], env_config: dict[str, Any],
                       timesteps: int, evaluation: dict[str, Any] | None = None,
                       extra: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = cfg["policy"]
    return {
        "kind": PolicyNetwork.checkpoint_kind,
        "architecture": network.architecture,
        "model_state": network.state_dict(),
        "policy": {"mode": policy["mode"], "residual_scale": float(policy.get("residual_scale", 0.5))},
        "env_config": env_config,
        "baseline_config": load_config("baseline") if policy["mode"] == "residual" else None,
        "config": cfg,
        "training_seed": int(cfg["seed"]),
        "total_timesteps": int(timesteps),
        "evaluation": evaluation,
        **(extra or {}),
    }


def _policy_for(network: PolicyNetwork, cfg: dict[str, Any], base: RuleBasedPlanner | None) -> PPOPolicy:
    mode = cfg["policy"]["mode"]
    return PPOPolicy(network, mode, float(cfg["policy"].get("residual_scale", 0.5)), base)


def train(output: str | Path, config: dict[str, Any] | None = None, *, smoke: bool = False,
          resume: bool = False, verbose: bool = True, max_updates: int | None = None) -> dict[str, Any]:
    """Train PPO v2 into ``output``.

    ``max_updates`` stops after that many updates in this call and keeps
    ``state.pt`` so a time-boxed job can continue with ``resume=True``.
    """
    cfg = copy.deepcopy(config or load_config("ppo_v2"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])
    _seed_everything(seed)
    torch.set_num_threads(int(cfg.get("torch_threads", 1)))
    env_config = resolve_env_config(cfg.get("robot_profile"), cfg.get("environment_overrides"))
    lagrangian = cfg.get("lagrangian") or {}
    use_cost = bool(lagrangian.get("enabled", False))
    num_envs = int(cfg["vector_env"]["num_envs"])
    rollout_steps = int(cfg["rollout_steps"])
    total = int(cfg["total_timesteps"])
    epochs = int(cfg["epochs"])
    evaluation_cfg = dict(cfg.get("evaluation") or {})
    if smoke:
        total, num_envs = int(cfg["total_timesteps_smoke"]), min(num_envs, 2)
        rollout_steps, epochs = min(rollout_steps, 128), min(epochs, 2)
        evaluation_cfg.update({"episodes_per_scenario": 1, "workers": 1, "backend": "sync",
                               "scenarios": ["S0"]})
    batch = num_envs * rollout_steps
    updates = max(1, math.ceil(total / batch))
    architecture = {**cfg["network"], "cost_critic": use_cost}
    network = PolicyNetwork(architecture)
    mode = cfg["policy"]["mode"]
    base = RuleBasedPlanner(env_config, load_config("baseline")) if mode == "residual" else None
    scale = float(cfg["policy"].get("residual_scale", 0.5))
    warm = cfg.get("warm_start") or {}
    warm_loss = None
    if mode == "residual":
        network.zero_mean_head()
    elif warm.get("dataset"):
        warm_loss = warm_start_actor(network, warm["dataset"], warm, seed)
    reference = copy.deepcopy(network).eval() if float(warm.get("anchor_coef", 0.0)) > 0.0 else None
    optimizer = torch.optim.Adam(network.parameters(), lr=float(cfg["learning_rate"]), eps=1e-5)
    value_stats, cost_stats = RunningMeanStd(), RunningMeanStd()
    lagrange = float(lagrangian.get("lambda_init", 0.0))
    sampler = torch.Generator().manual_seed(seed + 1)
    start_update, timesteps = 0, 0
    best_score: tuple[float, ...] | None = None
    best_evaluation: dict[str, Any] | None = None
    state_path, log_path = output / "state.pt", output / "training.csv"
    if resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        network.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        value_stats, cost_stats = RunningMeanStd(**state["value_stats"]), RunningMeanStd(**state["cost_stats"])
        lagrange, start_update, timesteps = state["lagrange"], state["update"], state["timesteps"]
        best_score = tuple(state["best_score"]) if state["best_score"] is not None else None
        best_evaluation = state["best_evaluation"]
        sampler.set_state(state["sampler_state"])
        torch.set_rng_state(state["torch_rng"])
    else:
        log_path.unlink(missing_ok=True)
    (output / "config.json").write_text(json.dumps({"training": cfg, "environment": env_config, "smoke": smoke},
                                                   indent=2, default=str), encoding="utf-8")
    # A resumed run gets fresh slot streams; it is reproducible but not
    # bit-identical to an uninterrupted run.
    slot_seed = seed + 7919 * start_update
    specs = [SlotSpec(env_config, index, slot_seed, curriculum=cfg["curriculum"]["stages"])
             for index in range(num_envs)]
    vector = make_vector_env(specs, "sync" if smoke else str(cfg["vector_env"].get("backend", "sync")))
    gamma, lam = float(cfg["gamma"]), float(cfg["gae_lambda"])
    normalize = bool(cfg.get("value_normalization", True))
    started = time.perf_counter()
    stop_at = updates if max_updates is None else min(updates, start_update + int(max_updates))
    try:
        observations = vector.reset()
        for update in range(start_update, stop_at):
            progress = timesteps / max(1, total)
            vector.set_progress(progress)
            for group in optimizer.param_groups:
                group["lr"] = _schedule(float(cfg["learning_rate"]),
                                        float(cfg.get("learning_rate_final", cfg["learning_rate"])), progress)
            entropy_coef = _schedule(float(cfg["entropy_coef"]),
                                     float(cfg.get("entropy_coef_final", cfg["entropy_coef"])), progress)
            shape = (rollout_steps, num_envs)
            obs_buffer = np.zeros((*shape, 296), dtype=np.float32)
            latent_buffer = np.zeros((*shape, 2), dtype=np.float32)
            logp_buffer, values, cost_values = np.zeros(shape), np.zeros(shape), np.zeros(shape)
            rewards, costs, dones = np.zeros(shape), np.zeros(shape), np.zeros(shape)
            truncated_finals: list[tuple[int, int, np.ndarray]] = []
            episodes: list[dict[str, Any]] = []
            for step in range(rollout_steps):
                tensor = torch.as_tensor(observations)
                with torch.no_grad():
                    mean, log_std, value, cost_value = network(tensor)
                    distribution = Normal(mean, log_std.exp())
                    latent = mean + log_std.exp() * torch.randn(mean.shape, generator=sampler)
                    log_prob = distribution.log_prob(latent).sum(-1)
                policy_action = torch.tanh(latent).numpy()
                action = (combine_residual(np.stack([base.predict(item) for item in observations]),
                                           policy_action, scale) if base is not None else policy_action)
                next_observations, reward, cost, terminated, truncated, infos = vector.step(action)
                obs_buffer[step], latent_buffer[step], logp_buffer[step] = observations, latent.numpy(), log_prob.numpy()
                values[step] = _denormalize(value.numpy(), value_stats, normalize)
                if cost_value is not None:
                    cost_values[step] = _denormalize(cost_value.numpy(), cost_stats, normalize)
                rewards[step], costs[step] = reward, cost
                dones[step] = terminated | truncated
                for index, info in enumerate(infos):
                    if "final_observation" in info:
                        episodes.append(info)
                        if truncated[index] and not terminated[index]:
                            truncated_finals.append((step, index, info["final_observation"]))
                observations = next_observations
                timesteps += num_envs
            with torch.no_grad():
                _, _, last_value, last_cost = network(torch.as_tensor(observations))
                if truncated_finals:
                    finals = torch.as_tensor(np.stack([item[2] for item in truncated_finals]))
                    _, _, final_value, final_cost = network(finals)
            last_values = _denormalize(last_value.numpy(), value_stats, normalize)
            last_costs = (_denormalize(last_cost.numpy(), cost_stats, normalize)
                          if last_cost is not None else np.zeros(num_envs))
            if truncated_finals:
                # Time-limit truncation is not failure: bootstrap the value of
                # the final observation instead of treating it as terminal.
                final_values = _denormalize(final_value.numpy(), value_stats, normalize)
                final_costs = (_denormalize(final_cost.numpy(), cost_stats, normalize)
                               if final_cost is not None else np.zeros(len(truncated_finals)))
                for position, (step, index, _) in enumerate(truncated_finals):
                    rewards[step, index] += gamma * final_values[position]
                    costs[step, index] += gamma * final_costs[position]
            advantages = _gae(rewards, values, dones, last_values, gamma, lam)
            returns = advantages + values
            cost_advantages = _gae(costs, cost_values, dones, last_costs, gamma, lam) if use_cost else None
            cost_returns = cost_advantages + cost_values if use_cost else None
            if use_cost:
                lagrange = min(max(0.0, lagrange + float(lagrangian.get("lambda_lr", 0.05))
                                   * (float(costs.mean()) - float(lagrangian.get("cost_limit", 0.02)))),
                               float(lagrangian.get("lambda_max", 10.0)))
            if normalize:
                value_stats.update(returns.reshape(-1))
                targets = (returns - value_stats.mean) / value_stats.std
                if use_cost:
                    cost_stats.update(cost_returns.reshape(-1))
                    cost_targets = (cost_returns - cost_stats.mean) / cost_stats.std
            else:
                targets, cost_targets = returns, cost_returns
            combined = _standardize(advantages.reshape(-1))
            if use_cost:
                combined = (combined - lagrange * _standardize(cost_advantages.reshape(-1))) / (1.0 + lagrange)
            obs_t = torch.as_tensor(obs_buffer.reshape(-1, 296))
            latent_t = torch.as_tensor(latent_buffer.reshape(-1, 2))
            logp_t = torch.as_tensor(logp_buffer.reshape(-1), dtype=torch.float32)
            advantage_t = torch.as_tensor(combined, dtype=torch.float32)
            target_t = torch.as_tensor(targets.reshape(-1), dtype=torch.float32)
            cost_target_t = torch.as_tensor(cost_targets.reshape(-1), dtype=torch.float32) if use_cost else None
            stats = {"policy_loss": math.nan, "value_loss": math.nan, "cost_value_loss": math.nan,
                     "entropy": math.nan, "approx_kl": 0.0, "clip_fraction": 0.0, "kl_stop": 0}
            minibatch, clip_ratio = int(cfg["minibatch_size"]), float(cfg["clip_ratio"])
            target_kl = float(cfg.get("target_kl") or math.inf)
            for _ in range(epochs):
                permutation = torch.randperm(len(obs_t), generator=sampler)
                for start in range(0, len(obs_t), minibatch):
                    index = permutation[start:start + minibatch]
                    mean, log_std, value, cost_value = network(obs_t[index])
                    distribution = Normal(mean, log_std.exp())
                    log_ratio = distribution.log_prob(latent_t[index]).sum(-1) - logp_t[index]
                    ratio = log_ratio.exp()
                    approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())
                    if approx_kl > 1.5 * target_kl:
                        stats["kl_stop"] = 1
                        break
                    adv = advantage_t[index]
                    policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv).mean()
                    value_loss = torch.mean((value - target_t[index]) ** 2)
                    cost_loss = (torch.mean((cost_value - cost_target_t[index]) ** 2)
                                 if cost_target_t is not None else torch.zeros(()))
                    entropy = distribution.entropy().sum(-1).mean()
                    loss = (policy_loss + float(cfg["value_coef"]) * (value_loss + cost_loss)
                            - entropy_coef * entropy)
                    if reference is not None:
                        with torch.no_grad():
                            reference_mean = reference.mean(obs_t[index])
                        loss = loss + float(warm["anchor_coef"]) * torch.mean(
                            (torch.tanh(mean) - torch.tanh(reference_mean)) ** 2)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), float(cfg["max_grad_norm"]))
                    optimizer.step()
                    stats.update({"policy_loss": float(policy_loss.item()), "value_loss": float(value_loss.item()),
                                  "cost_value_loss": float(cost_loss.item()), "entropy": float(entropy.item()),
                                  "approx_kl": approx_kl,
                                  "clip_fraction": float(((ratio - 1.0).abs() > clip_ratio).float().mean().item())})
                if stats["kl_stop"]:
                    break
            row = {
                "update": update + 1, "timesteps": timesteps, "progress": timesteps / max(1, total),
                "stage": episodes[-1]["stage"] if episodes else "",
                "learning_rate": optimizer.param_groups[0]["lr"], "entropy_coef": entropy_coef,
                "episodes": len(episodes),
                "success_rate": float(np.mean([bool(e["success"]) for e in episodes])) if episodes else math.nan,
                "collision_rate": float(np.mean([bool(e["collision"]) for e in episodes])) if episodes else math.nan,
                "timeout_rate": float(np.mean([bool(e["truncated"]) for e in episodes])) if episodes else math.nan,
                "episode_return": float(np.mean([e["episode_return"] for e in episodes])) if episodes else math.nan,
                "episode_cost": float(np.mean([e["episode_cost"] for e in episodes])) if episodes else math.nan,
                "step_cost": float(costs.mean()), "lagrange": lagrange,
                "log_std_mean": float(network.log_std.mean().item()), **stats,
                "eval_success_rate": math.nan, "eval_collision_rate": math.nan, "eval_timeout_rate": math.nan,
                "eval_mean_clearance": math.nan, "eval_mean_time": math.nan,
                "wall_seconds": time.perf_counter() - started,
            }
            interval = int(evaluation_cfg.get("interval_updates", 0))
            if interval > 0 and ((update + 1) % interval == 0 or update + 1 == updates):
                protocol = load_config(str(evaluation_cfg["protocol"]))
                if "scenarios" in evaluation_cfg:
                    protocol["scenarios"] = list(evaluation_cfg["scenarios"])
                metrics, _ = evaluate_protocol(
                    _policy_for(network, cfg, base), protocol, robot_profile=cfg.get("robot_profile"),
                    episodes_per_scenario=evaluation_cfg.get("episodes_per_scenario"),
                    workers=int(evaluation_cfg.get("workers", 1)), backend=str(evaluation_cfg.get("backend", "sync")))
                network.train()
                metrics = {key: value for key, value in metrics.items() if key != "by_shield"}
                row.update({f"eval_{key}": metrics[key] for key in
                            ("success_rate", "collision_rate", "timeout_rate", "mean_clearance", "mean_time")})
                score = selection_score(metrics)
                if best_score is None or score > best_score:
                    best_score, best_evaluation = score, {**metrics, "timesteps": timesteps, "update": update + 1}
                    torch.save(checkpoint_payload(network, cfg, env_config, timesteps, best_evaluation),
                               output / "best.pt")
            with log_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            if verbose and (update == start_update or (update + 1) % max(1, updates // 20) == 0
                            or update + 1 == updates):
                print(f"PPO v2 update {update + 1}/{updates} | steps={timesteps} | episodes={row['episodes']} "
                      f"| success={row['success_rate']:.3f} | collision={row['collision_rate']:.3f} "
                      f"| kl={row['approx_kl']:.4f} | lambda={lagrange:.3f}", flush=True)
            checkpoint_every = int(cfg.get("checkpoint_interval_updates", 0))
            pausing = update + 1 == stop_at < updates
            if update + 1 < updates and (pausing or (checkpoint_every > 0 and (update + 1) % checkpoint_every == 0)):
                torch.save({"model_state": network.state_dict(), "optimizer_state": optimizer.state_dict(),
                            "value_stats": value_stats.as_dict(), "cost_stats": cost_stats.as_dict(),
                            "lagrange": lagrange, "update": update + 1, "timesteps": timesteps,
                            "best_score": list(best_score) if best_score else None,
                            "best_evaluation": best_evaluation, "sampler_state": sampler.get_state(),
                            "torch_rng": torch.get_rng_state()}, state_path)
    finally:
        vector.close()
    if stop_at < updates:
        paused = {"output": str(output), "status": "paused", "completed_updates": stop_at, "updates": updates,
                  "total_timesteps": timesteps, "state": str(state_path)}
        (output / "result.json").write_text(json.dumps(paused, indent=2), encoding="utf-8")
        return paused
    torch.save(checkpoint_payload(network, cfg, env_config, timesteps, None,
                                  {"value_stats": value_stats.as_dict(), "lagrange": lagrange}),
               output / "final.pt")
    state_path.unlink(missing_ok=True)
    result = {
        "output": str(output), "status": "complete", "final_checkpoint": str(output / "final.pt"),
        "best_checkpoint": str(output / "best.pt") if (output / "best.pt").exists() else None,
        "training_log": str(log_path), "total_timesteps": timesteps, "updates": updates,
        "best_evaluation": best_evaluation, "warm_start_mse": warm_loss, "policy_mode": mode,
        "robot_profile": env_config.get("robot_profile"), "smoke": smoke,
        "wall_seconds": time.perf_counter() - started,
        "note": "smoke run validates execution; it is not a convergence claim" if smoke else "full run",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result
