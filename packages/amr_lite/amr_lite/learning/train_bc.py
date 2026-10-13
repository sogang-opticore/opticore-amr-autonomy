from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from amr_lite.config import ROOT, load_config

from .dataset import episode_split, load_dataset
from .networks import ActorMLP


def train(dataset_dir: str | Path | list[str | Path], output: str | Path, smoke: bool = True,
          config: dict | None = None) -> dict[str, float]:
    cfg = config or load_config("bc")
    torch.set_num_threads(1)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    data = load_dataset(dataset_dir)
    train_idx, validation_idx = episode_split(data, cfg["validation_fraction"], cfg["seed"])
    if not len(train_idx):
        train_idx = np.arange(len(data.observations))
    risk_weight = float(cfg.get("risk_weight", 1.0))
    weighted = risk_weight != 1.0 and data.minimum_clearance is not None
    tensors = [torch.from_numpy(data.observations[train_idx]), torch.from_numpy(data.actions[train_idx])]
    if weighted:
        # Up-weight states close to obstacles (DAgger risk states) in the loss.
        risky = data.minimum_clearance[train_idx] < float(cfg.get("risk_clearance", 0.7))
        tensors.append(torch.from_numpy(np.where(risky, risk_weight, 1.0).astype(np.float32)))
    loader = DataLoader(TensorDataset(*tensors),
                        batch_size=cfg["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(cfg["seed"]))
    model = ActorMLP(hidden_size=cfg["hidden_size"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.MSELoss()
    epochs = cfg["epochs_smoke"] if smoke else cfg["epochs_full"]
    training_loss = 0.0
    model.train()
    for _ in range(epochs):
        for batch in loader:
            observations, actions = batch[0], batch[1]
            prediction = model(observations)
            if weighted:
                weights = batch[2][:, None]
                loss = torch.sum(weights * (prediction - actions) ** 2) / (2.0 * torch.sum(weights))
            else:
                loss = loss_fn(prediction, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss = float(loss.item())
    model.eval()
    indices = validation_idx if len(validation_idx) else train_idx
    with torch.no_grad():
        truth = torch.from_numpy(data.actions[indices])
        prediction = model(torch.from_numpy(data.observations[indices]))
        mse = float(nn.functional.mse_loss(prediction, truth).item())
        mae = torch.mean(torch.abs(prediction - truth), dim=0).numpy()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "bc", "hidden_size": cfg["hidden_size"], "model_state": model.state_dict(),
                "config": cfg, "metrics": {"mse": mse, "v_mae": float(mae[0]), "w_mae": float(mae[1])}}, target)
    metrics = {"train_loss": training_loss, "mse": mse, "v_mae": float(mae[0]), "w_mae": float(mae[1]),
               "train_samples": int(len(train_idx)), "validation_samples": int(len(indices))}
    target.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train behavior cloning policy")
    parser.add_argument("--dataset", nargs="+", default=[str(ROOT / "artifacts/datasets/teacher")])
    parser.add_argument("--output", default=str(ROOT / "artifacts/checkpoints/bc_smoke.pt"))
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args.dataset, args.output, not args.full), indent=2))


if __name__ == "__main__":
    main()
