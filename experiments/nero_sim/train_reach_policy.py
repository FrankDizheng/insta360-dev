"""Train a minimal behavior-cloning policy for random NERO reach trajectories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.nero_sim.reach_policy import ReachPolicyMLP, ReachTrajectoryDataset  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    criterion = nn.SmoothL1Loss(reduction="sum")
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            y = batch["action"].to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += float(loss.item())
            total_count += int(x.shape[0])
    return total_loss / max(total_count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal NERO reach behavior-cloning policy")
    parser.add_argument(
        "--dataset",
        default=str(REPO_ROOT / "experiments" / "nero_sim" / "outputs" / "random_reach_dataset.jsonl"),
        help="Path to JSONL expert dataset",
    )
    parser.add_argument(
        "--extra-dataset",
        action="append",
        default=[],
        help="Optional extra dataset path(s), e.g. DAgger corrective JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "experiments" / "nero_sim" / "outputs" / "reach_policy"),
        help="Directory to save checkpoint and metrics",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_paths = [args.dataset] + list(args.extra_dataset)
    dataset = ReachTrajectoryDataset(dataset_paths)
    val_size = max(1, int(math.ceil(len(dataset) * args.val_split)))
    train_size = max(1, len(dataset) - val_size)
    if train_size + val_size > len(dataset):
        val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReachPolicyMLP(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.SmoothL1Loss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "reach_policy.pt"
    metrics_path = output_dir / "metrics.json"

    best_val = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for batch in train_loader:
            x = batch["input"].to(device)
            y = batch["action"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * int(x.shape[0])
            sample_count += int(x.shape[0])

        train_loss = running_loss / max(sample_count, 1)
        val_loss = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hidden_dim": args.hidden_dim,
                    "dataset": str(args.dataset),
                    "best_val_loss": best_val,
                },
                ckpt_path,
            )

    metrics = {
        "dataset": str(args.dataset),
        "extra_datasets": list(args.extra_dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(device),
        "train_samples": train_size,
        "val_samples": val_size,
        "best_val_loss": best_val,
        "history": history,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved_checkpoint: {ckpt_path}")
    print(f"saved_metrics: {metrics_path}")


if __name__ == "__main__":
    main()
