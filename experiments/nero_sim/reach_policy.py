"""Dataset and model helpers for A-to-B reach policy training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.utils.data import Dataset

from nero import tcp_position


INPUT_DIM = 16  # 7 joint angles + current tcp xyz + target xyz + tcp error xyz
ACTION_DIM = 7  # next joint delta


@dataclass
class ReachSample:
    input_vec: list[float]
    action_delta: list[float]
    tcp_target_xyz: list[float]
    current_q_rad: list[float]
    next_q_rad: list[float]
    segment: str
    episode_id: int
    step_index: int


def build_policy_input(current_q: list[float], target_xyz: list[float]) -> list[float]:
    current_tcp_xyz = tcp_position(current_q).tolist()
    tcp_error_xyz = [t - c for t, c in zip(target_xyz, current_tcp_xyz, strict=True)]
    return current_q + current_tcp_xyz + target_xyz + tcp_error_xyz


def _segment_to_samples(
    episode_id: int,
    segment_name: str,
    segment_payload: dict[str, Any],
) -> list[ReachSample]:
    plan = segment_payload["plan"]
    path_rad = plan["path_rad"]
    target_xyz = list(segment_payload["xyz_m"])
    samples: list[ReachSample] = []
    for idx in range(len(path_rad) - 1):
        current_q = list(path_rad[idx])
        next_q = list(path_rad[idx + 1])
        delta = [n - c for c, n in zip(current_q, next_q, strict=True)]
        input_vec = build_policy_input(current_q, target_xyz)
        samples.append(
            ReachSample(
                input_vec=input_vec,
                action_delta=delta,
                tcp_target_xyz=target_xyz,
                current_q_rad=current_q,
                next_q_rad=next_q,
                segment=segment_name,
                episode_id=episode_id,
                step_index=idx,
            )
        )
    return samples


def _dagger_record_to_sample(record: dict[str, Any]) -> ReachSample:
    current_q = list(record["current_q_rad"])
    next_q = list(record["expert_next_q_rad"])
    target_xyz = list(record["target_xyz_m"])
    delta = [n - c for c, n in zip(current_q, next_q, strict=True)]
    return ReachSample(
        input_vec=build_policy_input(current_q, target_xyz),
        action_delta=delta,
        tcp_target_xyz=target_xyz,
        current_q_rad=current_q,
        next_q_rad=next_q,
        segment=str(record.get("segment", "dagger")),
        episode_id=int(record.get("episode_id", -1)),
        step_index=int(record.get("step_index", 0)),
    )


def _iter_dataset_paths(dataset_paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(dataset_paths, (str, Path)):
        return [Path(dataset_paths)]
    return [Path(p) for p in dataset_paths]


def load_reach_samples(dataset_paths: str | Path | Iterable[str | Path]) -> list[ReachSample]:
    samples: list[ReachSample] = []
    paths = _iter_dataset_paths(dataset_paths)
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                payload = json.loads(line)
                record_type = payload.get("record_type", "episode")
                if record_type == "dagger":
                    samples.append(_dagger_record_to_sample(payload))
                    continue
                episode_id = int(payload["episode_id"])
                samples.extend(_segment_to_samples(episode_id, "point_a", payload["point_a"]))
                samples.extend(_segment_to_samples(episode_id, "point_b", payload["point_b"]))
    if not samples:
        raise RuntimeError(f"No trajectory samples found in {paths}")
    return samples


class ReachTrajectoryDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, dataset_paths: str | Path | Iterable[str | Path]):
        self.samples = load_reach_samples(dataset_paths)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "input": torch.tensor(sample.input_vec, dtype=torch.float32),
            "action": torch.tensor(sample.action_delta, dtype=torch.float32),
            "target_xyz": torch.tensor(sample.tcp_target_xyz, dtype=torch.float32),
            "current_q": torch.tensor(sample.current_q_rad, dtype=torch.float32),
            "next_q": torch.tensor(sample.next_q_rad, dtype=torch.float32),
        }


class ReachPolicyMLP(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ACTION_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
