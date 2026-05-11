"""Generate expert A->B reach trajectories for SimNERO training.

Each sample contains:
  - a reachable point A from the home pose
  - a reachable point B from the joint configuration that reaches A
  - the expert joint-space paths for home->A and A->B

The output is JSONL so later training code can stream it easily.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nero.planning import (  # noqa: E402
    ReachSample,
    WorkspaceSamplerConfig,
    deg_to_rad,
    planner_result_to_dict,
    rad_to_deg,
    sample_reachable_target,
)
from nero.types import HOME_ANGLES_DEG  # noqa: E402


def _reach_sample_to_dict(sample: ReachSample) -> dict[str, object]:
    payload = {
        "xyz_m": list(sample.xyz_m),
        "attempts": sample.attempts,
        "plan": planner_result_to_dict(sample.plan),
    }
    payload["plan"]["start_deg"] = rad_to_deg(sample.plan.start_rad)
    payload["plan"]["goal_deg"] = rad_to_deg(sample.plan.goal_rad)
    payload["plan"]["path_deg"] = [rad_to_deg(waypoint) for waypoint in sample.plan.path_rad]
    return payload


def build_episode(
    rng: np.random.Generator,
    config: WorkspaceSamplerConfig,
    episode_id: int,
) -> dict[str, object]:
    home_rad = deg_to_rad(HOME_ANGLES_DEG)
    point_a = sample_reachable_target(rng, home_rad, config=config)
    point_b = sample_reachable_target(rng, point_a.plan.goal_rad, config=config)
    return {
        "episode_id": episode_id,
        "task": "random_reach_a_to_b",
        "home_deg": list(HOME_ANGLES_DEG),
        "point_a": _reach_sample_to_dict(point_a),
        "point_b": _reach_sample_to_dict(point_b),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random SimNERO expert trajectories")
    parser.add_argument("--samples", type=int, default=100, help="Number of A->B episodes to generate")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "experiments" / "nero_sim" / "outputs" / "random_reach_dataset.jsonl"),
        help="Destination JSONL file",
    )
    parser.add_argument("--x-min", type=float, default=-0.18)
    parser.add_argument("--x-max", type=float, default=0.18)
    parser.add_argument("--y-min", type=float, default=0.28)
    parser.add_argument("--y-max", type=float, default=0.52)
    parser.add_argument("--z-min", type=float, default=0.16)
    parser.add_argument("--z-max", type=float, default=0.30)
    parser.add_argument("--tolerance-mm", type=float, default=4.0, help="Planner tolerance in millimetres")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = WorkspaceSamplerConfig(
        x_range_m=(args.x_min, args.x_max),
        y_range_m=(args.y_min, args.y_max),
        z_range_m=(args.z_min, args.z_max),
        tolerance_m=args.tolerance_mm / 1000.0,
    )
    rng = np.random.default_rng(args.seed)

    with output_path.open("w", encoding="utf-8") as f:
        for episode_id in range(args.samples):
            episode = build_episode(rng, config, episode_id)
            f.write(json.dumps(episode, ensure_ascii=True) + "\n")

    print(f"Wrote {args.samples} episodes to {output_path}")


if __name__ == "__main__":
    main()
