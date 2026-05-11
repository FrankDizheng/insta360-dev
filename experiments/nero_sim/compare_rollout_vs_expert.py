"""Render an overlay of expert and policy rollout trajectories for one episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.nero_sim.eval_reach_policy import load_episode, load_policy, rollout_to_target  # noqa: E402
from nero.planning import deg_to_rad  # noqa: E402
from nero.types import HOME_ANGLES_DEG  # noqa: E402
from nero.visualize import preview_expert_rollout_comparison  # noqa: E402


def _combine_paths(path_a: list[list[float]], path_b: list[list[float]]) -> list[list[float]]:
    combined = list(path_a)
    if path_b:
        if combined and path_b and combined[-1] == path_b[0]:
            combined.extend(path_b[1:])
        else:
            combined.extend(path_b)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare expert and rollout trajectories for one NERO episode")
    parser.add_argument(
        "--dataset",
        default=str(REPO_ROOT / "experiments" / "nero_sim" / "outputs" / "train_medium512.jsonl"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "experiments" / "nero_sim" / "outputs" / "reach_policy_medium512" / "reach_policy.pt"),
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--success-tol-mm", type=float, default=12.0)
    parser.add_argument("--save", default="", help="Optional output image path")
    parser.add_argument("--summary-output", default="", help="Optional JSON summary output path")
    parser.add_argument("--no-block", action="store_true", help="Do not block on the matplotlib window")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_policy(args.checkpoint, device)
    episode = load_episode(args.dataset, args.episode)

    q_home = deg_to_rad(HOME_ANGLES_DEG)
    point_a_xyz = list(episode["point_a"]["xyz_m"])
    rollout_a = rollout_to_target(
        model,
        q_home,
        point_a_xyz,
        segment="point_a",
        device=device,
        max_steps=args.max_steps,
        success_tol_m=args.success_tol_mm / 1000.0,
    )
    point_b_xyz = list(episode["point_b"]["xyz_m"])
    rollout_b = rollout_to_target(
        model,
        list(rollout_a["q_history_rad"][-1]),
        point_b_xyz,
        segment="point_b",
        device=device,
        max_steps=args.max_steps,
        success_tol_m=args.success_tol_mm / 1000.0,
    )

    expert_path = _combine_paths(episode["point_a"]["plan"]["path_rad"], episode["point_b"]["plan"]["path_rad"])
    rollout_path = _combine_paths(rollout_a["q_history_rad"], rollout_b["q_history_rad"])

    preview_expert_rollout_comparison(
        expert_path,
        rollout_path,
        point_a_xyz=point_a_xyz,
        point_b_xyz=point_b_xyz,
        block=not args.no_block,
        save_path=args.save or None,
    )

    summary = {
        "episode_id": int(episode["episode_id"]),
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "point_a_success": rollout_a["success"],
        "point_a_final_error_m": rollout_a["final_error_m"],
        "point_b_success": rollout_b["success"],
        "point_b_final_error_m": rollout_b["final_error_m"],
        "expert_frames": len(expert_path),
        "rollout_frames": len(rollout_path),
    }
    print(json.dumps(summary, indent=2))
    if args.summary_output:
        out = Path(args.summary_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
