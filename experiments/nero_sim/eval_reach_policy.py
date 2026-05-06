"""Roll out a trained reach policy on random A->B episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.nero_sim.reach_policy import (  # noqa: E402
    LEGACY_INPUT_DIM,
    ReachPolicyMLP,
    build_policy_input,
    feature_mode_to_input_dim,
)
from nero import clamp_joints_rad, tcp_position  # noqa: E402
from nero.types import HOME_ANGLES_DEG  # noqa: E402
from nero.planning import deg_to_rad, rad_to_deg  # noqa: E402


def load_episode(dataset_path: str | Path, episode_idx: int) -> dict:
    with Path(dataset_path).open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == episode_idx:
                return json.loads(line)
    raise IndexError(f"Episode {episode_idx} not found in {dataset_path}")


def load_policy(checkpoint_path: str | Path, device: torch.device) -> ReachPolicyMLP:
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload["model_state_dict"]
    first_weight = state_dict["net.0.weight"]
    hidden_dim = int(payload.get("hidden_dim", first_weight.shape[0]))
    feature_mode = payload.get("feature_mode")
    if feature_mode is not None:
        input_dim = int(payload.get("input_dim", feature_mode_to_input_dim(str(feature_mode))))
    else:
        input_dim = int(payload.get("input_dim", first_weight.shape[1] if hasattr(first_weight, "shape") else LEGACY_INPUT_DIM))
    model = ReachPolicyMLP(hidden_dim=hidden_dim, input_dim=input_dim).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def rollout_to_target(
    model: ReachPolicyMLP,
    start_q_rad: list[float],
    target_xyz: list[float],
    *,
    segment: str = "point_a",
    device: torch.device,
    max_steps: int = 80,
    success_tol_m: float = 0.01,
) -> dict[str, object]:
    q = list(start_q_rad)
    q_history = [list(q)]
    tcp_history = [tcp_position(q).tolist()]
    for step in range(max_steps):
        x = torch.tensor(
            build_policy_input(q, list(target_xyz), segment=segment, input_dim=model.input_dim),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        with torch.no_grad():
            delta = model(x)[0].cpu().numpy().tolist()
        q = clamp_joints_rad([a + da for a, da in zip(q, delta, strict=True)])
        tip = tcp_position(q).tolist()
        q_history.append(list(q))
        tcp_history.append(tip)
        err = float(np.linalg.norm(np.asarray(tip) - np.asarray(target_xyz)))
        if err <= success_tol_m:
            return {
                "success": True,
                "steps": step + 1,
                "final_error_m": err,
                "q_history_rad": q_history,
                "q_history_deg": [rad_to_deg(waypoint) for waypoint in q_history],
                "tcp_history_xyz_m": tcp_history,
            }
    final_err = float(np.linalg.norm(np.asarray(tcp_history[-1]) - np.asarray(target_xyz)))
    return {
        "success": False,
        "steps": max_steps,
        "final_error_m": final_err,
        "q_history_rad": q_history,
        "q_history_deg": [rad_to_deg(waypoint) for waypoint in q_history],
        "tcp_history_xyz_m": tcp_history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained NERO reach policy on one episode")
    parser.add_argument(
        "--dataset",
        default=str(REPO_ROOT / "experiments" / "nero_sim" / "outputs" / "random_reach_dataset.jsonl"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "experiments" / "nero_sim" / "outputs" / "reach_policy" / "reach_policy.pt"),
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--success-tol-mm", type=float, default=10.0)
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON path for rollout results",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_policy(args.checkpoint, device)
    episode = load_episode(args.dataset, args.episode)
    success_tol_m = args.success_tol_mm / 1000.0

    q_home = deg_to_rad(HOME_ANGLES_DEG)
    point_a_xyz = list(episode["point_a"]["xyz_m"])
    result_a = rollout_to_target(
        model,
        q_home,
        point_a_xyz,
        segment="point_a",
        device=device,
        max_steps=args.max_steps,
        success_tol_m=success_tol_m,
    )
    start_b = list(result_a["q_history_rad"][-1])
    point_b_xyz = list(episode["point_b"]["xyz_m"])
    result_b = rollout_to_target(
        model,
        start_b,
        point_b_xyz,
        segment="point_b",
        device=device,
        max_steps=args.max_steps,
        success_tol_m=success_tol_m,
    )

    summary = {
        "episode_id": int(episode["episode_id"]),
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "point_a": result_a,
        "point_b": result_b,
    }
    print(json.dumps(
        {
            "episode_id": summary["episode_id"],
            "point_a_success": result_a["success"],
            "point_a_final_error_m": result_a["final_error_m"],
            "point_b_success": result_b["success"],
            "point_b_final_error_m": result_b["final_error_m"],
        },
        indent=2,
    ))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
