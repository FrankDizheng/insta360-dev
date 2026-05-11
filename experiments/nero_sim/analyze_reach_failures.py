"""Analyze and export the worst rollout failures for a trained reach policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.nero_sim.compare_rollout_vs_expert import _combine_paths  # noqa: E402
from experiments.nero_sim.eval_reach_policy import load_episode, load_policy, rollout_to_target  # noqa: E402
from nero.planning import deg_to_rad  # noqa: E402
from nero.types import HOME_ANGLES_DEG  # noqa: E402
from nero.visualize import preview_expert_rollout_comparison  # noqa: E402


def evaluate_episode(model, dataset_path: str | Path, episode_idx: int, *, device: torch.device, max_steps: int, success_tol_m: float) -> dict[str, object]:
    episode = load_episode(dataset_path, episode_idx)
    q_home = deg_to_rad(HOME_ANGLES_DEG)
    point_a_xyz = list(episode["point_a"]["xyz_m"])
    rollout_a = rollout_to_target(
        model,
        q_home,
        point_a_xyz,
        segment="point_a",
        device=device,
        max_steps=max_steps,
        success_tol_m=success_tol_m,
    )
    point_b_xyz = list(episode["point_b"]["xyz_m"])
    rollout_b = rollout_to_target(
        model,
        list(rollout_a["q_history_rad"][-1]),
        point_b_xyz,
        segment="point_b",
        device=device,
        max_steps=max_steps,
        success_tol_m=success_tol_m,
    )
    return {
        "episode_id": int(episode["episode_id"]),
        "point_a_success": bool(rollout_a["success"]),
        "point_a_final_error_m": float(rollout_a["final_error_m"]),
        "point_b_success": bool(rollout_b["success"]),
        "point_b_final_error_m": float(rollout_b["final_error_m"]),
    }


def render_episode_comparison(
    model,
    dataset_path: str | Path,
    episode_idx: int,
    *,
    device: torch.device,
    max_steps: int,
    success_tol_m: float,
    save_path: Path,
) -> dict[str, object]:
    episode = load_episode(dataset_path, episode_idx)
    q_home = deg_to_rad(HOME_ANGLES_DEG)
    point_a_xyz = list(episode["point_a"]["xyz_m"])
    rollout_a = rollout_to_target(
        model,
        q_home,
        point_a_xyz,
        segment="point_a",
        device=device,
        max_steps=max_steps,
        success_tol_m=success_tol_m,
    )
    point_b_xyz = list(episode["point_b"]["xyz_m"])
    rollout_b = rollout_to_target(
        model,
        list(rollout_a["q_history_rad"][-1]),
        point_b_xyz,
        segment="point_b",
        device=device,
        max_steps=max_steps,
        success_tol_m=success_tol_m,
    )

    expert_path = _combine_paths(episode["point_a"]["plan"]["path_rad"], episode["point_b"]["plan"]["path_rad"])
    rollout_path = _combine_paths(rollout_a["q_history_rad"], rollout_b["q_history_rad"])
    preview_expert_rollout_comparison(
        expert_path,
        rollout_path,
        point_a_xyz=point_a_xyz,
        point_b_xyz=point_b_xyz,
        block=False,
        save_path=str(save_path),
    )
    return {
        "episode_id": int(episode["episode_id"]),
        "image": str(save_path),
        "point_a_success": bool(rollout_a["success"]),
        "point_a_final_error_m": float(rollout_a["final_error_m"]),
        "point_b_success": bool(rollout_b["success"]),
        "point_b_final_error_m": float(rollout_b["final_error_m"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze worst rollout failures and export comparison images")
    parser.add_argument(
        "--dataset",
        default=str(REPO_ROOT / "experiments" / "nero_sim" / "outputs" / "train_calibrated512.jsonl"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "experiments" / "nero_sim" / "outputs" / "reach_policy_dagger_formal64" / "reach_policy.pt"),
    )
    parser.add_argument("--episodes", type=int, default=64, help="Number of episodes to scan from the dataset")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--success-tol-mm", type=float, default=12.0)
    parser.add_argument("--top-k", type=int, default=5, help="How many worst episodes to render")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "experiments" / "nero_sim" / "outputs" / "failure_analysis"),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_policy(args.checkpoint, device)
    success_tol_m = args.success_tol_mm / 1000.0

    rows: list[dict[str, object]] = []
    for episode_idx in range(args.episodes):
        rows.append(
            evaluate_episode(
                model,
                args.dataset,
                episode_idx,
                device=device,
                max_steps=args.max_steps,
                success_tol_m=success_tol_m,
            )
        )

    rows_sorted = sorted(
        rows,
        key=lambda r: max(float(r["point_a_final_error_m"]), float(r["point_b_final_error_m"])),
        reverse=True,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "failure_summary.json"
    summary = {
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "episodes_scanned": args.episodes,
        "top_k": args.top_k,
        "worst_episodes": rows_sorted[: args.top_k],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rendered: list[dict[str, object]] = []
    for rank, row in enumerate(rows_sorted[: args.top_k], start=1):
        episode_id = int(row["episode_id"])
        save_path = out_dir / f"failure_rank{rank:02d}_episode{episode_id}.png"
        rendered.append(
            render_episode_comparison(
                model,
                args.dataset,
                episode_id,
                device=device,
                max_steps=args.max_steps,
                success_tol_m=success_tol_m,
                save_path=save_path,
            )
        )

    rendered_path = out_dir / "rendered_failures.json"
    rendered_path.write_text(json.dumps(rendered, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved_summary: {summary_path}")
    print(f"saved_rendered: {rendered_path}")


if __name__ == "__main__":
    main()
