"""Generate DAgger-style corrective samples from policy rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.nero_sim.eval_reach_policy import load_episode, load_policy  # noqa: E402
from experiments.nero_sim.reach_policy import build_policy_input  # noqa: E402
from nero import clamp_joints_rad, tcp_position  # noqa: E402
from nero.planning import MotionPlanConfig, deg_to_rad, plan_tcp_motion  # noqa: E402
from nero.types import HOME_ANGLES_DEG  # noqa: E402


def rollout_with_corrections(
    model,
    *,
    start_q_rad: list[float],
    target_xyz: list[float],
    episode_id: int,
    segment: str,
    device: torch.device,
    max_steps: int,
    success_tol_m: float,
) -> tuple[list[dict[str, object]], list[float], dict[str, object]]:
    q = list(start_q_rad)
    corrections: list[dict[str, object]] = []
    final_summary: dict[str, object] = {}
    planner_cfg = MotionPlanConfig(tolerance_m=min(success_tol_m, 0.01))

    for step_idx in range(max_steps):
        current_tcp = tcp_position(q).tolist()
        error_m = float(torch.tensor(target_xyz).sub(torch.tensor(current_tcp)).norm().item())
        expert_plan = plan_tcp_motion(target_xyz, q, config=planner_cfg)
        if len(expert_plan.path_rad) >= 2:
            expert_next_q = list(expert_plan.path_rad[1])
        else:
            expert_next_q = list(expert_plan.goal_rad)

        corrections.append(
            {
                "record_type": "dagger",
                "episode_id": episode_id,
                "segment": segment,
                "step_index": step_idx,
                "target_xyz_m": list(target_xyz),
                "current_q_rad": list(q),
                "current_tcp_xyz_m": current_tcp,
                "current_error_m": error_m,
                "expert_next_q_rad": expert_next_q,
                "expert_goal_q_rad": list(expert_plan.goal_rad),
                "expert_position_error_m": float(expert_plan.position_error_m),
                "expert_cost": float(expert_plan.cost),
            }
        )

        if error_m <= success_tol_m:
            final_summary = {"success": True, "steps": step_idx, "final_error_m": error_m}
            break

        x = torch.tensor(
            build_policy_input(q, list(target_xyz), segment=segment, input_dim=model.input_dim),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        with torch.no_grad():
            delta = model(x)[0].cpu().numpy().tolist()
        q = clamp_joints_rad([a + da for a, da in zip(q, delta, strict=True)])
    else:
        final_tcp = tcp_position(q).tolist()
        final_error_m = float(torch.tensor(target_xyz).sub(torch.tensor(final_tcp)).norm().item())
        final_summary = {"success": False, "steps": max_steps, "final_error_m": final_error_m}

    return corrections, q, final_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DAgger corrective samples from calibrated sim-nero rollouts")
    parser.add_argument(
        "--dataset",
        default=str(REPO_ROOT / "experiments" / "nero_sim" / "outputs" / "train_calibrated512.jsonl"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "experiments" / "nero_sim" / "outputs" / "reach_policy_calibrated512" / "reach_policy.pt"),
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "experiments" / "nero_sim" / "outputs" / "dagger_calibrated.jsonl"),
    )
    parser.add_argument("--episodes", type=int, default=64, help="Number of dataset episodes to roll out")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--success-tol-mm", type=float, default=12.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_policy(args.checkpoint, device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    summary = {
        "episodes": 0,
        "point_a_successes": 0,
        "point_b_successes": 0,
        "correction_records": 0,
    }

    def _print_progress(current_episode: int) -> None:
        elapsed = max(time.perf_counter() - started_at, 1e-6)
        avg_per_episode = elapsed / max(current_episode, 1)
        remaining = max(args.episodes - current_episode, 0)
        eta = remaining * avg_per_episode
        bar_width = 24
        progress = current_episode / max(args.episodes, 1)
        filled = int(round(progress * bar_width))
        bar = "#" * filled + "-" * (bar_width - filled)
        print(
            f"[{bar}] {current_episode}/{args.episodes} "
            f"elapsed={elapsed/60.0:.1f}m "
            f"eta={eta/60.0:.1f}m "
            f"corr={summary['correction_records']} "
            f"a_ok={summary['point_a_successes']} "
            f"b_ok={summary['point_b_successes']}",
            flush=True,
        )

    with output_path.open("w", encoding="utf-8") as f:
        print(f"Generating DAgger corrections -> {output_path}", flush=True)
        for episode_idx in range(args.episodes):
            episode = load_episode(args.dataset, episode_idx)
            episode_id = int(episode["episode_id"])
            q_home = deg_to_rad(HOME_ANGLES_DEG)
            point_a_xyz = list(episode["point_a"]["xyz_m"])
            corr_a, q_after_a, result_a = rollout_with_corrections(
                model,
                start_q_rad=q_home,
                target_xyz=point_a_xyz,
                episode_id=episode_id,
                segment="point_a",
                device=device,
                max_steps=args.max_steps,
                success_tol_m=args.success_tol_mm / 1000.0,
            )
            point_b_xyz = list(episode["point_b"]["xyz_m"])
            corr_b, _q_after_b, result_b = rollout_with_corrections(
                model,
                start_q_rad=q_after_a,
                target_xyz=point_b_xyz,
                episode_id=episode_id,
                segment="point_b",
                device=device,
                max_steps=args.max_steps,
                success_tol_m=args.success_tol_mm / 1000.0,
            )
            for record in corr_a + corr_b:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")

            summary["episodes"] += 1
            summary["point_a_successes"] += int(bool(result_a["success"]))
            summary["point_b_successes"] += int(bool(result_b["success"]))
            summary["correction_records"] += len(corr_a) + len(corr_b)
            f.flush()
            _print_progress(summary["episodes"])

    print(json.dumps(summary, indent=2))
    print(f"saved_dagger_dataset: {output_path}")


if __name__ == "__main__":
    main()
