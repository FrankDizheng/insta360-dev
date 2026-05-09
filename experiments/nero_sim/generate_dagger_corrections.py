"""Generate DAgger-style corrective samples from policy rollouts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
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
from nero.planning import MotionPlanConfig, deg_to_rad, motion_plan_config_for_preset, plan_tcp_motion  # noqa: E402
from nero.types import JOINT_LIMITS_RAD  # noqa: E402
from nero.types import HOME_ANGLES_DEG  # noqa: E402


@dataclass
class TeacherProfile:
    cached_labels: int = 0
    fast_replans: int = 0
    heavy_replans: int = 0
    consistency_rejects: int = 0
    skipped_records: int = 0
    teacher_times_ms: list[float] = field(default_factory=list)

    def add_teacher_time(self, elapsed_s: float) -> None:
        self.teacher_times_ms.append(elapsed_s * 1000.0)

    def to_summary(self) -> dict[str, object]:
        avg_ms = sum(self.teacher_times_ms) / max(len(self.teacher_times_ms), 1)
        max_ms = max(self.teacher_times_ms) if self.teacher_times_ms else 0.0
        return {
            "cached_labels": self.cached_labels,
            "fast_replans": self.fast_replans,
            "heavy_replans": self.heavy_replans,
            "consistency_rejects": self.consistency_rejects,
            "skipped_records": self.skipped_records,
            "avg_teacher_ms": avg_ms,
            "max_teacher_ms": max_ms,
        }


def _dist_xyz(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b, strict=True)))


def _joint_l2(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b, strict=True)))


def _joint_delta(a: list[float], b: list[float]) -> list[float]:
    return [float(next_v) - float(current_v) for current_v, next_v in zip(a, b, strict=True)]


def _delta_l2(delta: list[float]) -> float:
    return math.sqrt(sum(float(v) ** 2 for v in delta))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_norm = _delta_l2(a)
    b_norm = _delta_l2(b)
    if a_norm < 1e-9 or b_norm < 1e-9:
        return 1.0
    return sum(float(x) * float(y) for x, y in zip(a, b, strict=True)) / (a_norm * b_norm)


def _bounded_next_q(current_q: list[float], desired_q: list[float], *, max_delta_rad: float) -> list[float]:
    bounded: list[float] = []
    for current, desired in zip(current_q, desired_q, strict=True):
        delta = max(-max_delta_rad, min(max_delta_rad, float(desired) - float(current)))
        bounded.append(float(current) + delta)
    return clamp_joints_rad(bounded)


def _plan_next_q(plan: object, q: list[float], *, max_delta_rad: float) -> list[float]:
    if len(plan.path_rad) >= 2:
        return _bounded_next_q(q, list(plan.path_rad[1]), max_delta_rad=max_delta_rad)
    return _bounded_next_q(q, list(plan.goal_rad), max_delta_rad=max_delta_rad)


def _near_joint_limit(q: list[float], *, margin_frac: float = 0.06) -> bool:
    for angle, (lo, hi) in zip(q, JOINT_LIMITS_RAD, strict=True):
        span = max(hi - lo, 1e-6)
        if min(angle - lo, hi - angle) < margin_frac * span:
            return True
    return False


def _plan_with_timing(
    target_xyz: list[float],
    q: list[float],
    *,
    config: MotionPlanConfig,
    profile: TeacherProfile,
) -> object:
    started = time.perf_counter()
    plan = plan_tcp_motion(target_xyz, q, config=config)
    profile.add_teacher_time(time.perf_counter() - started)
    return plan


def _cached_next_q(
    q: list[float],
    current_tcp: list[float],
    cached_path: list[list[float]],
    cached_tcps: list[list[float]],
    *,
    correction_gain: float,
    max_delta_rad: float,
) -> tuple[list[float] | None, int, float, float]:
    if len(cached_path) < 2:
        return None, -1, float("inf"), float("inf")

    best_idx = 0
    best_joint_dist = float("inf")
    for idx, waypoint in enumerate(cached_path[:-1]):
        dist = _joint_l2(q, waypoint)
        if dist < best_joint_dist:
            best_idx = idx
            best_joint_dist = dist

    tcp_dist = _dist_xyz(current_tcp, cached_tcps[best_idx]) if best_idx < len(cached_tcps) else float("inf")
    nearest_q = cached_path[best_idx]
    path_next_q = cached_path[best_idx + 1]
    path_delta = [n - c for c, n in zip(nearest_q, path_next_q, strict=True)]
    correction_delta = [correction_gain * (c - current) for current, c in zip(q, nearest_q, strict=True)]
    desired_q = [current + d_path + d_corr for current, d_path, d_corr in zip(q, path_delta, correction_delta, strict=True)]
    return _bounded_next_q(q, desired_q, max_delta_rad=max_delta_rad), best_idx, best_joint_dist, tcp_dist


def _consistency_reference_next_q(
    q: list[float],
    current_tcp: list[float],
    anchor_path: list[list[float]],
    anchor_tcps: list[list[float]],
    *,
    correction_gain: float,
    max_delta_rad: float,
) -> tuple[list[float] | None, int, float, float]:
    return _cached_next_q(
        q,
        current_tcp,
        anchor_path,
        anchor_tcps,
        correction_gain=correction_gain,
        max_delta_rad=max_delta_rad,
    )


def _fast_label_consistent(
    *,
    q: list[float],
    fast_next_q: list[float],
    reference_next_q: list[float] | None,
    reference_joint_dist: float,
    reference_tcp_dist: float,
    max_reference_joint_dist: float,
    max_reference_tcp_dist_m: float,
    max_action_l2_diff_rad: float,
    min_cosine: float,
) -> tuple[bool, dict[str, float]]:
    if reference_next_q is None:
        return False, {
            "consistency_cosine": -1.0,
            "consistency_action_l2_diff_rad": float("inf"),
            "consistency_reference_joint_dist_rad": reference_joint_dist,
            "consistency_reference_tcp_dist_m": reference_tcp_dist,
        }

    fast_delta = _joint_delta(q, fast_next_q)
    reference_delta = _joint_delta(q, reference_next_q)
    action_l2_diff = _joint_l2(fast_delta, reference_delta)
    cosine = _cosine_similarity(fast_delta, reference_delta)
    ok = (
        reference_joint_dist <= max_reference_joint_dist
        and reference_tcp_dist <= max_reference_tcp_dist_m
        and action_l2_diff <= max_action_l2_diff_rad
        and cosine >= min_cosine
    )
    return ok, {
        "consistency_cosine": cosine,
        "consistency_action_l2_diff_rad": action_l2_diff,
        "consistency_reference_joint_dist_rad": reference_joint_dist,
        "consistency_reference_tcp_dist_m": reference_tcp_dist,
    }


def _should_keep_record(
    *,
    segment: str,
    step_idx: int,
    error_m: float,
    min_record_error_m: float,
    keep_every_n_success: int,
    teacher_source: str,
    is_hard_case: bool,
) -> bool:
    if segment == "point_b":
        return True
    if is_hard_case or teacher_source == "full_recheck":
        return True
    if error_m >= min_record_error_m:
        return True
    return step_idx % max(1, keep_every_n_success) == 0


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
    teacher_mode: str,
    fast_planner_cfg: MotionPlanConfig,
    full_planner_cfg: MotionPlanConfig,
    replan_interval: int,
    replan_tcp_threshold_m: float,
    replan_joint_threshold_rad: float,
    min_record_error_m: float,
    keep_every_n_success: int,
    hard_case_full_recheck: bool,
    cached_correction_gain: float,
    max_teacher_delta_rad: float,
    enable_teacher_consistency: bool,
    consistency_correction_gain: float,
    consistency_joint_threshold_rad: float,
    consistency_tcp_threshold_m: float,
    consistency_action_l2_threshold_rad: float,
    consistency_cosine_min: float,
) -> tuple[list[dict[str, object]], list[float], dict[str, object], TeacherProfile]:
    q = list(start_q_rad)
    corrections: list[dict[str, object]] = []
    final_summary: dict[str, object] = {}
    profile = TeacherProfile()
    cached_path: list[list[float]] = []
    cached_tcps: list[list[float]] = []
    anchor_path: list[list[float]] = []
    anchor_tcps: list[list[float]] = []
    previous_error_m = float("inf")

    if teacher_mode == "cached_staged" or (teacher_mode == "every_step_fast" and enable_teacher_consistency):
        full_plan = _plan_with_timing(target_xyz, q, config=full_planner_cfg, profile=profile)
        profile.heavy_replans += 1
        anchor_path = [list(waypoint) for waypoint in full_plan.path_rad]
        anchor_tcps = [tcp_position(waypoint).tolist() for waypoint in anchor_path]
        if teacher_mode == "cached_staged":
            cached_path = list(anchor_path)
            cached_tcps = list(anchor_tcps)

    for step_idx in range(max_steps):
        current_tcp = tcp_position(q).tolist()
        error_m = _dist_xyz(list(target_xyz), current_tcp)
        expert_plan = None
        expert_next_q: list[float] | None = None
        expert_goal_q: list[float] = list(q)
        expert_position_error_m = 0.0
        expert_cost = 0.0
        teacher_source = "cached"
        cache_idx = -1
        cache_joint_dist = float("inf")
        cache_tcp_dist = float("inf")
        consistency_details = {
            "consistency_cosine": 1.0,
            "consistency_action_l2_diff_rad": 0.0,
            "consistency_reference_joint_dist_rad": 0.0,
            "consistency_reference_tcp_dist_m": 0.0,
        }

        if teacher_mode in {"every_step_staged", "every_step_fast"}:
            if teacher_mode == "every_step_fast":
                expert_plan = _plan_with_timing(target_xyz, q, config=fast_planner_cfg, profile=profile)
                profile.fast_replans += 1
                teacher_source = "fast"
                fast_next_q = _plan_next_q(expert_plan, q, max_delta_rad=max_teacher_delta_rad)
                hard_case = (
                    not bool(expert_plan.ok)
                    or float(expert_plan.position_error_m) > max(success_tol_m, fast_planner_cfg.tolerance_m)
                    or _near_joint_limit(q)
                )
                if enable_teacher_consistency:
                    ref_next_q, _ref_idx, ref_joint_dist, ref_tcp_dist = _consistency_reference_next_q(
                        q,
                        current_tcp,
                        anchor_path,
                        anchor_tcps,
                        correction_gain=consistency_correction_gain,
                        max_delta_rad=max_teacher_delta_rad,
                    )
                    consistent, consistency_details = _fast_label_consistent(
                        q=q,
                        fast_next_q=fast_next_q,
                        reference_next_q=ref_next_q,
                        reference_joint_dist=ref_joint_dist,
                        reference_tcp_dist=ref_tcp_dist,
                        max_reference_joint_dist=consistency_joint_threshold_rad,
                        max_reference_tcp_dist_m=consistency_tcp_threshold_m,
                        max_action_l2_diff_rad=consistency_action_l2_threshold_rad,
                        min_cosine=consistency_cosine_min,
                    )
                    if not consistent:
                        hard_case = True
                        profile.consistency_rejects += 1
                if hard_case_full_recheck and hard_case:
                    expert_plan = _plan_with_timing(target_xyz, q, config=full_planner_cfg, profile=profile)
                    profile.heavy_replans += 1
                    teacher_source = "full_recheck"
                    anchor_path = [list(waypoint) for waypoint in expert_plan.path_rad]
                    anchor_tcps = [tcp_position(waypoint).tolist() for waypoint in anchor_path]
            else:
                expert_plan = _plan_with_timing(target_xyz, q, config=full_planner_cfg, profile=profile)
                profile.heavy_replans += 1
                teacher_source = "full"
        else:
            expert_next_q, cache_idx, cache_joint_dist, cache_tcp_dist = _cached_next_q(
                q,
                current_tcp,
                cached_path,
                cached_tcps,
                correction_gain=cached_correction_gain,
                max_delta_rad=max_teacher_delta_rad,
            )
            error_rebounded = previous_error_m < float("inf") and error_m > previous_error_m + 0.005
            needs_replan = (
                expert_next_q is None
                or (step_idx > 0 and step_idx % max(1, replan_interval) == 0)
                or cache_tcp_dist > replan_tcp_threshold_m
                or cache_joint_dist > replan_joint_threshold_rad
                or error_rebounded
            )
            if needs_replan:
                expert_plan = _plan_with_timing(target_xyz, q, config=fast_planner_cfg, profile=profile)
                profile.fast_replans += 1
                teacher_source = "fast"
                hard_case = (
                    not bool(expert_plan.ok)
                    or float(expert_plan.position_error_m) > max(success_tol_m, fast_planner_cfg.tolerance_m)
                    or _near_joint_limit(q)
                    or error_rebounded
                )
                if hard_case_full_recheck and hard_case:
                    expert_plan = _plan_with_timing(target_xyz, q, config=full_planner_cfg, profile=profile)
                    profile.heavy_replans += 1
                    teacher_source = "full_recheck"
                cached_path = [list(waypoint) for waypoint in expert_plan.path_rad]
                cached_tcps = [tcp_position(waypoint).tolist() for waypoint in cached_path]
            else:
                profile.cached_labels += 1

        if expert_plan is not None:
            if len(expert_plan.path_rad) >= 2:
                expert_next_q = _bounded_next_q(q, list(expert_plan.path_rad[1]), max_delta_rad=max_teacher_delta_rad)
            else:
                expert_next_q = _bounded_next_q(q, list(expert_plan.goal_rad), max_delta_rad=max_teacher_delta_rad)
            expert_goal_q = list(expert_plan.goal_rad)
            expert_position_error_m = float(expert_plan.position_error_m)
            expert_cost = float(expert_plan.cost)
        elif expert_next_q is not None:
            expert_goal_q = list(cached_path[-1]) if cached_path else list(expert_next_q)
            expert_position_error_m = cache_tcp_dist

        if expert_next_q is None:
            expert_next_q = list(q)

        is_hard_case = teacher_source == "full_recheck" or _near_joint_limit(q) or error_m > 0.20
        if _should_keep_record(
            segment=segment,
            step_idx=step_idx,
            error_m=error_m,
            min_record_error_m=min_record_error_m,
            keep_every_n_success=keep_every_n_success,
            teacher_source=teacher_source,
            is_hard_case=is_hard_case,
        ):
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
                    "expert_goal_q_rad": expert_goal_q,
                    "expert_position_error_m": expert_position_error_m,
                    "expert_cost": expert_cost,
                    "teacher_source": teacher_source,
                    "cache_index": cache_idx,
                    "cache_joint_distance_rad": cache_joint_dist,
                    "cache_tcp_distance_m": cache_tcp_dist,
                    **consistency_details,
                }
            )
        else:
            profile.skipped_records += 1

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
        previous_error_m = error_m
    else:
        final_tcp = tcp_position(q).tolist()
        final_error_m = _dist_xyz(list(target_xyz), final_tcp)
        final_summary = {"success": False, "steps": max_steps, "final_error_m": final_error_m}

    final_summary["profile"] = profile.to_summary()
    return corrections, q, final_summary, profile


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
    parser.add_argument("--teacher-mode", choices=["every_step_staged", "every_step_fast", "cached_staged"], default="cached_staged")
    parser.add_argument("--planner-preset", choices=["dagger_fast", "dagger_local", "full_staged"], default="dagger_fast")
    parser.add_argument("--replan-interval", type=int, default=8)
    parser.add_argument("--replan-tcp-threshold-mm", type=float, default=25.0)
    parser.add_argument("--replan-joint-threshold-rad", type=float, default=0.18)
    parser.add_argument("--min-record-error-mm", type=float, default=8.0)
    parser.add_argument("--keep-every-n-success", type=int, default=4)
    parser.add_argument("--hard-case-full-recheck", action="store_true")
    parser.add_argument("--cached-correction-gain", type=float, default=0.35)
    parser.add_argument("--max-teacher-delta-rad", type=float, default=0.08)
    parser.add_argument("--enable-teacher-consistency", action="store_true")
    parser.add_argument("--consistency-correction-gain", type=float, default=0.25)
    parser.add_argument("--consistency-joint-threshold-rad", type=float, default=0.20)
    parser.add_argument("--consistency-tcp-threshold-mm", type=float, default=35.0)
    parser.add_argument("--consistency-action-l2-threshold-rad", type=float, default=0.08)
    parser.add_argument("--consistency-cosine-min", type=float, default=0.25)
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
        "cached_labels": 0,
        "fast_replans": 0,
        "heavy_replans": 0,
        "consistency_rejects": 0,
        "skipped_records": 0,
    }
    success_tol_m = args.success_tol_mm / 1000.0
    fast_planner_cfg = motion_plan_config_for_preset(args.planner_preset, tolerance_m=min(success_tol_m, 0.01))
    full_planner_cfg = motion_plan_config_for_preset("full_staged", tolerance_m=min(success_tol_m, 0.004))

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
            f"cached={summary['cached_labels']} "
            f"fast={summary['fast_replans']} "
            f"heavy={summary['heavy_replans']} "
            f"reject={summary['consistency_rejects']} "
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
            corr_a, q_after_a, result_a, profile_a = rollout_with_corrections(
                model,
                start_q_rad=q_home,
                target_xyz=point_a_xyz,
                episode_id=episode_id,
                segment="point_a",
                device=device,
                max_steps=args.max_steps,
                success_tol_m=success_tol_m,
                teacher_mode=args.teacher_mode,
                fast_planner_cfg=fast_planner_cfg,
                full_planner_cfg=full_planner_cfg,
                replan_interval=args.replan_interval,
                replan_tcp_threshold_m=args.replan_tcp_threshold_mm / 1000.0,
                replan_joint_threshold_rad=args.replan_joint_threshold_rad,
                min_record_error_m=args.min_record_error_mm / 1000.0,
                keep_every_n_success=args.keep_every_n_success,
                hard_case_full_recheck=args.hard_case_full_recheck,
                cached_correction_gain=args.cached_correction_gain,
                max_teacher_delta_rad=args.max_teacher_delta_rad,
                enable_teacher_consistency=args.enable_teacher_consistency,
                consistency_correction_gain=args.consistency_correction_gain,
                consistency_joint_threshold_rad=args.consistency_joint_threshold_rad,
                consistency_tcp_threshold_m=args.consistency_tcp_threshold_mm / 1000.0,
                consistency_action_l2_threshold_rad=args.consistency_action_l2_threshold_rad,
                consistency_cosine_min=args.consistency_cosine_min,
            )
            point_b_xyz = list(episode["point_b"]["xyz_m"])
            corr_b, _q_after_b, result_b, profile_b = rollout_with_corrections(
                model,
                start_q_rad=q_after_a,
                target_xyz=point_b_xyz,
                episode_id=episode_id,
                segment="point_b",
                device=device,
                max_steps=args.max_steps,
                success_tol_m=success_tol_m,
                teacher_mode=args.teacher_mode,
                fast_planner_cfg=fast_planner_cfg,
                full_planner_cfg=full_planner_cfg,
                replan_interval=args.replan_interval,
                replan_tcp_threshold_m=args.replan_tcp_threshold_mm / 1000.0,
                replan_joint_threshold_rad=args.replan_joint_threshold_rad,
                min_record_error_m=args.min_record_error_mm / 1000.0,
                keep_every_n_success=args.keep_every_n_success,
                hard_case_full_recheck=args.hard_case_full_recheck,
                cached_correction_gain=args.cached_correction_gain,
                max_teacher_delta_rad=args.max_teacher_delta_rad,
                enable_teacher_consistency=args.enable_teacher_consistency,
                consistency_correction_gain=args.consistency_correction_gain,
                consistency_joint_threshold_rad=args.consistency_joint_threshold_rad,
                consistency_tcp_threshold_m=args.consistency_tcp_threshold_mm / 1000.0,
                consistency_action_l2_threshold_rad=args.consistency_action_l2_threshold_rad,
                consistency_cosine_min=args.consistency_cosine_min,
            )
            for record in corr_a + corr_b:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")

            summary["episodes"] += 1
            summary["point_a_successes"] += int(bool(result_a["success"]))
            summary["point_b_successes"] += int(bool(result_b["success"]))
            summary["correction_records"] += len(corr_a) + len(corr_b)
            for key in ("cached_labels", "fast_replans", "heavy_replans", "consistency_rejects", "skipped_records"):
                summary[key] += int(profile_a.to_summary()[key]) + int(profile_b.to_summary()[key])
            f.flush()
            episode_profile = {
                "episode_id": episode_id,
                "point_a": result_a,
                "point_b": result_b,
            }
            print(f"episode_profile: {json.dumps(episode_profile, ensure_ascii=True)}", flush=True)
            _print_progress(summary["episodes"])

    print(json.dumps(summary, indent=2))
    print(f"saved_dagger_dataset: {output_path}")


if __name__ == "__main__":
    main()
