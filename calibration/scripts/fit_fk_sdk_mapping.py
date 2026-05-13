"""Fit a q_sdk -> q_dh calibration layer for offline NERO FK.

This script is intentionally offline-only: it reads manual SDK flange samples,
fits a joint mapping plus base-frame transform against ``nero.kinematics``, and
writes diagnostics.  It does not connect to or command the robot.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nero.kinematics import flange_position  # noqa: E402


@dataclass(frozen=True)
class Sample:
    sample_index: int
    q_sdk_rad: np.ndarray
    sdk_flange_xyz_m: np.ndarray


def load_samples(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        q = np.array(raw["q_rad"][:7], dtype=np.float64)
        xyz = np.array(raw["sdk_flange_pose_m_rad"][:3], dtype=np.float64)
        samples.append(Sample(int(raw.get("sample_index", line_no)), q, xyz))
    if len(samples) < 4:
        raise ValueError("Need at least 4 samples to fit and validate a mapping")
    return samples


def rotation_from_vector(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / theta
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


def rotation_to_vector(rot: np.ndarray) -> np.ndarray:
    cos_theta = float(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)
    scale = theta / (2.0 * math.sin(theta))
    return scale * np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])


def fit_rigid_transform(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    h = (src - src_centroid).T @ (dst - dst_centroid)
    u, _s, vt = np.linalg.svd(h)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0.0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    trans = dst_centroid - rot @ src_centroid
    return rot, trans


def fk_points(q_sdk: np.ndarray, signs: np.ndarray, offsets_rad: np.ndarray) -> np.ndarray:
    points = []
    for q in q_sdk:
        q_dh = q * signs + offsets_rad
        points.append(flange_position(q_dh.tolist(), clamp=False))
    return np.array(points, dtype=np.float64)


def predictions(q_sdk: np.ndarray, signs: np.ndarray, params: np.ndarray) -> np.ndarray:
    offsets = params[:7]
    rot = rotation_from_vector(params[7:10])
    trans = params[10:13]
    points = fk_points(q_sdk, signs, offsets)
    return (points @ rot.T) + trans


def residual_vector(q_sdk: np.ndarray, sdk_xyz: np.ndarray, signs: np.ndarray, params: np.ndarray) -> np.ndarray:
    return (predictions(q_sdk, signs, params) - sdk_xyz).reshape(-1)


def optimize_params(
    q_sdk: np.ndarray,
    sdk_xyz: np.ndarray,
    signs: np.ndarray,
    initial_offsets: np.ndarray,
    *,
    max_iter: int = 50,
) -> np.ndarray:
    initial_fk = fk_points(q_sdk, signs, initial_offsets)
    rot, trans = fit_rigid_transform(initial_fk, sdk_xyz)
    params = np.concatenate([initial_offsets, rotation_to_vector(rot), trans])

    damping = 1e-3
    eps = 1e-5
    for _ in range(max_iter):
        residual = residual_vector(q_sdk, sdk_xyz, signs, params)
        current_cost = float(residual @ residual)
        jac = np.zeros((residual.size, params.size), dtype=np.float64)
        for j in range(params.size):
            step = eps * max(1.0, abs(float(params[j])))
            p2 = params.copy()
            p2[j] += step
            jac[:, j] = (residual_vector(q_sdk, sdk_xyz, signs, p2) - residual) / step

        lhs = jac.T @ jac + damping * np.eye(params.size, dtype=np.float64)
        rhs = -(jac.T @ residual)
        try:
            delta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
            candidate = params + scale * delta
            # Keep joint offsets bounded; larger wraps are equivalent and make diagnostics harder to read.
            candidate[:7] = (candidate[:7] + math.pi) % (2.0 * math.pi) - math.pi
            candidate_cost = float(residual_vector(q_sdk, sdk_xyz, signs, candidate) @ residual_vector(q_sdk, sdk_xyz, signs, candidate))
            if candidate_cost < current_cost:
                params = candidate
                damping = max(damping * 0.5, 1e-8)
                accepted = True
                break
        if not accepted:
            damping = min(damping * 10.0, 1e8)
        if np.linalg.norm(delta) < 1e-8:
            break
    return params


def metrics(q_sdk: np.ndarray, sdk_xyz: np.ndarray, signs: np.ndarray, params: np.ndarray) -> dict:
    pred = predictions(q_sdk, signs, params)
    err_mm = (pred - sdk_xyz) * 1000.0
    norms = np.linalg.norm(err_mm, axis=1)
    return {
        "rms_mm": float(np.sqrt(np.mean(norms**2))),
        "mean_mm": float(np.mean(norms)),
        "max_mm": float(np.max(norms)),
        "per_sample_error_mm": err_mm.tolist(),
        "per_sample_error_norm_mm": norms.tolist(),
        "predicted_sdk_flange_xyz_m": pred.tolist(),
    }


def coarse_offset_candidates() -> list[np.ndarray]:
    base = [np.zeros(7, dtype=np.float64)]
    ninety = math.pi / 2.0
    # Common SDK-vs-DH zero hypotheses for proximal joints. Distal offsets are left at zero
    # because flange position is weakly observable for J5-J7 in this small dataset.
    for o1, o2, o3, o4 in product((-ninety, 0.0, ninety), repeat=4):
        arr = np.array([o1, o2, o3, o4, 0.0, 0.0, 0.0], dtype=np.float64)
        base.append(arr)
    return base


def rank_initial_candidates(
    q_sdk: np.ndarray,
    sdk_xyz: np.ndarray,
    sign_sets: Iterable[tuple[int, ...]],
) -> list[tuple[float, tuple[int, ...], np.ndarray]]:
    candidates = coarse_offset_candidates()
    ranked: list[tuple[float, tuple[int, ...], np.ndarray]] = []
    for sign_tuple in sign_sets:
        signs = np.array(sign_tuple, dtype=np.float64)
        for initial_offsets in candidates:
            initial_fk = fk_points(q_sdk, signs, initial_offsets)
            rot, trans = fit_rigid_transform(initial_fk, sdk_xyz)
            params = np.concatenate([initial_offsets, rotation_to_vector(rot), trans])
            ranked.append((metrics(q_sdk, sdk_xyz, signs, params)["rms_mm"], sign_tuple, initial_offsets))
    return sorted(ranked, key=lambda item: item[0])


def select_best_fit(
    q_sdk: np.ndarray,
    sdk_xyz: np.ndarray,
    sign_sets: Iterable[tuple[int, ...]],
    *,
    optimize_top_n: int = 24,
) -> tuple[np.ndarray, np.ndarray, dict]:
    best: tuple[float, np.ndarray, np.ndarray, dict] | None = None
    # The coarse rigid-transform score is cheap and good enough to reject most
    # sign/offset hypotheses before finite-difference LM optimization.
    for _initial_rms, sign_tuple, initial_offsets in rank_initial_candidates(q_sdk, sdk_xyz, sign_sets)[:optimize_top_n]:
        signs = np.array(sign_tuple, dtype=np.float64)
        params = optimize_params(q_sdk, sdk_xyz, signs, initial_offsets)
        m = metrics(q_sdk, sdk_xyz, signs, params)
        if best is None or m["rms_mm"] < best[0]:
            best = (m["rms_mm"], signs, params, m)
    if best is None:
        raise RuntimeError("No fit candidates were evaluated")
    _score, signs, params, m = best
    return signs, params, m


def all_sign_sets() -> list[tuple[int, ...]]:
    return list(product((-1, 1), repeat=7))


def leave_one_out(samples: list[Sample], *, optimize_top_n: int) -> list[dict]:
    rows = []
    for held_out in range(len(samples)):
        train = [s for i, s in enumerate(samples) if i != held_out]
        q_train = np.array([s.q_sdk_rad for s in train], dtype=np.float64)
        xyz_train = np.array([s.sdk_flange_xyz_m for s in train], dtype=np.float64)
        signs, params, train_metrics = select_best_fit(q_train, xyz_train, all_sign_sets(), optimize_top_n=optimize_top_n)
        q_test = np.array([samples[held_out].q_sdk_rad], dtype=np.float64)
        xyz_test = np.array([samples[held_out].sdk_flange_xyz_m], dtype=np.float64)
        test_metrics = metrics(q_test, xyz_test, signs, params)
        rows.append(
            {
                "held_out_sample_index": samples[held_out].sample_index,
                "train_rms_mm": train_metrics["rms_mm"],
                "test_error_norm_mm": test_metrics["per_sample_error_norm_mm"][0],
                "signs": signs.astype(int).tolist(),
                "offsets_deg": np.rad2deg(params[:7]).tolist(),
            }
        )
    return rows


def build_mapping_json(signs: np.ndarray, params: np.ndarray) -> dict:
    rot = rotation_from_vector(params[7:10])
    return {
        "model": "q_dh = signs * q_sdk + offsets_rad; sdk_xyz ~= R_base_from_dh * fk_dh_xyz + t_base_from_dh",
        "signs_j1_to_j7": signs.astype(int).tolist(),
        "offsets_rad_j1_to_j7": params[:7].tolist(),
        "offsets_deg_j1_to_j7": np.rad2deg(params[:7]).tolist(),
        "R_base_from_dh": rot.tolist(),
        "t_base_from_dh_m": params[10:13].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit q_sdk -> q_dh mapping against manual SDK flange samples.")
    parser.add_argument(
        "--samples",
        type=Path,
        default=REPO_ROOT / "calibration/results/fk_sdk_manual_alignment_2026-05-12/samples.jsonl",
        help="JSONL file containing manual FK/SDK flange samples",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--skip-loo", action="store_true", help="Skip leave-one-out validation")
    parser.add_argument("--optimize-top-n", type=int, default=24, help="Number of coarse candidates to optimize")
    parser.add_argument("--loo-optimize-top-n", type=int, default=12, help="Number of coarse candidates to optimize per LOO fold")
    args = parser.parse_args()

    samples = load_samples(args.samples)
    q_sdk = np.array([s.q_sdk_rad for s in samples], dtype=np.float64)
    sdk_xyz = np.array([s.sdk_flange_xyz_m for s in samples], dtype=np.float64)

    signs, params, train_metrics = select_best_fit(q_sdk, sdk_xyz, all_sign_sets(), optimize_top_n=args.optimize_top_n)
    loo = [] if args.skip_loo else leave_one_out(samples, optimize_top_n=args.loo_optimize_top_n)
    loo_norms = [row["test_error_norm_mm"] for row in loo]

    output = {
        "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_samples": str(args.samples),
        "sample_count": len(samples),
        "mapping": build_mapping_json(signs, params),
        "fit_metrics_all_samples": train_metrics,
        "leave_one_out": {
            "rows": loo,
            "rms_mm": float(np.sqrt(np.mean(np.square(loo_norms)))) if loo_norms else None,
            "mean_mm": float(np.mean(loo_norms)) if loo_norms else None,
            "max_mm": float(np.max(loo_norms)) if loo_norms else None,
        },
        "interpretation": {
            "warning": "Seven flange-position samples are enough for diagnosis, not enough to trust this mapping for real-robot collision checking.",
            "usable_threshold_mm": 30.0,
            "recommended_action": "Collect more diverse held-out flange samples before wiring this mapping into planner safety checks.",
        },
    }

    output_path = args.output or args.samples.parent / "q_sdk_to_dh_fit_result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "fit_rms_mm": train_metrics["rms_mm"], "loo_rms_mm": output["leave_one_out"]["rms_mm"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
