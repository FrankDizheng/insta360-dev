import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from handeye_board_runtime import (
    board_point_to_base,
    connect_robot,
    detect_board_pose,
    load_handeye,
    load_json,
    open_camera,
    pose_to_transform,
    project_board_point,
    read_fresh_frame,
    resolve_board_target,
    save_json,
)


def solve_tcp_offset(samples: list[dict]) -> tuple[np.ndarray, list[float]]:
    if len(samples) < 3:
        raise RuntimeError("Need at least 3 samples to solve TCP offset")

    A_rows = []
    b_rows = []
    for sample in samples:
        base_T_flange = np.array(sample["base_T_flange"], dtype=np.float64)
        target_base = np.array(sample["target_point_in_base_xyz_m"], dtype=np.float64)
        A_rows.append(base_T_flange[:3, :3])
        b_rows.append(target_base - base_T_flange[:3, 3])

    A = np.vstack(A_rows)
    b = np.concatenate(b_rows)
    tcp_xyz, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 3:
        raise RuntimeError("Sample poses are not diverse enough to solve TCP offset")

    residuals = []
    for sample in samples:
        base_T_flange = np.array(sample["base_T_flange"], dtype=np.float64)
        target_base = np.array(sample["target_point_in_base_xyz_m"], dtype=np.float64)
        pred = base_T_flange[:3, :3] @ tcp_xyz + base_T_flange[:3, 3]
        residuals.append(float(np.linalg.norm(pred - target_base)))
    return tcp_xyz, residuals


def load_existing_samples(samples_dir: Path) -> list[dict]:
    samples = []
    for path in sorted(samples_dir.glob("tcp_sample_*.json")):
        samples.append(load_json(path))
    return samples


def next_sample_index(samples_dir: Path) -> int:
    existing = sorted(samples_dir.glob("tcp_sample_*.json"))
    if not existing:
        return 1
    return max(int(path.stem.split("_")[-1]) for path in existing) + 1


def draw_sample_overlay(
    frame: np.ndarray,
    handeye: dict,
    detection: dict,
    target_board_xyz: list[float],
    label: str,
) -> np.ndarray:
    vis = frame.copy()
    cv2.aruco.drawDetectedMarkers(vis, detection["marker_corners"], detection["marker_ids"])
    cv2.aruco.drawDetectedCornersCharuco(vis, detection["charuco_corners"], detection["charuco_ids"])
    cv2.drawFrameAxes(
        vis,
        handeye["camera_matrix_np"],
        handeye["dist_coeffs_np"],
        detection["rvec"],
        detection["tvec"],
        0.05,
    )
    u, v = project_board_point(target_board_xyz, detection, handeye)
    cv2.circle(vis, (u, v), 12, (255, 0, 255), 2)
    cv2.putText(vis, label, (u + 12, v - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate gripper-center TCP from manual board-touch samples.")
    parser.add_argument("--handeye", required=True, help="Path to handeye_result.json")
    parser.add_argument("--output", default="", help="Output TCP JSON path")
    parser.add_argument("--samples-dir", default="", help="Directory for sample images and JSONs")
    parser.add_argument("--tool-name", default="gripper_center_closed", help="Name of the TCP reference point being calibrated")
    parser.add_argument("--camera-device", default="auto", help="V4L2 color camera device path or 'auto'")
    parser.add_argument("--robot-channel", default="can0", help="Robot CAN channel")
    parser.add_argument("--single", action="store_true", help="Capture one sample and exit")
    parser.add_argument("--solve", action="store_true", help="Solve TCP from existing samples and exit")
    parser.add_argument(
        "--target",
        choices=["origin", "center", "board_xy"],
        default="origin",
        help="Board target to touch for each sample",
    )
    parser.add_argument(
        "--board-xy",
        type=float,
        nargs=2,
        metavar=("X_M", "Y_M"),
        help="Board-frame X/Y coordinates in meters when --target=board_xy",
    )
    args = parser.parse_args()

    handeye_path = Path(args.handeye)
    handeye = load_handeye(handeye_path)
    target_board_xyz, target_name = resolve_board_target(handeye["board"], args.target, args.board_xy)

    output_path = Path(args.output) if args.output else handeye_path.parent / "gripper_tcp.json"
    samples_dir = Path(args.samples_dir) if args.samples_dir else handeye_path.parent / "tcp_calibration_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    if args.solve:
        samples = load_existing_samples(samples_dir)
        tcp_xyz, residuals = solve_tcp_offset(samples)
        result = {
            "tool_name": args.tool_name,
            "frame": "flange_T_tcp",
            "tcp_offset_xyzrpy_m_rad": [float(tcp_xyz[0]), float(tcp_xyz[1]), float(tcp_xyz[2]), 0.0, 0.0, 0.0],
            "sample_count": len(samples),
            "target_name": target_name,
            "target_point_in_board_xyz_m": target_board_xyz,
            "sample_files": [path.name for path in sorted(samples_dir.glob("tcp_sample_*.json"))],
            "residuals_m": residuals,
            "residual_rmse_m": float(np.sqrt(np.mean(np.square(residuals)))),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_json(output_path, result)
        print(f"saved_tcp_json: {output_path}")
        print("tcp_offset_xyzrpy_m_rad:", result["tcp_offset_xyzrpy_m_rad"])
        print("residual_rmse_m:", result["residual_rmse_m"])
        return

    print("Instructions:")
    print(f"1. Use the same physical contact point for all samples: {args.tool_name}.")
    print("2. Manually move that contact point onto the same board target.")
    print("3. Keep the board visible to the camera.")
    print("4. Capture at least 3 samples with noticeably different flange orientations.")
    print()
    print("Target point in board frame (m):", target_board_xyz)
    print(f"Samples will be saved to: {samples_dir}")
    print()

    robot = connect_robot(robot_channel=args.robot_channel)
    cap = open_camera(
        args.camera_device,
        int(handeye["camera_calibration"]["image_size"][0]),
        int(handeye["camera_calibration"]["image_size"][1]),
    )

    samples = []
    sample_idx = next_sample_index(samples_dir)
    try:
        if args.single:
            frame = read_fresh_frame(cap)
            detection = detect_board_pose(frame, handeye)
            flange_pose = np.array(robot.get_flange_pose().msg, dtype=np.float64)
            base_T_flange = pose_to_transform(flange_pose)
            base_T_camera = base_T_flange @ handeye["flange_T_camera_np"]
            base_T_board = base_T_camera @ detection["camera_T_board"]
            target_base_xyz = board_point_to_base(base_T_board, target_board_xyz)

            vis = draw_sample_overlay(frame, handeye, detection, target_board_xyz, f"tcp_sample_{sample_idx:03d}")
            image_name = f"tcp_sample_{sample_idx:03d}.png"
            json_name = f"tcp_sample_{sample_idx:03d}.json"
            image_path = samples_dir / image_name
            json_path = samples_dir / json_name
            cv2.imwrite(str(image_path), vis)

            sample = {
                "sample_index": sample_idx,
                "tool_name": args.tool_name,
                "target_name": target_name,
                "target_point_in_board_xyz_m": target_board_xyz,
                "target_point_in_base_xyz_m": target_base_xyz.tolist(),
                "base_T_flange": base_T_flange.tolist(),
                "flange_pose_xyzrpy_m_rad": flange_pose.tolist(),
                "detected_marker_count": int(len(detection["marker_ids"])),
                "detected_charuco_count": int(len(detection["charuco_ids"])),
                "annotated_image": image_name,
            }
            save_json(json_path, sample)
            print(f"Captured {json_name}")
            return

        while True:
            cmd = input("[Enter=capture, s=solve, q=quit] ").strip().lower()
            if cmd == "q":
                return
            if cmd == "s":
                break

            frame = read_fresh_frame(cap)
            detection = detect_board_pose(frame, handeye)
            flange_pose = np.array(robot.get_flange_pose().msg, dtype=np.float64)
            base_T_flange = pose_to_transform(flange_pose)
            base_T_camera = base_T_flange @ handeye["flange_T_camera_np"]
            base_T_board = base_T_camera @ detection["camera_T_board"]
            target_base_xyz = board_point_to_base(base_T_board, target_board_xyz)

            vis = draw_sample_overlay(frame, handeye, detection, target_board_xyz, f"tcp_sample_{sample_idx:03d}")
            image_name = f"tcp_sample_{sample_idx:03d}.png"
            json_name = f"tcp_sample_{sample_idx:03d}.json"
            image_path = samples_dir / image_name
            json_path = samples_dir / json_name
            cv2.imwrite(str(image_path), vis)

            sample = {
                "sample_index": sample_idx,
                "target_name": target_name,
                "target_point_in_board_xyz_m": target_board_xyz,
                "target_point_in_base_xyz_m": target_base_xyz.tolist(),
                "base_T_flange": base_T_flange.tolist(),
                "flange_pose_xyzrpy_m_rad": flange_pose.tolist(),
                "detected_marker_count": int(len(detection["marker_ids"])),
                "detected_charuco_count": int(len(detection["charuco_ids"])),
                "annotated_image": image_name,
            }
            save_json(json_path, sample)
            samples.append(sample)
            print(f"Captured {json_name}")
            sample_idx += 1
    finally:
        cap.release()

    tcp_xyz, residuals = solve_tcp_offset(samples)
    result = {
        "tool_name": args.tool_name,
        "frame": "flange_T_tcp",
        "tcp_offset_xyzrpy_m_rad": [float(tcp_xyz[0]), float(tcp_xyz[1]), float(tcp_xyz[2]), 0.0, 0.0, 0.0],
        "sample_count": len(samples),
        "target_name": target_name,
        "target_point_in_board_xyz_m": target_board_xyz,
        "sample_files": [f"tcp_sample_{i:03d}.json" for i in range(1, len(samples) + 1)],
        "residuals_m": residuals,
        "residual_rmse_m": float(np.sqrt(np.mean(np.square(residuals)))),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(output_path, result)

    print(f"saved_tcp_json: {output_path}")
    print("tcp_offset_xyzrpy_m_rad:", result["tcp_offset_xyzrpy_m_rad"])
    print("residual_rmse_m:", result["residual_rmse_m"])


if __name__ == "__main__":
    main()
