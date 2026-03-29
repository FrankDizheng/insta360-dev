import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from handeye_board_runtime import (
    board_point_to_base,
    capture_live_board_result,
    load_handeye,
    project_board_point,
    resolve_board_target,
    save_json,
)


def draw_target_overlay(
    frame: np.ndarray,
    handeye: dict,
    detection: dict,
    target_board_xyz: list[float],
    target_name: str,
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
    cv2.circle(vis, (u, v), 12, (0, 0, 255), 2)
    cv2.drawMarker(vis, (u, v), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
    cv2.putText(
        vis,
        target_name,
        (u + 12, v - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(description="Locate a target point on the ChArUco board in base coordinates.")
    parser.add_argument("--handeye", required=True, help="Path to handeye_result.json")
    parser.add_argument("--output-dir", default="", help="Directory for annotated image and JSON output")
    parser.add_argument("--camera-device", default="auto", help="V4L2 color camera device path or 'auto'")
    parser.add_argument("--robot-channel", default="can0", help="Robot CAN channel")
    parser.add_argument(
        "--target",
        choices=["origin", "center", "board_xy"],
        default="origin",
        help="Named board target to locate",
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
    output_dir = Path(args.output_dir) if args.output_dir else handeye_path.parent / "locate_target"
    output_dir.mkdir(parents=True, exist_ok=True)

    live_result, frame, detection = capture_live_board_result(
        handeye_path=handeye_path,
        camera_device=args.camera_device,
        robot_channel=args.robot_channel,
    )
    target_board_xyz, target_name = resolve_board_target(handeye["board"], args.target, args.board_xy)
    target_base_xyz = board_point_to_base(live_result["base_T_board"], target_board_xyz)
    board_normal_in_base = live_result["base_T_board"][:3, 2]

    vis = draw_target_overlay(frame, handeye, detection, target_board_xyz, target_name)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = output_dir / f"locate_target_{stamp}.png"
    json_path = output_dir / f"locate_target_{stamp}.json"
    cv2.imwrite(str(image_path), vis)

    result = {
        "timestamp": stamp,
        "camera_device": args.camera_device,
        "target_name": target_name,
        "target_point_in_board_xyz_m": target_board_xyz,
        "target_point_in_base_xyz_m": target_base_xyz.tolist(),
        "board_normal_in_base_xyz": board_normal_in_base.tolist(),
        "detected_marker_count": int(len(detection["marker_ids"])),
        "detected_charuco_count": int(len(detection["charuco_ids"])),
        "base_T_board": live_result["base_T_board"].tolist(),
        "base_T_flange": live_result["base_T_flange"].tolist(),
        "annotated_image": image_path.name,
    }
    save_json(json_path, result)

    print(f"saved_locate_image: {image_path}")
    print(f"saved_locate_json: {json_path}")
    print("target_point_in_base_xyz_m:", result["target_point_in_base_xyz_m"])
    print("board_normal_in_base_xyz:", result["board_normal_in_base_xyz"])


if __name__ == "__main__":
    main()
