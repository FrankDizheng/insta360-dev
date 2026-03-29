import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pyAgxArm import AgxArmFactory, create_agx_arm_config


DICT_NAME_TO_ID = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def pose_to_transform(pose: np.ndarray) -> np.ndarray:
    x, y, z, roll, pitch, yaw = pose.tolist()
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(roll, pitch, yaw)
    T[:3, 3] = [x, y, z]
    return T


def wait_for_robot_feedback(robot, timeout_s: float = 4.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if robot.get_flange_pose() is not None and robot.get_joint_angles() is not None:
            return
        time.sleep(0.05)
    raise RuntimeError("Robot feedback timeout")


def ensure_robot_feedback(robot) -> None:
    try:
        wait_for_robot_feedback(robot, timeout_s=2.0)
        return
    except RuntimeError:
        pass

    print("No robot feedback yet, switching to normal mode...")
    robot.set_normal_mode()
    time.sleep(0.5)
    wait_for_robot_feedback(robot, timeout_s=4.0)


def open_camera(device: str, width: int, height: int):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def read_fresh_frame(cap, flush_count: int = 5):
    frame = None
    ok = False
    for _ in range(flush_count):
        ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to read frame")
    return frame


def create_board(board_cfg: dict):
    dict_id = DICT_NAME_TO_ID[board_cfg["dictionary"]]
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard_create(
        int(board_cfg["squares_x"]),
        int(board_cfg["squares_y"]),
        float(board_cfg["square_length_m"]),
        float(board_cfg["marker_length_m"]),
        aruco_dict,
    )
    return aruco_dict, board


def create_detector_params():
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()
    return cv2.aruco.DetectorParameters()


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def matrix_to_xyzrpy(T: np.ndarray) -> list[float]:
    x, y, z = T[:3, 3].tolist()
    roll = math.atan2(T[2, 1], T[2, 2])
    pitch = math.atan2(-T[2, 0], math.sqrt(T[2, 1] ** 2 + T[2, 2] ** 2))
    yaw = math.atan2(T[1, 0], T[0, 0])
    return [x, y, z, roll, pitch, yaw]


def save_debug_frame(output_dir: Path, frame: np.ndarray, prefix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{prefix}_{stamp}.png"
    cv2.imwrite(str(path), frame)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate eye-in-hand hand-eye calibration with one live frame.")
    parser.add_argument("--handeye", required=True, help="Path to handeye_result.json")
    parser.add_argument("--output-dir", default="", help="Directory to save validation artifacts")
    parser.add_argument("--camera-device", default="auto", help="V4L2 color camera device path or 'auto'")
    parser.add_argument("--robot-channel", default="can0", help="Robot CAN channel")
    args = parser.parse_args()

    handeye_path = Path(args.handeye)
    handeye = load_json(handeye_path)
    board_cfg = handeye["board"]
    camera_cfg = handeye["camera_calibration"]
    output_dir = Path(args.output_dir) if args.output_dir else handeye_path.parent / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_matrix = np.array(camera_cfg["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(camera_cfg["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    flange_T_camera = np.array(handeye["handeye"]["flange_T_camera"], dtype=np.float64)

    print("Connecting robot...")
    robot_cfg = create_agx_arm_config(
        robot="nero",
        comm="can",
        channel=args.robot_channel,
        interface="socketcan",
    )
    robot = AgxArmFactory.create_arm(robot_cfg)
    robot.connect()
    ensure_robot_feedback(robot)

    print("Opening camera...")
    cap = open_camera(args.camera_device, camera_cfg["image_size"][0], camera_cfg["image_size"][1])
    frame = read_fresh_frame(cap)
    cap.release()

    aruco_dict, board = create_board(board_cfg)
    params = create_detector_params()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    if marker_ids is None or len(marker_ids) == 0:
        debug_path = save_debug_frame(output_dir, frame, "validate_no_markers")
        raise RuntimeError(f"No board markers detected in live frame; saved {debug_path}")

    ok, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        gray,
        board,
    )
    if not ok or charuco_ids is None or len(charuco_ids) < 8:
        debug_path = save_debug_frame(output_dir, frame, "validate_low_charuco")
        raise RuntimeError(f"Failed to interpolate enough ChArUco corners; saved {debug_path}")

    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners,
        charuco_ids,
        board,
        camera_matrix,
        dist_coeffs,
        None,
        None,
    )
    if not ok:
        debug_path = save_debug_frame(output_dir, frame, "validate_pose_failed")
        raise RuntimeError(f"Failed to estimate board pose from live frame; saved {debug_path}")

    flange_pose = np.array(robot.get_flange_pose().msg, dtype=np.float64)
    base_T_flange = pose_to_transform(flange_pose)

    camera_T_board = to_transform(rvec, tvec)
    base_T_camera = base_T_flange @ flange_T_camera
    base_T_board = base_T_camera @ camera_T_board
    board_T_base = invert_transform(base_T_board)

    vis = frame.copy()
    cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
    cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
    cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = output_dir / f"validate_{stamp}.png"
    json_path = output_dir / f"validate_{stamp}.json"
    cv2.imwrite(str(image_path), vis)

    result = {
        "timestamp": stamp,
        "camera_device": args.camera_device,
        "detected_marker_count": int(len(marker_ids)),
        "detected_charuco_count": int(len(charuco_ids)),
        "base_T_flange": base_T_flange.tolist(),
        "flange_T_camera": flange_T_camera.tolist(),
        "camera_T_board": camera_T_board.tolist(),
        "base_T_board": base_T_board.tolist(),
        "board_T_base": board_T_base.tolist(),
        "board_origin_in_base_xyz_m": base_T_board[:3, 3].tolist(),
        "board_origin_in_base_xyzrpy_m_rad": matrix_to_xyzrpy(base_T_board),
        "annotated_image": image_path.name,
    }
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"saved_validation_image: {image_path}")
    print(f"saved_validation_json: {json_path}")
    print("board_origin_in_base_xyz_m:", result["board_origin_in_base_xyz_m"])


if __name__ == "__main__":
    main()
