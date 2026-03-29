import json
import math
import time
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

DEFAULT_CAMERA_CANDIDATES = [
    "/dev/video6",
    "/dev/video2",
    "/dev/video4",
    "/dev/video0",
    "/dev/video1",
    "/dev/video3",
    "/dev/video5",
    "/dev/video7",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def matrix_to_xyzrpy(T: np.ndarray) -> list[float]:
    x, y, z = T[:3, 3].tolist()
    roll = math.atan2(T[2, 1], T[2, 2])
    pitch = math.atan2(-T[2, 0], math.sqrt(T[2, 1] ** 2 + T[2, 2] ** 2))
    yaw = math.atan2(T[1, 0], T[0, 0])
    return [x, y, z, roll, pitch, yaw]


def pose_to_transform(pose: list[float] | np.ndarray) -> np.ndarray:
    x, y, z, roll, pitch, yaw = np.asarray(pose, dtype=np.float64).tolist()
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(roll, pitch, yaw)
    T[:3, 3] = [x, y, z]
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
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

    robot.set_normal_mode()
    time.sleep(0.5)
    wait_for_robot_feedback(robot, timeout_s=4.0)


def connect_robot(robot_channel: str = "can0"):
    robot_cfg = create_agx_arm_config(
        robot="nero",
        comm="can",
        channel=robot_channel,
        interface="socketcan",
    )
    robot = AgxArmFactory.create_arm(robot_cfg)
    robot.connect()
    ensure_robot_feedback(robot)
    return robot


def wait_motion_done(robot, timeout_s: float = 12.0, poll_interval_s: float = 0.1) -> bool:
    time.sleep(0.5)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = robot.get_arm_status()
        if status is not None and getattr(status.msg, "motion_status", None) == 0:
            return True
        time.sleep(poll_interval_s)
    return False


def _readable_camera_summary(device: str) -> str:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return f"{device}: open=false"

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, frame = cap.read()
    shape = tuple(frame.shape) if ok and frame is not None else None
    fourcc_i = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_i >> (8 * i)) & 0xFF) for i in range(4)) if fourcc_i else "unknown"
    cap.release()
    return f"{device}: open=true read={ok} shape={shape} fourcc={fourcc}"


def detect_camera_device(preferred_device: str = "auto") -> str:
    candidates = []
    if preferred_device and preferred_device != "auto":
        candidates.append(preferred_device)
    candidates.extend([dev for dev in DEFAULT_CAMERA_CANDIDATES if dev not in candidates])

    for device in candidates:
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        ok_open = cap.isOpened()
        ok_read = False
        if ok_open:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok_read, frame = cap.read()
            ok_read = ok_read and frame is not None
        cap.release()
        if ok_open and ok_read:
            return device

    summaries = []
    for device in candidates:
        summaries.append(_readable_camera_summary(device))
    raise RuntimeError(
        "Failed to detect a readable color camera device. "
        f"Tried: {', '.join(candidates)}. "
        f"Probe results: {'; '.join(summaries)}"
    )


def open_camera(device: str, width: int, height: int, warmup_s: float = 1.0):
    selected_device = detect_camera_device(device)
    cap = cv2.VideoCapture(selected_device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open selected camera device {selected_device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if warmup_s > 0.0:
        time.sleep(warmup_s)
    return cap


def read_fresh_frame(
    cap,
    flush_count: int = 3,
    retry_count: int = 3,
    retry_sleep_s: float = 0.3,
):
    last_error = "unknown"
    for attempt in range(retry_count):
        frame = None
        ok = False
        for _ in range(flush_count):
            ok, candidate = cap.read()
            if ok and candidate is not None:
                frame = candidate
        if ok and frame is not None:
            return frame
        last_error = f"attempt={attempt + 1}/{retry_count} ok={ok} frame_is_none={frame is None}"
        if attempt + 1 < retry_count and retry_sleep_s > 0.0:
            time.sleep(retry_sleep_s)
    raise RuntimeError(f"Failed to read frame ({last_error})")


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


def load_handeye(handeye_path: str | Path) -> dict:
    handeye = load_json(Path(handeye_path))
    handeye["camera_matrix_np"] = np.array(handeye["camera_calibration"]["camera_matrix"], dtype=np.float64)
    handeye["dist_coeffs_np"] = np.array(handeye["camera_calibration"]["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    handeye["flange_T_camera_np"] = np.array(handeye["handeye"]["flange_T_camera"], dtype=np.float64)
    return handeye


def load_tcp_offset(tcp_path: str | Path) -> list[float]:
    data = load_json(Path(tcp_path))
    if "tcp_offset_xyzrpy_m_rad" not in data:
        raise RuntimeError(f"tcp_offset_xyzrpy_m_rad missing in {tcp_path}")
    return list(data["tcp_offset_xyzrpy_m_rad"])


def detect_board_pose(frame: np.ndarray, handeye: dict) -> dict:
    board_cfg = handeye["board"]
    aruco_dict, board = create_board(board_cfg)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    params = create_detector_params()
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    if marker_ids is None or len(marker_ids) == 0:
        raise RuntimeError("No board markers detected")

    ok, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        gray,
        board,
    )
    if not ok or charuco_ids is None or len(charuco_ids) < 8:
        raise RuntimeError("Failed to interpolate enough ChArUco corners")

    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners,
        charuco_ids,
        board,
        handeye["camera_matrix_np"],
        handeye["dist_coeffs_np"],
        None,
        None,
    )
    if not ok:
        raise RuntimeError("Failed to estimate board pose")

    return {
        "board": board,
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
        "rvec": rvec,
        "tvec": tvec,
        "camera_T_board": to_transform(rvec, tvec),
    }


def project_board_point(point_board_xyz: list[float], detection: dict, handeye: dict) -> tuple[int, int]:
    obj = np.array([point_board_xyz], dtype=np.float32)
    img_pts, _ = cv2.projectPoints(
        obj,
        detection["rvec"],
        detection["tvec"],
        handeye["camera_matrix_np"],
        handeye["dist_coeffs_np"],
    )
    u, v = img_pts.reshape(-1, 2)[0]
    return int(round(u)), int(round(v))


def board_point_to_base(base_T_board: np.ndarray, point_board_xyz: list[float]) -> np.ndarray:
    p = np.array([point_board_xyz[0], point_board_xyz[1], point_board_xyz[2], 1.0], dtype=np.float64)
    return (base_T_board @ p)[:3]


def resolve_board_target(board_cfg: dict, target_name: str, board_xy: list[float] | None) -> tuple[list[float], str]:
    if target_name == "origin":
        return [0.0, 0.0, 0.0], "board_origin"
    if target_name == "center":
        x = (int(board_cfg["squares_x"]) - 1) * float(board_cfg["square_length_m"]) * 0.5
        y = (int(board_cfg["squares_y"]) - 1) * float(board_cfg["square_length_m"]) * 0.5
        return [x, y, 0.0], "board_center"
    if target_name == "board_xy":
        if board_xy is None or len(board_xy) != 2:
            raise RuntimeError("--board-xy requires X Y in meters")
        return [float(board_xy[0]), float(board_xy[1]), 0.0], f"board_xy_{board_xy[0]}_{board_xy[1]}"
    raise RuntimeError(f"Unsupported target: {target_name}")


def capture_live_board_result(
    handeye_path: str | Path,
    camera_device: str = "/dev/video6",
    robot_channel: str = "can0",
) -> tuple[dict, np.ndarray, dict]:
    handeye = load_handeye(handeye_path)
    robot = connect_robot(robot_channel=robot_channel)
    cap = open_camera(
        camera_device,
        int(handeye["camera_calibration"]["image_size"][0]),
        int(handeye["camera_calibration"]["image_size"][1]),
    )
    try:
        frame = read_fresh_frame(cap)
    finally:
        cap.release()

    detection = detect_board_pose(frame, handeye)
    flange_pose = np.array(robot.get_flange_pose().msg, dtype=np.float64)
    base_T_flange = pose_to_transform(flange_pose)
    base_T_camera = base_T_flange @ handeye["flange_T_camera_np"]
    base_T_board = base_T_camera @ detection["camera_T_board"]
    result = {
        "handeye_path": str(Path(handeye_path)),
        "base_T_flange": base_T_flange,
        "base_T_camera": base_T_camera,
        "base_T_board": base_T_board,
        "flange_pose_xyzrpy_m_rad": flange_pose.tolist(),
        "board_origin_in_base_xyz_m": base_T_board[:3, 3].tolist(),
        "board_origin_in_base_xyzrpy_m_rad": matrix_to_xyzrpy(base_T_board),
    }
    return result, frame, detection


def capture_live_board_result_with_robot(
    handeye: dict,
    robot,
    camera_device: str = "/dev/video6",
) -> tuple[dict, np.ndarray, dict]:
    cap = open_camera(
        camera_device,
        int(handeye["camera_calibration"]["image_size"][0]),
        int(handeye["camera_calibration"]["image_size"][1]),
    )
    try:
        frame = read_fresh_frame(cap)
    finally:
        cap.release()

    detection = detect_board_pose(frame, handeye)
    flange_msg = robot.get_flange_pose()
    if flange_msg is None:
        raise RuntimeError("Flange pose unavailable while capturing live board result")
    flange_pose = np.array(flange_msg.msg, dtype=np.float64)
    base_T_flange = pose_to_transform(flange_pose)
    base_T_camera = base_T_flange @ handeye["flange_T_camera_np"]
    base_T_board = base_T_camera @ detection["camera_T_board"]
    result = {
        "base_T_flange": base_T_flange,
        "base_T_camera": base_T_camera,
        "base_T_board": base_T_board,
        "flange_pose_xyzrpy_m_rad": flange_pose.tolist(),
        "board_origin_in_base_xyz_m": base_T_board[:3, 3].tolist(),
        "board_origin_in_base_xyzrpy_m_rad": matrix_to_xyzrpy(base_T_board),
    }
    return result, frame, detection
