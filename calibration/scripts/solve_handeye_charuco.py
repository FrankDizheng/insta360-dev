import argparse
import json
from pathlib import Path

import cv2
import numpy as np


BOARD_SQUARES_X = 12
BOARD_SQUARES_Y = 9
SQUARE_LENGTH_M = 0.015
MARKER_LENGTH_M = 0.01125


DICT_CANDIDATES = [
    ("DICT_4X4_50", cv2.aruco.DICT_4X4_50),
    ("DICT_4X4_100", cv2.aruco.DICT_4X4_100),
    ("DICT_5X5_50", cv2.aruco.DICT_5X5_50),
    ("DICT_5X5_100", cv2.aruco.DICT_5X5_100),
    ("DICT_6X6_50", cv2.aruco.DICT_6X6_50),
    ("DICT_6X6_100", cv2.aruco.DICT_6X6_100),
    ("DICT_6X6_250", cv2.aruco.DICT_6X6_250),
    ("DICT_7X7_50", cv2.aruco.DICT_7X7_50),
    ("DICT_7X7_100", cv2.aruco.DICT_7X7_100),
]


def load_sample(json_path: Path) -> dict:
    return json.loads(json_path.read_text(encoding="utf-8"))


def create_board(dictionary_id: int):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard_create(
        BOARD_SQUARES_X,
        BOARD_SQUARES_Y,
        SQUARE_LENGTH_M,
        MARKER_LENGTH_M,
        aruco_dict,
    )
    return aruco_dict, board


def create_detector_params():
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()
    return cv2.aruco.DetectorParameters()


def board_dictionary_score(image_paths: list[Path], dictionary_id: int) -> tuple[int, int]:
    aruco_dict, board = create_board(dictionary_id)
    detected_images = 0
    total_charuco = 0
    detector_params = create_detector_params()

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)
        if marker_ids is None or len(marker_ids) == 0:
            continue
        ok, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board,
        )
        if ok and charuco_ids is not None and len(charuco_ids) >= 8:
            detected_images += 1
            total_charuco += len(charuco_ids)
    return detected_images, total_charuco


def pick_dictionary(image_paths: list[Path]) -> tuple[str, int]:
    best_name = ""
    best_id = -1
    best_score = (-1, -1)
    for name, dictionary_id in DICT_CANDIDATES:
        score = board_dictionary_score(image_paths, dictionary_id)
        if score > best_score:
            best_name = name
            best_id = dictionary_id
            best_score = score
    if best_id < 0 or best_score[0] <= 0:
        raise RuntimeError("Failed to detect ChArUco board with candidate dictionaries")
    print(f"selected_dictionary: {best_name} detected_images={best_score[0]} total_charuco={best_score[1]}")
    return best_name, best_id


def detect_charuco(image: np.ndarray, aruco_dict, board):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector_params = create_detector_params()
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)
    if marker_ids is None or len(marker_ids) == 0:
        return None
    ok, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        gray,
        board,
    )
    if not ok or charuco_ids is None or len(charuco_ids) < 8:
        return None
    return gray, marker_corners, marker_ids, charuco_corners, charuco_ids


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve eye-in-hand hand-eye calibration from ChArUco dataset.")
    parser.add_argument("--dataset-dir", required=True, help="Directory with sample_XXX.png/json pairs.")
    parser.add_argument("--output", default="", help="Output JSON path. Defaults to dataset_dir/handeye_result.json")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    json_files = sorted(dataset_dir.glob("sample_*.json"))
    if not json_files:
        raise RuntimeError("No sample_*.json files found")

    image_paths = [dataset_dir / load_sample(j)["image_path"] for j in json_files]
    dict_name, dict_id = "DICT_5X5_100", cv2.aruco.DICT_5X5_100
    print(f"selected_dictionary: {dict_name}")
    aruco_dict, board = create_board(dict_id)

    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None
    valid_samples = []

    for json_path in json_files:
        sample = load_sample(json_path)
        image_path = dataset_dir / sample["image_path"]
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image_size = (image.shape[1], image.shape[0])
        det = detect_charuco(image, aruco_dict, board)
        if det is None:
            print(f"skip_detection_failed: {json_path.name}")
            continue
        gray, marker_corners, marker_ids, charuco_corners, charuco_ids = det
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        valid_samples.append(
            {
                "json_path": json_path,
                "sample": sample,
                "gray": gray,
                "marker_corners": marker_corners,
                "marker_ids": marker_ids,
                "charuco_corners": charuco_corners,
                "charuco_ids": charuco_ids,
            }
        )

    if len(valid_samples) < 6:
        raise RuntimeError(f"Not enough valid ChArUco detections: {len(valid_samples)}")

    reproj_error, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners,
        all_charuco_ids,
        board,
        image_size,
        None,
        None,
    )
    print(f"camera_calibration_reproj_error: {reproj_error}")

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []
    used_names = []

    for item in valid_samples:
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            item["charuco_corners"],
            item["charuco_ids"],
            board,
            camera_matrix,
            dist_coeffs,
            None,
            None,
        )
        if not ok:
            print(f"skip_pose_failed: {item['json_path'].name}")
            continue

        base_T_flange = np.array(item["sample"]["robot"]["base_T_flange"], dtype=np.float64)
        R_gripper2base.append(base_T_flange[:3, :3])
        t_gripper2base.append(base_T_flange[:3, 3].reshape(3, 1))

        R_tc, _ = cv2.Rodrigues(rvec)
        R_target2cam.append(R_tc)
        t_target2cam.append(tvec.reshape(3, 1))
        used_names.append(item["json_path"].name)

    if len(R_gripper2base) < 6:
        raise RuntimeError(f"Not enough valid board poses for hand-eye: {len(R_gripper2base)}")

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )

    flange_T_camera = np.eye(4)
    flange_T_camera[:3, :3] = R_cam2gripper
    flange_T_camera[:3, 3] = t_cam2gripper.reshape(3)
    camera_T_flange = invert_transform(flange_T_camera)

    result = {
        "board": {
            "type": "charuco",
            "squares_x": BOARD_SQUARES_X,
            "squares_y": BOARD_SQUARES_Y,
            "square_length_m": SQUARE_LENGTH_M,
            "marker_length_m": MARKER_LENGTH_M,
            "dictionary": dict_name,
        },
        "dataset_dir": str(dataset_dir),
        "sample_count_total": len(json_files),
        "sample_count_valid_detection": len(valid_samples),
        "sample_count_used_handeye": len(used_names),
        "used_samples": used_names,
        "camera_calibration": {
            "image_size": list(image_size),
            "reprojection_error": float(reproj_error),
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        },
        "handeye": {
            "frame": "flange_T_camera",
            "flange_T_camera": flange_T_camera.tolist(),
            "camera_T_flange": camera_T_flange.tolist(),
            "translation_m": flange_T_camera[:3, 3].tolist(),
        },
    }

    output_path = Path(args.output) if args.output else dataset_dir / "handeye_result.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved_result: {output_path}")
    print("flange_T_camera_translation_m:", result["handeye"]["translation_m"])


if __name__ == "__main__":
    main()
