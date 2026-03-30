"""
Depth-fused board localization + safe 10mm standoff test (v2).

Uses position polling for motion completion instead of motion_status.
"""

import numpy as np
import cv2
import time
import sys
sys.path.insert(0, "/home/pi")

from pyorbbecsdk import *
from handeye_board_runtime import (
    load_handeye, connect_robot, pose_to_transform, matrix_to_xyzrpy,
    create_board, create_detector_params, to_transform, load_tcp_offset,
)


def wait_move_done(robot, target_xyz, tol_mm=0.5, timeout_s=20.0):
    """Wait until flange reaches target position (position-based check)."""
    time.sleep(0.3)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        fp = np.array(robot.get_flange_pose().msg[:3], dtype=np.float64)
        err = np.linalg.norm(fp - np.array(target_xyz[:3])) * 1000
        if err < tol_mm:
            time.sleep(0.3)
            return True
        time.sleep(0.1)
    return False


def capture_aligned_rgbd():
    ctx = Context()
    dev = ctx.query_devices()[0]
    pipe = Pipeline(dev)
    config = Config()
    cp = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
    dp = pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
    config.enable_stream(cp)
    config.enable_stream(dp)
    align_filter = AlignFilter(OBStreamType.COLOR_STREAM)
    pipe.start(config)
    time.sleep(2.0)
    ci = pipe.get_camera_param().rgb_intrinsic
    intrinsics = {"fx": ci.fx, "fy": ci.fy, "cx": ci.cx, "cy": ci.cy}
    for _ in range(15):
        pipe.wait_for_frames(1000)
    frameset = pipe.wait_for_frames(1000)
    aligned = align_filter.process(frameset)
    cf = aligned.get_color_frame()
    df = aligned.get_depth_frame()
    color = cv2.imdecode(np.frombuffer(cf.get_data(), np.uint8), cv2.IMREAD_COLOR)
    depth_raw = np.frombuffer(df.get_data(), np.uint16).reshape(df.get_height(), df.get_width())
    scale = df.get_depth_scale()
    pipe.stop()
    return color, depth_raw, scale, intrinsics


def detect_board_depth_fused(color, depth_raw, depth_scale, intrinsics, board_cfg):
    cm = np.array([[intrinsics["fx"],0,intrinsics["cx"]],
                   [0,intrinsics["fy"],intrinsics["cy"]],
                   [0,0,1]], dtype=np.float64)
    dc = np.zeros((5,1), dtype=np.float64)
    aruco_dict, board = create_board(board_cfg)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    params = create_detector_params()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    if ids is None or len(ids) < 4:
        raise RuntimeError(f"Too few markers: {0 if ids is None else len(ids)}")
    ok, cc, cids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
    if not ok or len(cids) < 8:
        raise RuntimeError("Not enough charuco corners")
    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(cc, cids, board, cm, dc, None, None)
    if not ok:
        raise RuntimeError("PnP failed")

    camera_T_board = to_transform(rvec, tvec)
    pnp_z = float(tvec.flatten()[2]) * 1000
    h, w = depth_raw.shape

    diffs = []
    for i, corner in enumerate(cc):
        u, v = int(round(corner[0,0])), int(round(corner[0,1]))
        if u < 4 or u >= w-4 or v < 4 or v >= h-4:
            continue
        roi = depth_raw[v-3:v+4, u-3:u+4]
        valid = roi[roi > 0]
        if len(valid) < 3:
            continue
        d_mm = float(np.median(valid)) * depth_scale
        cid = int(cids.flatten()[i])
        row = cid // (board_cfg["squares_x"] - 1)
        col = cid % (board_cfg["squares_x"] - 1)
        bx = (col+1) * board_cfg["square_length_m"]
        by = (row+1) * board_cfg["square_length_m"]
        p_cam = (camera_T_board @ np.array([bx, by, 0, 1.0]))[:3]
        diffs.append(d_mm - float(p_cam[2])*1000)

    if len(diffs) >= 3:
        z_corr = np.median(diffs) / 1000.0
        camera_T_board_fused = camera_T_board.copy()
        camera_T_board_fused[2, 3] += z_corr
        print(f"  PnP Z={pnp_z:.1f}mm, depth correction={z_corr*1000:.1f}mm, fused Z={pnp_z+z_corr*1000:.1f}mm ({len(diffs)} corners)")
        return camera_T_board_fused, True
    else:
        print(f"  PnP Z={pnp_z:.1f}mm, no depth fusion (only {len(diffs)} corners)")
        return camera_T_board, False


def main():
    STANDOFF_MM = 10.0
    SPEED = 10

    handeye = load_handeye("/home/pi/handeye_dataset/session1/handeye_result.json")
    tcp_offset = load_tcp_offset("/home/pi/handeye_dataset/session1/gripper_tcp_left_front_tip_samples_004_006.json")

    robot = connect_robot()
    robot.set_tcp_offset(tcp_offset)
    robot.set_speed_percent(SPEED)
    robot.enable()
    time.sleep(0.3)

    fp0 = np.array(robot.get_flange_pose().msg, dtype=np.float64)
    tp0 = np.array(robot.get_tcp_pose().msg, dtype=np.float64)
    base_T_flange0 = pose_to_transform(fp0)
    base_T_camera0 = base_T_flange0 @ handeye["flange_T_camera_np"]
    print(f"[init] Flange Z={fp0[2]*1000:.1f}mm, TCP Z={tp0[2]*1000:.1f}mm, Camera Z={base_T_camera0[2,3]*1000:.1f}mm")

    # 1. capture + detect
    print("\n[1] Capturing RGBD + detecting board...")
    color, depth, scale, intr = capture_aligned_rgbd()
    camera_T_board, is_fused = detect_board_depth_fused(color, depth, scale, intr, handeye["board"])
    print(f"  Depth fused: {is_fused}")

    # 2. compute board pose in base
    base_T_board = base_T_camera0 @ camera_T_board
    board_origin = base_T_board[:3, 3]
    board_normal = base_T_board[:3, 2].copy()
    board_normal /= np.linalg.norm(board_normal)
    if board_normal[2] < 0:
        board_normal = -board_normal

    # target: board center
    bcfg = handeye["board"]
    tx = (bcfg["squares_x"]-1) * bcfg["square_length_m"] * 0.5
    ty = (bcfg["squares_y"]-1) * bcfg["square_length_m"] * 0.5
    target_base = (base_T_board @ np.array([tx, ty, 0, 1]))[:3]

    print(f"\n[2] Board origin in base: [{board_origin[0]*1000:.1f}, {board_origin[1]*1000:.1f}, {board_origin[2]*1000:.1f}] mm")
    print(f"  Board normal: [{board_normal[0]:.3f}, {board_normal[1]:.3f}, {board_normal[2]:.3f}]")
    print(f"  Target (center): [{target_base[0]*1000:.1f}, {target_base[1]*1000:.1f}, {target_base[2]*1000:.1f}] mm")

    # 3. compute standoff: 10mm above target along normal
    standoff_xyz = target_base + board_normal * (STANDOFF_MM / 1000.0)
    print(f"  Standoff: [{standoff_xyz[0]*1000:.1f}, {standoff_xyz[1]*1000:.1f}, {standoff_xyz[2]*1000:.1f}] mm")

    # keep current orientation
    standoff_tcp = [float(standoff_xyz[0]), float(standoff_xyz[1]), float(standoff_xyz[2]),
                    float(tp0[3]), float(tp0[4]), float(tp0[5])]

    # convert to flange coordinates
    flange_target = robot.get_tcp2flange_pose(standoff_tcp)
    print(f"  Flange target: [{flange_target[0]*1000:.1f}, {flange_target[1]*1000:.1f}, {flange_target[2]*1000:.1f}] mm")

    # 4. first move to 50mm above (intermediate safe point)
    safe50_xyz = target_base + board_normal * (50.0 / 1000.0)
    safe50_tcp = [float(safe50_xyz[0]), float(safe50_xyz[1]), float(safe50_xyz[2]),
                  float(tp0[3]), float(tp0[4]), float(tp0[5])]
    safe50_flange = robot.get_tcp2flange_pose(safe50_tcp)

    print(f"\n[3] Moving to 50mm intermediate (move_p)...")
    robot.enable()
    robot.move_p(safe50_flange)
    ok = wait_move_done(robot, safe50_flange, tol_mm=1.0, timeout_s=15.0)
    fp1 = np.array(robot.get_flange_pose().msg, dtype=np.float64)
    tp1 = np.array(robot.get_tcp_pose().msg, dtype=np.float64)
    tcp1_dist_above = float(np.dot(tp1[:3] - target_base, board_normal)) * 1000
    print(f"  Done={ok}, TCP Z={tp1[2]*1000:.1f}mm, height above board={tcp1_dist_above:.1f}mm")

    if tcp1_dist_above < 20:
        print("[ABORT] Too close at intermediate! Stopping.")
        return

    # 5. move_l to 10mm standoff
    print(f"\n[4] Moving to {STANDOFF_MM}mm standoff (move_l)...")
    robot.move_l(flange_target)
    ok2 = wait_move_done(robot, flange_target, tol_mm=1.0, timeout_s=15.0)
    fp2 = np.array(robot.get_flange_pose().msg, dtype=np.float64)
    tp2 = np.array(robot.get_tcp_pose().msg, dtype=np.float64)
    tcp2_dist_above = float(np.dot(tp2[:3] - target_base, board_normal)) * 1000
    print(f"  Done={ok2}, TCP Z={tp2[2]*1000:.1f}mm, height above board={tcp2_dist_above:.1f}mm")
    print(f"  Expected standoff: {STANDOFF_MM:.1f}mm, actual: {tcp2_dist_above:.1f}mm, error: {tcp2_dist_above - STANDOFF_MM:.1f}mm")

    if abs(tcp2_dist_above - STANDOFF_MM) < 3.0:
        print(f"\n[SUCCESS] Depth-fused localization accurate to {abs(tcp2_dist_above - STANDOFF_MM):.1f}mm")
    else:
        print(f"\n[CHECK] Standoff error is {abs(tcp2_dist_above - STANDOFF_MM):.1f}mm")

    # 6. lift back to safe
    print("\n[5] Lifting back to safe height...")
    robot.move_p(list(fp0))
    time.sleep(5)
    fp3 = list(robot.get_flange_pose().msg)
    print(f"  Back to Z={fp3[2]*1000:.1f}mm")
    print("\n[done]")


if __name__ == "__main__":
    main()
