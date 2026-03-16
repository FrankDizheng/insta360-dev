"""VLM-driven pick-and-place: Qwen sees the simulation, decides actions, robot executes.

Architecture:
  Sim camera frame -> Qwen VLM -> high-level action -> IK + execute -> next frame
"""
import os, sys, time, json, re, warnings, tempfile
import numpy as np
import torch
import cv2
import sapien, sapien.render
import transforms3d

sapien.render.set_log_level("warning")
warnings.filterwarnings("ignore", message="Unknown attribute")

import mani_skill.envs
import gymnasium as gym
import pytorch_kinematics as pk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bridge"))
from model_feed import analyze_image, check_ollama

# ── Constants ──────────────────────────────────────────────

URDF_PATH = r"C:\Python313\Lib\site-packages\mani_skill\assets\robots\panda\panda_v2.urdf"
GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0
JOINT_LOWER = torch.tensor([-2.8, -1.7, -2.8, -3.0, -2.8, -0.01, -2.8])
JOINT_UPPER = torch.tensor([ 2.8,  1.7,  2.8, -0.07,  2.8,  3.75,  2.8])
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures", "vlm_run")

SYSTEM_PROMPT = """You are a robot controller. You see images from a simulation camera showing a Panda robot arm, a red cube on a table, and a green sphere marking the goal position.

Your task: pick up the red cube and move it to the green sphere's position.

Based on the current image, output EXACTLY ONE JSON action. Valid actions:

{"action": "move_above", "target": "cube"}     - move gripper above the red cube
{"action": "move_above", "target": "goal"}     - move gripper above the green goal
{"action": "lower", "target": "cube"}          - lower gripper to grasp the cube
{"action": "lower", "target": "goal"}          - lower gripper to place at the goal
{"action": "grasp"}                            - close the gripper
{"action": "lift"}                             - lift the gripper up
{"action": "release"}                          - open the gripper
{"action": "done"}                             - task is complete

Rules:
- Output ONLY the JSON, no other text.
- Choose the SINGLE best next action based on what you see.
- Typical sequence: move_above cube -> lower cube -> grasp -> lift -> move_above goal -> lower goal -> release -> done
- If the cube is already in the gripper, skip to moving toward goal.
- If the cube is already at the goal, output done.
"""

# ── IK ─────────────────────────────────────────────────────

chain = pk.build_serial_chain_from_urdf(open(URDF_PATH).read(), "panda_hand_tcp")

def solve_ik(target_pos_base, q_init, n_tries=10):
    best_q, best_err = q_init.clone(), float("inf")
    for trial in range(n_tries):
        q = q_init.clone() + (torch.randn(7) * 0.15 if trial > 0 else 0)
        q = torch.clamp(q, JOINT_LOWER, JOINT_UPPER)
        for _ in range(150):
            tf = chain.forward_kinematics(q.unsqueeze(0))
            pos = tf.get_matrix()[0, :3, 3]
            err = target_pos_base - pos
            dist = err.norm().item()
            if dist < best_err:
                best_err = dist
                best_q = q.clone()
            if dist < 0.001:
                return best_q, dist
            J = chain.jacobian(q.unsqueeze(0))[0, :3, :]
            JJT = J @ J.T + 0.005 * torch.eye(3)
            dq = J.T @ torch.linalg.solve(JJT, err)
            q = q + 0.3 * dq
            q = torch.clamp(q, JOINT_LOWER, JOINT_UPPER)
    return best_q, best_err

def world_to_base(pos, base_offset):
    return pos - base_offset

# ── Capture ────────────────────────────────────────────────

def setup_camera(sub_scene):
    cam_entity = sapien.Entity()
    cam = sapien.render.RenderCameraComponent(640, 480)
    cam.set_fovy(1.0)
    cam.set_near(0.01)
    cam.set_far(100)
    cam_entity.add_component(cam)

    forward = np.array([0, 0, 0]) - np.array([0.8, -0.6, 0.6])
    forward /= np.linalg.norm(forward)
    up = np.array([0, 0, 1])
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rot_mat = np.stack([forward, -right, up], axis=1)
    quat = transforms3d.quaternions.mat2quat(rot_mat)
    cam_entity.set_pose(sapien.Pose(p=[0.8, -0.6, 0.6], q=quat))
    sub_scene.add_entity(cam_entity)
    return cam

def capture_frame(sub_scene, cam, save_path):
    sub_scene.update_render()
    cam.take_picture()
    rgba = cam.get_picture("Color")
    rgb = rgba[:, :, :3]
    from PIL import Image
    Image.fromarray(rgb).save(save_path)
    return save_path

# ── Action Execution ───────────────────────────────────────

def execute_motion(env, q_from, q_to, grip_from, grip_to, steps=90):
    """Smoothly interpolate from one joint config to another."""
    for i in range(steps):
        t = 0.5 * (1.0 - np.cos(i / steps * np.pi))
        arm = (1.0 - t) * q_from + t * q_to
        grip = grip_from + t * (grip_to - grip_from)
        action = np.append(arm, grip).astype(np.float32)
        env.step(torch.tensor(action).unsqueeze(0))

def execute_action(env, action_dict, obs, base_offset, q_current, gripper_state):
    """Execute a high-level VLM action. Returns (new_q, new_gripper_state)."""
    act = action_dict.get("action", "")
    target = action_dict.get("target", "")

    obj_w = obs["extra"]["obj_pose"][0, :3].numpy()
    goal_w = obs["extra"]["goal_pos"][0].numpy()
    cube_z = obj_w[2]
    grasp_z = cube_z + 0.005
    above_z = grasp_z + 0.12
    lift_z = grasp_z + 0.18

    if act == "move_above":
        pos_w = obj_w if target == "cube" else goal_w
        pos_b = world_to_base(pos_w, base_offset)
        tgt = torch.tensor([pos_b[0], pos_b[1], above_z], dtype=torch.float32)
        q_new, err = solve_ik(tgt, q_current)
        print(f"    IK err={err:.4f}m", flush=True)
        execute_motion(env, q_current.numpy(), q_new.numpy(), gripper_state, gripper_state)
        return q_new, gripper_state

    elif act == "lower":
        pos_w = obj_w if target == "cube" else goal_w
        pos_b = world_to_base(pos_w, base_offset)
        goal_z = max(goal_w[2], grasp_z) if target == "goal" else grasp_z
        tgt = torch.tensor([pos_b[0], pos_b[1], goal_z], dtype=torch.float32)
        q_new, err = solve_ik(tgt, q_current)
        print(f"    IK err={err:.4f}m", flush=True)
        execute_motion(env, q_current.numpy(), q_new.numpy(), gripper_state, gripper_state)
        return q_new, gripper_state

    elif act == "grasp":
        execute_motion(env, q_current.numpy(), q_current.numpy(),
                       GRIPPER_OPEN, GRIPPER_CLOSE, steps=60)
        return q_current, GRIPPER_CLOSE

    elif act == "lift":
        pos_b = world_to_base(obj_w, base_offset)
        tgt = torch.tensor([pos_b[0], pos_b[1], lift_z], dtype=torch.float32)
        q_new, err = solve_ik(tgt, q_current)
        execute_motion(env, q_current.numpy(), q_new.numpy(), gripper_state, gripper_state)
        return q_new, gripper_state

    elif act == "release":
        execute_motion(env, q_current.numpy(), q_current.numpy(),
                       GRIPPER_CLOSE, GRIPPER_OPEN, steps=60)
        return q_current, GRIPPER_OPEN

    elif act == "done":
        print("    Task complete!", flush=True)
        return q_current, gripper_state

    else:
        print(f"    Unknown action: {act}", flush=True)
        return q_current, gripper_state

# ── VLM Query ──────────────────────────────────────────────

def parse_vlm_action(response_text):
    """Extract JSON action from VLM response (handles messy output)."""
    match = re.search(r'\{[^}]+\}', response_text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"action": "done"}

def query_vlm(image_path, step_num):
    """Ask VLM what action to take next."""
    prompt = SYSTEM_PROMPT + f"\n\nThis is step {step_num}. What is the next action?"
    print(f"  Querying VLM...", end=" ", flush=True)
    t0 = time.time()
    response = analyze_image(image_path, prompt, stream=False)
    dt = time.time() - t0
    print(f"({dt:.1f}s)", flush=True)
    print(f"  VLM says: {response.strip()}", flush=True)
    return parse_vlm_action(response)

# ── Main ───────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Checking Ollama...", flush=True)
    if not check_ollama():
        print("ERROR: Ollama not running or model not found!", flush=True)
        return

    print("Creating PickCube-v1...", flush=True)
    env = gym.make(
        "PickCube-v1",
        obs_mode="state_dict",
        render_mode=None,
        num_envs=1,
        sim_backend="cpu",
        control_mode="pd_joint_pos",
        max_episode_steps=999999,
    )
    obs, info = env.reset()

    robot = env.unwrapped.agent.robot
    base_p = robot.pose.raw_pose[0, :3].numpy()
    q_arm = torch.tensor(robot.get_qpos()[0].numpy()[:7], dtype=torch.float32)
    gripper = GRIPPER_OPEN

    scene = env.unwrapped.scene
    sub_scene = scene.sub_scenes[0]
    cam = setup_camera(sub_scene)

    tcp_w = obs["extra"]["tcp_pose"][0, :3].numpy()
    obj_w = obs["extra"]["obj_pose"][0, :3].numpy()
    goal_w = obs["extra"]["goal_pos"][0].numpy()
    print(f"TCP:  {tcp_w}", flush=True)
    print(f"Cube: {obj_w}", flush=True)
    print(f"Goal: {goal_w}", flush=True)

    MAX_STEPS = 12
    print(f"\n{'='*60}", flush=True)
    print("VLM CLOSED-LOOP CONTROL", flush=True)
    print(f"{'='*60}\n", flush=True)

    for step in range(MAX_STEPS):
        frame_path = os.path.join(OUT_DIR, f"step_{step:02d}.png")
        capture_frame(sub_scene, cam, frame_path)

        print(f"Step {step}:", flush=True)
        action_dict = query_vlm(frame_path, step)
        act_name = action_dict.get("action", "unknown")
        act_target = action_dict.get("target", "")
        print(f"  Action: {act_name} {act_target}", flush=True)

        if act_name == "done":
            print("\nVLM says task is done!", flush=True)
            break

        obs_fresh, _, _, _, _ = env.step(
            torch.tensor(np.append(q_arm.numpy(), gripper).astype(np.float32)).unsqueeze(0)
        )

        q_arm, gripper = execute_action(
            env, action_dict, obs_fresh, base_p, q_arm, gripper
        )

        obs_after = env.step(
            torch.tensor(np.append(q_arm.numpy(), gripper).astype(np.float32)).unsqueeze(0)
        )[0]
        tcp = obs_after["extra"]["tcp_pose"][0, :3].numpy()
        obj = obs_after["extra"]["obj_pose"][0, :3].numpy()
        dist = np.linalg.norm(tcp - obj)
        print(f"  Result: tcp-obj={dist:.4f}m\n", flush=True)

        capture_frame(sub_scene, cam, os.path.join(OUT_DIR, f"step_{step:02d}_after.png"))

    final_path = os.path.join(OUT_DIR, "final.png")
    capture_frame(sub_scene, cam, final_path)
    print(f"\nFinal frame saved to {final_path}", flush=True)

    env.close()
    print("Done!", flush=True)

if __name__ == "__main__":
    main()
