"""VLM pick-and-place demo — generates a video (GIF) of the full process."""
import os, sys, time, json, re, warnings
import numpy as np
import torch
import sapien, sapien.render
import transforms3d

sapien.render.set_log_level("warning")
warnings.filterwarnings("ignore", message="Unknown attribute")

import mani_skill.envs
import gymnasium as gym
import pytorch_kinematics as pk
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bridge"))
from model_feed import analyze_image

URDF_PATH = r"C:\Python313\Lib\site-packages\mani_skill\assets\robots\panda\panda_v2.urdf"
GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0
JOINT_LOWER = torch.tensor([-2.8, -1.7, -2.8, -3.0, -2.8, -0.01, -2.8])
JOINT_UPPER = torch.tensor([ 2.8,  1.7,  2.8, -0.07,  2.8,  3.75,  2.8])
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures", "vlm_video")

SYSTEM_PROMPT = """You are a robot controller. You see a simulation with a Panda robot arm, a red cube on a table, and a green sphere marking the goal.

Your task: pick up the red cube and move it to the green sphere.

Output EXACTLY ONE JSON action. Valid actions:
{"action": "move_above", "target": "cube"}
{"action": "move_above", "target": "goal"}
{"action": "lower", "target": "cube"}
{"action": "lower", "target": "goal"}
{"action": "grasp"}
{"action": "lift"}
{"action": "release"}
{"action": "done"}

Rules:
- Output ONLY the JSON.
- Correct sequence: move_above cube -> lower cube -> grasp -> lift -> move_above goal -> lower goal -> release -> done
- If the gripper is clearly above the cube and open, the next step should be lower.
- If the gripper is at the cube level, grasp it.
- If cube is grasped, lift then move to goal.
"""

chain = pk.build_serial_chain_from_urdf(open(URDF_PATH).read(), "panda_hand_tcp")

def solve_ik(target, q_init, n_tries=10):
    best_q, best_err = q_init.clone(), float("inf")
    for trial in range(n_tries):
        q = q_init.clone() + (torch.randn(7) * 0.15 if trial > 0 else 0)
        q = torch.clamp(q, JOINT_LOWER, JOINT_UPPER)
        for _ in range(150):
            tf = chain.forward_kinematics(q.unsqueeze(0))
            pos = tf.get_matrix()[0, :3, 3]
            err = target - pos
            d = err.norm().item()
            if d < best_err:
                best_err = d; best_q = q.clone()
            if d < 0.001:
                return best_q, d
            J = chain.jacobian(q.unsqueeze(0))[0, :3, :]
            dq = J.T @ torch.linalg.solve(J @ J.T + 0.005 * torch.eye(3), err)
            q = torch.clamp(q + 0.3 * dq, JOINT_LOWER, JOINT_UPPER)
    return best_q, best_err

def w2b(pos, base):
    return pos - base

def setup_camera(sub_scene):
    ent = sapien.Entity()
    cam = sapien.render.RenderCameraComponent(640, 480)
    cam.set_fovy(1.0); cam.set_near(0.01); cam.set_far(100)
    ent.add_component(cam)
    fwd = -np.array([0.8, -0.6, 0.6]); fwd /= np.linalg.norm(fwd)
    up = np.array([0, 0, 1])
    r = np.cross(fwd, up); r /= np.linalg.norm(r)
    up = np.cross(r, fwd)
    q = transforms3d.quaternions.mat2quat(np.stack([fwd, -r, up], axis=1))
    ent.set_pose(sapien.Pose(p=[0.8, -0.6, 0.6], q=q))
    sub_scene.add_entity(ent)
    return cam

def snap(sub_scene, cam):
    sub_scene.update_render()
    cam.take_picture()
    return cam.get_picture("Color")[:, :, :3]

def add_label(img_array, text, step=None):
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        font_sm = ImageFont.truetype("arial.ttf", 16)
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_sm = font
    draw.rectangle([(0, 0), (640, 32)], fill=(0, 0, 0, 180))
    label = f"Step {step}: " if step is not None else ""
    draw.text((8, 6), label + text, fill=(255, 255, 255), font=font)
    draw.text((8, 460), "Qwen3.5 VLM -> Robot Control", fill=(200, 200, 200), font=font_sm)
    return np.array(img)

def execute_motion(env, q_from, q_to, g_from, g_to, steps, sub_scene, cam, frames, label, step_num, every_n=6):
    for i in range(steps):
        t = 0.5 * (1.0 - np.cos(i / steps * np.pi))
        arm = (1.0 - t) * q_from + t * q_to
        grip = g_from + t * (g_to - g_from)
        action = np.append(arm, grip).astype(np.float32)
        env.step(torch.tensor(action).unsqueeze(0))
        if i % every_n == 0:
            rgb = snap(sub_scene, cam)
            frames.append(add_label(rgb, label, step_num))

def parse_action(text):
    m = re.search(r'\{[^}]+\}', text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {"action": "done"}

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []

    print("Creating environment...", flush=True)
    env = gym.make("PickCube-v1", obs_mode="state_dict", render_mode=None,
                   num_envs=1, sim_backend="cpu", control_mode="pd_joint_pos",
                   max_episode_steps=999999)
    obs, info = env.reset()

    robot = env.unwrapped.agent.robot
    base = robot.pose.raw_pose[0, :3].numpy()
    q_arm = torch.tensor(robot.get_qpos()[0].numpy()[:7], dtype=torch.float32)
    grip = GRIPPER_OPEN

    scene = env.unwrapped.scene
    sub = scene.sub_scenes[0]
    cam = setup_camera(sub)

    # Initial frames
    for _ in range(10):
        rgb = snap(sub, cam)
        frames.append(add_label(rgb, "Scene ready - starting VLM control"))

    MAX_STEPS = 12
    action_history = []
    print("Starting VLM loop...\n", flush=True)

    for step in range(MAX_STEPS):
        rgb = snap(sub, cam)
        frames.append(add_label(rgb, "Thinking...", step))
        tmp = os.path.join(OUT_DIR, f"_tmp.png")
        Image.fromarray(rgb).save(tmp)

        history_text = ""
        if action_history:
            history_text = "\n\nActions already completed:\n"
            for i, a in enumerate(action_history):
                history_text += f"  Step {i}: {a}\n"
            history_text += f"\nNow decide step {step}. Do NOT repeat or skip steps."

        prompt = SYSTEM_PROMPT + history_text + f"\nStep {step}. What is the next action?"
        print(f"Step {step}: querying VLM...", end=" ", flush=True)
        t0 = time.time()
        resp = analyze_image(tmp, prompt, stream=False)
        dt = time.time() - t0
        act = parse_action(resp)
        act_name = act.get("action", "?")
        act_tgt = act.get("target", "")
        print(f"{act_name} {act_tgt} ({dt:.1f}s)", flush=True)

        action_history.append(f"{act_name} {act_tgt}".strip())

        if act_name == "done":
            for _ in range(15):
                frames.append(add_label(rgb, "DONE - Task complete!", step))
            break

        # Get fresh obs
        obs = env.step(torch.tensor(np.append(q_arm.numpy(), grip).astype(np.float32)).unsqueeze(0))[0]
        obj_w = obs["extra"]["obj_pose"][0, :3].numpy()
        goal_w = obs["extra"]["goal_pos"][0].numpy()
        cz = obj_w[2]; gz = cz + 0.005; az = gz + 0.12; lz = gz + 0.18
        obj_b = w2b(obj_w, base); goal_b = w2b(goal_w, base)

        label = f"{act_name} {act_tgt}"

        if act_name == "move_above":
            p = obj_b if act_tgt == "cube" else goal_b
            tgt = torch.tensor([p[0], p[1], az], dtype=torch.float32)
            q_new, _ = solve_ik(tgt, q_arm)
            execute_motion(env, q_arm.numpy(), q_new.numpy(), grip, grip, 90, sub, cam, frames, label, step)
            q_arm = q_new

        elif act_name == "lower":
            p = obj_b if act_tgt == "cube" else goal_b
            z = max(goal_w[2], gz) if act_tgt == "goal" else gz
            tgt = torch.tensor([p[0], p[1], z], dtype=torch.float32)
            q_new, _ = solve_ik(tgt, q_arm)
            execute_motion(env, q_arm.numpy(), q_new.numpy(), grip, grip, 90, sub, cam, frames, label, step)
            q_arm = q_new

        elif act_name == "grasp":
            execute_motion(env, q_arm.numpy(), q_arm.numpy(), GRIPPER_OPEN, GRIPPER_CLOSE, 60, sub, cam, frames, label, step)
            grip = GRIPPER_CLOSE

        elif act_name == "lift":
            p = obj_b
            tgt = torch.tensor([p[0], p[1], lz], dtype=torch.float32)
            q_new, _ = solve_ik(tgt, q_arm)
            execute_motion(env, q_arm.numpy(), q_new.numpy(), grip, grip, 90, sub, cam, frames, label, step)
            q_arm = q_new

        elif act_name == "release":
            execute_motion(env, q_arm.numpy(), q_arm.numpy(), GRIPPER_CLOSE, GRIPPER_OPEN, 60, sub, cam, frames, label, step)
            grip = GRIPPER_OPEN

    env.close()

    # Save GIF
    gif_path = os.path.join(OUT_DIR, "vlm_pick_and_place.gif")
    print(f"\nSaving GIF ({len(frames)} frames)...", flush=True)
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                       duration=80, loop=0)
    print(f"Saved to {gif_path}", flush=True)

    # Also save key frames as PNG
    key_indices = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]
    for i, idx in enumerate(key_indices):
        Image.fromarray(frames[idx]).save(os.path.join(OUT_DIR, f"key_{i}.png"))

    print("Done!", flush=True)

if __name__ == "__main__":
    main()
