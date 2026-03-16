"""Run pick-and-place headless and capture snapshots at each phase via SAPIEN camera."""
import os, warnings
import numpy as np
import torch
import sapien, sapien.render
import transforms3d

sapien.render.set_log_level("warning")
warnings.filterwarnings("ignore", message="Unknown attribute")

import mani_skill.envs
import gymnasium as gym
import pytorch_kinematics as pk
from PIL import Image

URDF_PATH = r"C:\Python313\Lib\site-packages\mani_skill\assets\robots\panda\panda_v2.urdf"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0
JOINT_LOWER = torch.tensor([-2.8, -1.7, -2.8, -3.0, -2.8, -0.01, -2.8])
JOINT_UPPER = torch.tensor([ 2.8,  1.7,  2.8, -0.07,  2.8,  3.75,  2.8])

chain = pk.build_serial_chain_from_urdf(open(URDF_PATH).read(), "panda_hand_tcp")

def solve_ik(target_pos_base, q_init=None, n_tries=10):
    if q_init is None:
        q_init = torch.zeros(7)
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

def world_to_base(world_pos, base_offset):
    return world_pos - base_offset

env = gym.make("PickCube-v1", obs_mode="state_dict", render_mode=None,
               num_envs=1, sim_backend="cpu", control_mode="pd_joint_pos",
               max_episode_steps=999999)
obs, info = env.reset()

robot = env.unwrapped.agent.robot
base_p = robot.pose.raw_pose[0, :3].numpy()
q_arm = torch.tensor(robot.get_qpos()[0].numpy()[:7], dtype=torch.float32)
obj_w = obs["extra"]["obj_pose"][0, :3].numpy()
goal_w = obs["extra"]["goal_pos"][0].numpy()

obj_b = world_to_base(obj_w, base_p)
goal_b = world_to_base(goal_w, base_p)
cube_z = obj_w[2]
grasp_z = cube_z + 0.005
above_z = grasp_z + 0.12
lift_z = grasp_z + 0.18
goal_z = max(goal_w[2], grasp_z)

q_current = q_arm.clone()
solved = {}
for name, pos in [("above cube", [obj_b[0], obj_b[1], above_z]),
                   ("reach cube", [obj_b[0], obj_b[1], grasp_z]),
                   ("lift",       [obj_b[0], obj_b[1], lift_z]),
                   ("above goal", [goal_b[0], goal_b[1], lift_z]),
                   ("at goal",    [goal_b[0], goal_b[1], goal_z]),
                   ("retreat",    [goal_b[0], goal_b[1], lift_z])]:
    tgt = torch.tensor(pos, dtype=torch.float32)
    q_sol, _ = solve_ik(tgt, q_init=q_current)
    q_current = q_sol.clone()
    solved[name] = q_sol.numpy()

states = [
    (q_arm.numpy(),         GRIPPER_OPEN,  "0_home",       90),
    (solved["above cube"],  GRIPPER_OPEN,  "1_above_cube", 90),
    (solved["reach cube"],  GRIPPER_OPEN,  "2_reach_cube", 90),
    (solved["reach cube"],  GRIPPER_CLOSE, "3_grasp",      60),
    (solved["lift"],        GRIPPER_CLOSE, "4_lift",       90),
    (solved["above goal"],  GRIPPER_CLOSE, "5_above_goal", 90),
    (solved["at goal"],     GRIPPER_CLOSE, "6_at_goal",    90),
    (solved["at goal"],     GRIPPER_OPEN,  "7_release",    60),
    (solved["retreat"],     GRIPPER_OPEN,  "8_retreat",    60),
]

# Add camera
scene = env.unwrapped.scene
sub_scene = scene.sub_scenes[0]
cam_entity = sapien.Entity()
cam = sapien.render.RenderCameraComponent(800, 600)
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

print("Executing pick-and-place with snapshots...", flush=True)
step = 0
for phase in range(len(states)):
    idx_to = min(phase + 1, len(states) - 1)
    arm_from, grip_from, name, duration = states[phase]
    arm_to, grip_to, _, _ = states[idx_to]
    for ps in range(duration):
        t = ps / duration
        t_s = 0.5 * (1.0 - np.cos(t * np.pi))
        arm = (1.0 - t_s) * arm_from + t_s * arm_to
        grip = grip_from + t_s * (grip_to - grip_from)
        action = np.append(arm, grip).astype(np.float32)
        obs, reward, _, _, info = env.step(torch.tensor(action).unsqueeze(0))
        step += 1

    sub_scene.update_render()
    cam.take_picture()
    rgba = cam.get_picture("Color")
    rgb = rgba[:, :, :3]
    out = os.path.join(OUT_DIR, f"pick_{name}.png")
    Image.fromarray(rgb).save(out)
    tcp = obs["extra"]["tcp_pose"][0, :3].numpy()
    obj = obs["extra"]["obj_pose"][0, :3].numpy()
    print(f"  {name}: tcp-obj={np.linalg.norm(tcp-obj):.4f}m  r={reward.item():.3f}  -> {out}", flush=True)

env.close()
print(f"Done! {step} steps, snapshots in {OUT_DIR}", flush=True)
