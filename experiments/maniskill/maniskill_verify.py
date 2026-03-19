"""Headless verification of pick-and-place — check IK accuracy + reward."""
import warnings
import numpy as np
import torch
import sapien, sapien.render

sapien.render.set_log_level("warning")
warnings.filterwarnings("ignore", message="Unknown attribute")

import mani_skill.envs
import gymnasium as gym
import pytorch_kinematics as pk

URDF_PATH = r"C:\Python313\Lib\site-packages\mani_skill\assets\robots\panda\panda_v2.urdf"
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
        q = q_init.clone()
        if trial > 0:
            q = q + torch.randn(7) * 0.15
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

# ── Test ───────────────────────────────────────────────────

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
qpos = robot.get_qpos()[0].numpy()
q_arm = torch.tensor(qpos[:7], dtype=torch.float32)

tcp_w = obs["extra"]["tcp_pose"][0, :3].numpy()
obj_w = obs["extra"]["obj_pose"][0, :3].numpy()
goal_w = obs["extra"]["goal_pos"][0].numpy()

print(f"Base:  {base_p}")
print(f"TCP:   {tcp_w}")
print(f"Cube:  {obj_w}")
print(f"Goal:  {goal_w}")

# Verify FK matches actual TCP
tf = chain.forward_kinematics(q_arm.unsqueeze(0))
fk_pos = tf.get_matrix()[0, :3, 3].numpy()
tcp_base = world_to_base(tcp_w, base_p)
print(f"\nFK TCP (base): {fk_pos}")
print(f"Actual TCP (base): {tcp_base}")
print(f"FK error: {np.linalg.norm(fk_pos - tcp_base):.6f}m")

# Solve IK for cube position
cube_z = obj_w[2]
grasp_z = cube_z + 0.005
above_z = grasp_z + 0.12
lift_z = grasp_z + 0.18

obj_b = world_to_base(obj_w, base_p)
goal_b = world_to_base(goal_w, base_p)

print(f"\nCube (base): {obj_b}")
print(f"Grasp Z: {grasp_z}  Above Z: {above_z}")

print("\n--- IK Solutions ---")
targets = [
    ("above cube", [obj_b[0], obj_b[1], above_z]),
    ("reach cube", [obj_b[0], obj_b[1], grasp_z]),
    ("lift",       [obj_b[0], obj_b[1], lift_z]),
    ("above goal", [goal_b[0], goal_b[1], lift_z]),
    ("at goal",    [goal_b[0], goal_b[1], above_z]),
    ("retreat",    [goal_b[0], goal_b[1], lift_z]),
]

q_current = q_arm.clone()
ik_solutions = {}
for name, pos in targets:
    tgt = torch.tensor(pos, dtype=torch.float32)
    q_sol, err = solve_ik(tgt, q_init=q_current)
    q_current = q_sol.clone()
    ik_solutions[name] = q_sol.numpy()
    # Verify FK
    tf_check = chain.forward_kinematics(q_sol.unsqueeze(0))
    fk_check = tf_check.get_matrix()[0, :3, 3].numpy()
    print(f"  {name:12s}  target={np.array(pos)}  FK={fk_check}  err={err:.4f}m")

# Now run the pick-and-place in simulation
print("\n--- Executing pick-and-place ---")
home_q = q_arm.numpy()
states = [
    (home_q,                      GRIPPER_OPEN,  "home",        90),
    (ik_solutions["above cube"],  GRIPPER_OPEN,  "above cube",  90),
    (ik_solutions["reach cube"],  GRIPPER_OPEN,  "reach cube",  90),
    (ik_solutions["reach cube"],  GRIPPER_CLOSE, "grasp",       60),
    (ik_solutions["lift"],        GRIPPER_CLOSE, "lift",        90),
    (ik_solutions["above goal"],  GRIPPER_CLOSE, "above goal",  90),
    (ik_solutions["at goal"],     GRIPPER_CLOSE, "at goal",     90),
    (ik_solutions["at goal"],     GRIPPER_OPEN,  "release",     60),
    (ik_solutions["retreat"],     GRIPPER_OPEN,  "retreat",     60),
]

step = 0
for phase in range(len(states)):
    idx_from = phase
    idx_to = min(phase + 1, len(states) - 1)
    arm_from, grip_from, name, duration = states[idx_from]
    arm_to, grip_to, _, _ = states[idx_to]

    for ps in range(duration):
        t = ps / duration
        t_s = 0.5 * (1.0 - np.cos(t * np.pi))
        arm = (1.0 - t_s) * arm_from + t_s * arm_to
        grip = grip_from + t_s * (grip_to - grip_from)
        action = np.append(arm, grip).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(torch.tensor(action).unsqueeze(0))
        step += 1

    tcp_now = obs["extra"]["tcp_pose"][0, :3].numpy()
    obj_now = obs["extra"]["obj_pose"][0, :3].numpy()
    dist = np.linalg.norm(tcp_now - obj_now)
    print(f"  Phase {phase} [{name:12s}]  tcp={tcp_now}  obj={obj_now}  dist={dist:.4f}  r={reward.item():.4f}")

print(f"\nTotal steps: {step}")
print(f"Final reward: {reward.item():.4f}")
if info.get("success", torch.tensor([False]))[0].item():
    print("SUCCESS!")
else:
    print("Not yet successful (cube may not be at goal).")

env.close()
