"""ManiSkill3 PickCube — pick-and-place with IK solver.

Uses pytorch_kinematics for inverse kinematics to make the
Panda robot smoothly reach, grasp, lift, and place the cube.
"""
import time, warnings
import numpy as np
import torch
import sapien, sapien.render

sapien.render.set_log_level("warning")
warnings.filterwarnings("ignore", message="Unknown attribute")

import mani_skill.envs
import gymnasium as gym
import pytorch_kinematics as pk

URDF_PATH = r"C:\Python313\Lib\site-packages\mani_skill\assets\robots\panda\panda_v2.urdf"
FPS = 30
FRAME_DT = 1.0 / FPS
GRIPPER_OPEN = 1.0    # normalized: 1 = fully open
GRIPPER_CLOSE = -1.0  # normalized: -1 = fully closed

JOINT_LOWER = torch.tensor([-2.8, -1.7, -2.8, -3.0, -2.8, -0.01, -2.8])
JOINT_UPPER = torch.tensor([ 2.8,  1.7,  2.8, -0.07,  2.8,  3.75,  2.8])

# ── IK ─────────────────────────────────────────────────────

chain = pk.build_serial_chain_from_urdf(open(URDF_PATH).read(), "panda_hand_tcp")

def solve_ik(target_pos_base, q_init=None, n_tries=10):
    """Solve IK for a 3D target position in robot-base frame."""
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
    """Convert world-frame position to robot-base frame."""
    return world_pos - base_offset


def plan_waypoints(obj_world, goal_world, base_offset, q_init):
    """Compute IK waypoints for a full pick-and-place cycle.

    Returns list of (arm_joints, gripper_value, name, duration) tuples.
    The main loop interpolates between consecutive pairs.
    Includes a "home" state at the start so the robot smoothly moves
    from its current position to the first target.
    """
    cube_z = obj_world[2]
    grasp_z = cube_z + 0.005
    above_z = grasp_z + 0.12
    lift_z  = grasp_z + 0.18

    obj_b = world_to_base(obj_world, base_offset)
    goal_b = world_to_base(goal_world, base_offset)

    goal_z = max(goal_world[2], grasp_z)

    ik_targets = [
        ("above cube",  [obj_b[0], obj_b[1], above_z]),
        ("reach cube",  [obj_b[0], obj_b[1], grasp_z]),
        ("lift cube",   [obj_b[0], obj_b[1], lift_z]),
        ("above goal",  [goal_b[0], goal_b[1], lift_z]),
        ("at goal",     [goal_b[0], goal_b[1], goal_z]),
        ("retreat",     [goal_b[0], goal_b[1], lift_z]),
    ]

    q_current = q_init.clone()
    solved = {}
    print("  Solving IK...", flush=True)
    for name, pos in ik_targets:
        tgt = torch.tensor(pos, dtype=torch.float32)
        q_sol, err = solve_ik(tgt, q_init=q_current)
        q_current = q_sol.clone()
        solved[name] = q_sol.numpy()
        print(f"    {name}: err={err:.4f}m", flush=True)

    states = [
        (q_init.numpy(),        GRIPPER_OPEN,  "home",        90),
        (solved["above cube"],  GRIPPER_OPEN,  "above cube",  90),
        (solved["reach cube"],  GRIPPER_OPEN,  "reach cube",  90),
        (solved["reach cube"],  GRIPPER_CLOSE, "grasp",       60),
        (solved["lift cube"],   GRIPPER_CLOSE, "lift cube",   90),
        (solved["above goal"],  GRIPPER_CLOSE, "above goal",  90),
        (solved["at goal"],     GRIPPER_CLOSE, "at goal",     90),
        (solved["at goal"],     GRIPPER_OPEN,  "release",     60),
        (solved["retreat"],     GRIPPER_OPEN,  "retreat",     60),
    ]
    return states


# ── Environment ────────────────────────────────────────────

print("Creating PickCube-v1...", flush=True)
env = gym.make(
    "PickCube-v1",
    obs_mode="state_dict",
    render_mode="human",
    num_envs=1,
    sim_backend="cpu",
    control_mode="pd_joint_pos",
    max_episode_steps=999999,
)
obs, info = env.reset()

robot = env.unwrapped.agent.robot
base_p = robot.pose.raw_pose[0, :3].numpy()
print(f"Robot base position: {base_p}", flush=True)

qpos_full = robot.get_qpos()[0].numpy()
q_arm = torch.tensor(qpos_full[:7], dtype=torch.float32)

tcp_world = obs["extra"]["tcp_pose"][0, :3].numpy()
obj_world = obs["extra"]["obj_pose"][0, :3].numpy()
goal_world = obs["extra"]["goal_pos"][0].numpy()

print(f"TCP  (world): {tcp_world}", flush=True)
print(f"Cube (world): {obj_world}", flush=True)
print(f"Goal (world): {goal_world}", flush=True)
print(f"Cube (base):  {world_to_base(obj_world, base_p)}", flush=True)

states = plan_waypoints(obj_world, goal_world, base_p, q_arm)

print(f"\n{len(states)} states ready. Starting demo...", flush=True)
print("Close the viewer window or Ctrl+C to stop.\n", flush=True)

# ── Main loop ──────────────────────────────────────────────

step = 0
phase = 0
phase_step = 0

try:
    while True:
        t0 = time.perf_counter()

        idx_from = phase % len(states)
        idx_to   = (phase + 1) % len(states)
        arm_from, grip_from, name_from, duration = states[idx_from]
        arm_to,   grip_to,   _,         _        = states[idx_to]

        t = min(phase_step / duration, 1.0)
        t_smooth = 0.5 * (1.0 - np.cos(t * np.pi))

        target_arm = (1.0 - t_smooth) * arm_from + t_smooth * arm_to
        target_grip = grip_from + t_smooth * (grip_to - grip_from)

        action = np.append(target_arm, target_grip).astype(np.float32)
        action_tensor = torch.tensor(action).unsqueeze(0)

        obs, reward, terminated, truncated, info = env.step(action_tensor)
        env.render()

        phase_step += 1
        step += 1

        if phase_step >= duration:
            phase_step = 0
            phase += 1
            if phase >= len(states):
                phase = 0
                obs, info = env.reset()
                q_arm = torch.tensor(robot.get_qpos()[0].numpy()[:7])
                obj_world = obs["extra"]["obj_pose"][0, :3].numpy()
                goal_world = obs["extra"]["goal_pos"][0].numpy()
                states = plan_waypoints(obj_world, goal_world, base_p, q_arm)
                print(f"\n  New episode at step {step}", flush=True)
            else:
                cur_name = states[phase % len(states)][2]
                tcp_now = obs["extra"]["tcp_pose"][0, :3].numpy()
                obj_now = obs["extra"]["obj_pose"][0, :3].numpy()
                dist = np.linalg.norm(tcp_now - obj_now)
                print(f"  [{phase}/{len(states)}] {cur_name}  tcp-obj={dist:.3f}m  r={reward.item():.3f}", flush=True)

        elapsed = time.perf_counter() - t0
        remaining = FRAME_DT - elapsed
        if remaining > 0:
            time.sleep(remaining)

except KeyboardInterrupt:
    print(f"\nStopped at step {step}", flush=True)
except Exception as e:
    print(f"\nError at step {step}: {e}", flush=True)
    import traceback
    traceback.print_exc()

env.close()
print("Done!", flush=True)
