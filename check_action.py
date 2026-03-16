import mani_skill.envs, gymnasium as gym, torch
env = gym.make("PickCube-v1", obs_mode="state_dict", render_mode=None,
               num_envs=1, sim_backend="cpu", control_mode="pd_joint_pos",
               max_episode_steps=999999)
obs, info = env.reset()
print("Action space:", env.action_space)
print("Action low:", env.action_space.low)
print("Action high:", env.action_space.high)
print()
robot = env.unwrapped.agent.robot
qpos = robot.get_qpos()[0]
print("Current qpos:", qpos.numpy())
print()
ctrl = env.unwrapped.agent.controller
for name, c in ctrl.controllers.items():
    print(f"{name}: {type(c).__name__}")
    cfg = c.config
    for attr in ["use_delta", "normalize_action", "use_target"]:
        print(f"  {attr} = {getattr(cfg, attr, 'N/A')}")
env.close()
