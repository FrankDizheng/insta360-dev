"""Debug ManiSkill3 crash step by step."""
import sapien
import sapien.render
import numpy as np

sapien.render.set_log_level("info")

print("1. Testing RenderSystem creation...", flush=True)
try:
    rs = sapien.render.RenderSystem("cuda:0")
    print(f"   RenderSystem on cuda:0 OK: {rs}", flush=True)
except Exception as e:
    print(f"   RenderSystem cuda:0 failed: {e}", flush=True)
    try:
        rs = sapien.render.RenderSystem()
        print(f"   RenderSystem default OK: {rs}", flush=True)
    except Exception as e2:
        print(f"   RenderSystem default also failed: {e2}", flush=True)

print("2. Testing ManiSkill env with state_only obs...", flush=True)
import mani_skill.envs
import gymnasium as gym

try:
    env = gym.make(
        "PickCube-v1",
        obs_mode="state",
        render_mode=None,
        num_envs=1,
        sim_backend="cpu",
    )
    print("   env created (no render)!", flush=True)
    obs, info = env.reset()
    print(f"   reset OK! obs keys: {obs.keys() if isinstance(obs, dict) else type(obs)}", flush=True)
    env.close()
except Exception as e:
    print(f"   Failed: {e}", flush=True)

print("3. Testing with rgb_array render...", flush=True)
try:
    env2 = gym.make(
        "PickCube-v1",
        obs_mode="state",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend="cpu",
    )
    print("   env created with render!", flush=True)
    obs, info = env2.reset()
    print("   reset OK!", flush=True)
    frame = env2.render()
    print(f"   render OK! type={type(frame)}", flush=True)
    env2.close()
except Exception as e:
    print(f"   Failed: {e}", flush=True)

print("Done.", flush=True)
