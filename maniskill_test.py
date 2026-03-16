"""ManiSkill3 — run physics only, then manually capture a SAPIEN camera frame."""
import os, numpy as np, torch

import sapien
import sapien.render
sapien.render.set_log_level("warning")

import mani_skill.envs
import gymnasium as gym

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
os.makedirs(OUT_DIR, exist_ok=True)

print("1. Creating env (no render)...", flush=True)
env = gym.make(
    "PickCube-v1",
    obs_mode="state",
    render_mode=None,
    num_envs=1,
    sim_backend="cpu",
)
print("2. Resetting...", flush=True)
obs, info = env.reset()
print(f"   Obs: {obs.shape}", flush=True)

print("3. Running 100 physics steps...", flush=True)
for step in range(100):
    action = env.action_space.sample() * 0.3
    obs, reward, terminated, truncated, info = env.step(action)
print(f"   Done. Final reward: {reward.item():.4f}", flush=True)

print("4. Adding manual camera to scene...", flush=True)
scene = env.unwrapped.scene
sub_scene = scene.sub_scenes[0]

cam_entity = sapien.Entity()
cam = sapien.render.RenderCameraComponent(640, 480)
cam.set_fovy(1.0)
cam.set_near(0.01)
cam.set_far(100)
cam_entity.add_component(cam)
import transforms3d

forward = np.array([0, 0, 0]) - np.array([1.0, -0.8, 0.8])
forward = forward / np.linalg.norm(forward)
up = np.array([0, 0, 1])
right = np.cross(forward, up)
right = right / np.linalg.norm(right)
up = np.cross(right, forward)
rot_mat = np.stack([forward, -right, up], axis=1)
quat = transforms3d.quaternions.mat2quat(rot_mat)

cam_entity.set_pose(sapien.Pose(p=[1.0, -0.8, 0.8], q=quat))
sub_scene.add_entity(cam_entity)
print("   Camera added.", flush=True)

print("5. Rendering...", flush=True)
sub_scene.update_render()
cam.take_picture()

pic_names = cam.get_picture_names()
print(f"   Available pictures: {pic_names}", flush=True)

rgba = cam.get_picture("Color")
print(f"   RGBA: shape={rgba.shape}, dtype={rgba.dtype}, min={rgba.min()}, max={rgba.max()}", flush=True)

if rgba.dtype == np.uint8:
    rgb = rgba[:, :, :3]
else:
    rgb = (np.clip(rgba[:, :, :3], 0, 1) * 255).astype(np.uint8)

from PIL import Image
out_path = os.path.join(OUT_DIR, "maniskill_pickcube.png")
Image.fromarray(rgb).save(out_path)
print(f"   Saved to {out_path}", flush=True)

env.close()
print("Done!", flush=True)
