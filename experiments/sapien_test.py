"""Minimal SAPIEN render test."""
import sapien
import sapien.render
import numpy as np

print("Creating engine and renderer...", flush=True)
sapien.render.set_log_level("warning")

scene = sapien.Scene()
print("Scene created!", flush=True)

scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
print("Lights added!", flush=True)

builder = scene.create_actor_builder()
builder.add_box_collision(half_size=[0.05, 0.05, 0.05])
builder.add_box_visual(half_size=[0.05, 0.05, 0.05], material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 1]))
box = builder.build(name="red_box")
box.set_pose(sapien.Pose(p=[0, 0, 0.05]))
print("Box created!", flush=True)

print("Skipping ground plane, going to camera...", flush=True)

cam = scene.add_camera(name="cam", width=640, height=480, fovy=1.0, near=0.1, far=100)
cam.set_pose(sapien.Pose(p=[0.5, -0.3, 0.3], q=[0.95, 0.05, 0.2, 0.1]))
print("Camera created!", flush=True)

scene.step()
scene.update_render()
cam.take_picture()
print("Picture taken!", flush=True)

rgba = cam.get_picture("Color")
rgb = (np.clip(rgba[:, :, :3], 0, 1) * 255).astype(np.uint8)

from PIL import Image
Image.fromarray(rgb).save("d:/DevProjects/insta360-dev/captures/sapien_test.png")
print(f"Saved! Shape: {rgb.shape}", flush=True)
