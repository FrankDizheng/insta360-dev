# Dual-Camera Bottle Moves Case

This case contains synchronized captures from:

- `fixed_camera`: Raspberry Pi USB camera (`/dev/video4`)
- `wrist_camera`: RGBD camera mounted on/near the robot wrist, captured through the robot `/scan` endpoint

The bottle is the transparent red-cap bottle used in the NERO grasp experiments.

## Groups

- `move_01` to `move_03`: the bottle was moved between captures; the arm/camera pose was intended to stay fixed.
- `move_04` to `move_06`: the bottle was kept fixed; the arm was manually moved so the wrist-camera viewpoint changes.

Each group contains:

- `fixed_camera.jpg`
- `fixed_camera_meta.json`
- `wrist_camera_color.jpg`
- `wrist_camera_depth.png`
- `wrist_camera_meta.json`

Use `manifest.json` as the index for all groups and capture conditions.
