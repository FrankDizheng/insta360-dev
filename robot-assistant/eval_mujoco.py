"""
Step 3: Evaluate fine-tuned SmolVLA in MuJoCo simulation.
Loads the SO-101 model, renders camera views, feeds them to the policy,
and applies the output actions to the simulated arm.
"""

import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import mujoco
import time

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

SCENE_XML = "/data/xinna/robot-arm/SO-ARM100/Simulation/SO101/scene.xml"
MODEL_PATH = "/data/xinna/robot-arm/outputs/smolvla_so100_pickplace/final"
NUM_STEPS = 300
RENDER_W, RENDER_H = 640, 480

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def log(msg):
    print(f"[EVAL] {msg}", flush=True)


def get_joint_ids(model):
    """Get MuJoCo joint IDs for our 6 joints."""
    ids = []
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid == -1:
            log(f"WARNING: joint '{name}' not found in model")
        ids.append(jid)
    return ids


def render_camera(model, data, renderer, cam_name="top"):
    """Render an offscreen camera view."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id == -1:
        return None
    renderer.update_scene(data, camera=cam_name)
    img = renderer.render()
    return img


def add_cameras_and_objects(model_xml_path):
    """Create an augmented scene XML with cameras and a target object."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(model_xml_path)
    root = tree.getroot()

    worldbody = root.find("worldbody")

    # Add top camera (overhead view)
    ET.SubElement(worldbody, "camera", {
        "name": "top",
        "pos": "0.0 0.0 0.8",
        "quat": "1 0 0 0",
        "fovy": "60",
    })

    # Add wrist camera placeholder (approximate position near end effector)
    ET.SubElement(worldbody, "camera", {
        "name": "wrist",
        "pos": "0.15 0.0 0.25",
        "quat": "0.707 0.707 0 0",
        "fovy": "90",
    })

    # Add a simple target object (red cube)
    ET.SubElement(root.find(".//asset") or ET.SubElement(root, "asset"), "material", {
        "name": "red",
        "rgba": "1 0.2 0.2 1",
    })
    cube = ET.SubElement(worldbody, "body", {
        "name": "target_cube",
        "pos": "0.15 0.05 0.02",
    })
    ET.SubElement(cube, "geom", {
        "type": "box",
        "size": "0.015 0.015 0.015",
        "material": "red",
        "mass": "0.01",
    })
    ET.SubElement(cube, "freejoint")

    augmented_path = model_xml_path.replace("scene.xml", "scene_eval.xml")
    tree.write(augmented_path)
    log(f"Wrote augmented scene: {augmented_path}")
    return augmented_path


def main():
    log("=" * 60)
    log("Step 3: MuJoCo Simulation Evaluation")
    log("=" * 60)

    # Augment scene with cameras and objects
    log("Creating augmented scene with cameras...")
    scene_path = add_cameras_and_objects(SCENE_XML)

    # Load MuJoCo model
    log("Loading MuJoCo model...")
    mj_model = mujoco.MjModel.from_xml_path(scene_path)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    joint_ids = get_joint_ids(mj_model)
    valid_joints = [(i, name) for i, (jid, name) in enumerate(zip(joint_ids, JOINT_NAMES)) if jid != -1]
    log(f"Found {len(valid_joints)} joints: {[name for _, name in valid_joints]}")

    # Print all joints in model for debugging
    log("All joints in MuJoCo model:")
    for i in range(mj_model.njnt):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        log(f"  joint {i}: {name}, qpos_idx={mj_model.jnt_qposadr[i]}")

    # Setup offscreen renderer with EGL
    log(f"Setting up EGL offscreen renderer ({RENDER_W}x{RENDER_H})...")
    gl_ctx = mujoco.GLContext(RENDER_W, RENDER_H)
    gl_ctx.make_current()
    renderer = mujoco.Renderer(mj_model, height=RENDER_H, width=RENDER_W)

    # Load policy
    log(f"Loading fine-tuned SmolVLA from {MODEL_PATH}...")
    policy = SmolVLAPolicy.from_pretrained(MODEL_PATH)
    policy = policy.to("cuda")
    policy.eval()
    log("Policy loaded!")

    # Get tokenizer
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    task_text = "pick up the red block"
    max_length = policy.config.tokenizer_max_length
    tokenized = tokenizer(
        [task_text],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    log(f"Task: '{task_text}'")

    # Run simulation loop
    log(f"\nRunning {NUM_STEPS} simulation steps...")
    log("-" * 60)

    actions_history = []
    for step in range(NUM_STEPS):
        # Render cameras
        img_top = render_camera(mj_model, mj_data, renderer, "top")
        img_wrist = render_camera(mj_model, mj_data, renderer, "wrist")

        if img_top is None or img_wrist is None:
            log(f"Camera render failed at step {step}")
            break

        # Convert images: HWC uint8 -> CHW float32, normalized to [0, 1]
        img_top_t = torch.from_numpy(img_top.copy()).permute(2, 0, 1).float() / 255.0
        img_wrist_t = torch.from_numpy(img_wrist.copy()).permute(2, 0, 1).float() / 255.0

        # Get current joint positions
        state = np.zeros(6)
        for idx, (policy_idx, name) in enumerate(valid_joints):
            jid = joint_ids[policy_idx]
            qpos_idx = mj_model.jnt_qposadr[jid]
            state[policy_idx] = mj_data.qpos[qpos_idx]

        # Build batch
        batch = {
            "observation.state": torch.tensor(state, dtype=torch.float32).unsqueeze(0).to("cuda"),
            "observation.images.camera1": img_top_t.unsqueeze(0).to("cuda"),
            "observation.images.camera2": img_wrist_t.unsqueeze(0).to("cuda"),
            "observation.language.tokens": tokenized["input_ids"].to("cuda"),
            "observation.language.attention_mask": tokenized["attention_mask"].bool().to("cuda"),
        }

        # Get action from policy
        with torch.no_grad():
            action = policy.select_action(batch)

        action_np = action.cpu().numpy().flatten()
        actions_history.append(action_np.copy())

        # Apply action to simulation (for valid joints)
        for idx, (policy_idx, name) in enumerate(valid_joints):
            jid = joint_ids[policy_idx]
            qpos_idx = mj_model.jnt_qposadr[jid]
            mj_data.ctrl[idx] = action_np[policy_idx]

        # Step simulation
        mujoco.mj_step(mj_model, mj_data)

        if step % 30 == 0:
            log(f"step {step:3d} | state={np.round(state, 3)} | action={np.round(action_np, 3)}")

    log("-" * 60)
    log("Simulation complete!")

    # Analyze actions
    actions_arr = np.array(actions_history)
    log(f"\nAction statistics over {len(actions_arr)} steps:")
    for i, name in enumerate(JOINT_NAMES):
        col = actions_arr[:, i]
        log(f"  {name:15s}: mean={col.mean():.4f}, std={col.std():.4f}, range=[{col.min():.4f}, {col.max():.4f}]")

    variance = actions_arr.std(axis=0).mean()
    if variance < 0.01:
        log("\nWARNING: Actions have very low variance - model may not be responding to visual input")
    else:
        log(f"\nActions show meaningful variance ({variance:.4f}) - model is responding to input!")

    log("\nStep 3 complete!")


if __name__ == "__main__":
    main()
