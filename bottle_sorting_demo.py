"""
Bottle sorting demo: VLM identifies different-shaped bottles and sorts them into bins.
Generates an animated GIF showing the full multi-cycle sorting process.
"""
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
from mani_skill.utils.building import actors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bridge"))
from model_feed import analyze_image

# ── Constants ──
URDF_PATH = r"C:\Python313\Lib\site-packages\mani_skill\assets\robots\panda\panda_v2.urdf"
GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0
JOINT_LOWER = torch.tensor([-2.8, -1.7, -2.8, -3.0, -2.8, -0.01, -2.8])
JOINT_UPPER = torch.tensor([ 2.8,  1.7,  2.8, -0.07,  2.8,  3.75,  2.8])
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures", "sorting_video")

# ── Bottle definitions ──
BOTTLES = {
    "tall_bottle": {
        "radius": 0.022, "half_length": 0.05,
        "color": [0.2, 0.4, 1.0, 1.0],
        "pos": [0.0, 0.10, 0.05],
        "bin": "bin_tall",
        "desc": "tall blue cylinder",
    },
    "wide_bottle": {
        "radius": 0.035, "half_length": 0.025,
        "color": [0.2, 0.9, 0.3, 1.0],
        "pos": [0.08, -0.06, 0.025],
        "bin": "bin_short",
        "desc": "short wide green cylinder",
    },
    "small_bottle": {
        "radius": 0.018, "half_length": 0.03,
        "color": [1.0, 0.5, 0.1, 1.0],
        "pos": [-0.05, 0.02, 0.03],
        "bin": "bin_short",
        "desc": "small orange cylinder",
    },
}

BINS = {
    "bin_tall": {
        "pos": [0.18, 0.20, 0.0],
        "color": [0.25, 0.25, 0.85, 0.9],
        "label": "Bin A (Tall)",
    },
    "bin_short": {
        "pos": [0.18, -0.20, 0.0],
        "color": [0.25, 0.85, 0.25, 0.9],
        "label": "Bin B (Short/Wide)",
    },
}

# ── VLM prompt ──
SYSTEM_PROMPT = """You are a robot controller for a bottle sorting task.

Scene objects:
- tall_bottle: a TALL blue cylinder → belongs in bin_tall (blue square, upper area)
- wide_bottle: a SHORT WIDE green cylinder → belongs in bin_short (green square, lower area)
- small_bottle: a SMALL orange cylinder → belongs in bin_short (green square, lower area)
- bin_tall: blue square target zone (for tall bottles)
- bin_short: green square target zone (for short/wide bottles)

Output EXACTLY ONE JSON action:
{"action": "move_above", "target": "<name>"}
{"action": "lower", "target": "<name>"}
{"action": "grasp"}
{"action": "lift"}
{"action": "release"}
{"action": "done"}

<name> can be: tall_bottle, wide_bottle, small_bottle, bin_tall, bin_short

For each bottle, the sequence is:
  move_above <bottle> → lower <bottle> → grasp → lift → move_above <correct_bin> → lower <correct_bin> → release

Then proceed to the next unsorted bottle. When all bottles are sorted, output {"action": "done"}.

Rules:
- Output ONLY the JSON, no other text.
- Sort one bottle at a time. Complete the full cycle before starting the next.
- tall_bottle → bin_tall; wide_bottle → bin_short; small_bottle → bin_short.
"""

# ── IK solver ──
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

def get_tcp_pos(q_arm):
    tf = chain.forward_kinematics(q_arm.unsqueeze(0))
    return tf.get_matrix()[0, :3, 3].numpy()

def w2b(pos, base):
    return pos - base

# ── Rendering ──
def setup_camera(sub_scene):
    ent = sapien.Entity()
    cam = sapien.render.RenderCameraComponent(640, 480)
    cam.set_fovy(1.0); cam.set_near(0.01); cam.set_far(100)
    ent.add_component(cam)
    fwd = -np.array([0.7, -0.5, 0.55]); fwd /= np.linalg.norm(fwd)
    up = np.array([0, 0, 1])
    r = np.cross(fwd, up); r /= np.linalg.norm(r)
    up = np.cross(r, fwd)
    q = transforms3d.quaternions.mat2quat(np.stack([fwd, -r, up], axis=1))
    ent.set_pose(sapien.Pose(p=[0.7, -0.5, 0.55], q=q))
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
        font = ImageFont.truetype("arial.ttf", 18)
        font_sm = ImageFont.truetype("arial.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_sm = font
    draw.rectangle([(0, 0), (640, 30)], fill=(0, 0, 0, 180))
    label = f"[Step {step}] " if step is not None else ""
    draw.text((8, 5), label + text, fill=(255, 255, 255), font=font)
    draw.text((8, 462), "Qwen VLM → Bottle Sorting Demo", fill=(200, 200, 200), font=font_sm)
    return np.array(img)

def execute_motion(env, q_from, q_to, g_from, g_to, steps, sub_scene, cam,
                   frames, label, step_num, held_bottle=None, base=None, every_n=6):
    for i in range(steps):
        t = 0.5 * (1.0 - np.cos(i / steps * np.pi))
        arm = (1.0 - t) * q_from + t * q_to
        grip = g_from + t * (g_to - g_from)
        action = np.append(arm, grip).astype(np.float32)
        env.step(torch.tensor(action).unsqueeze(0))

        if held_bottle is not None and base is not None:
            tcp = get_tcp_pos(torch.tensor(arm, dtype=torch.float32))
            world_tcp = tcp + base
            world_tcp[2] -= 0.05
            held_bottle.set_pose(sapien.Pose(p=world_tcp.tolist(), q=UPRIGHT_Q))

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

# ── Scene building ──
UPRIGHT_Q = transforms3d.euler.euler2quat(np.pi / 2, 0, 0, 'sxyz')

def build_scene(scene):
    """Add bottles (kinematic) and bins (static) to the scene."""
    bottle_actors = {}
    for name, spec in BOTTLES.items():
        builder = scene.create_actor_builder()
        builder.add_cylinder_collision(
            radius=spec["radius"],
            half_length=spec["half_length"], density=2000,
        )
        mat = sapien.render.RenderMaterial(base_color=spec["color"])
        builder.add_cylinder_visual(
            radius=spec["radius"],
            half_length=spec["half_length"], material=mat,
        )
        builder.set_initial_pose(sapien.Pose(p=spec["pos"], q=UPRIGHT_Q))
        actor = builder.build_kinematic(name=name)
        bottle_actors[name] = actor
        print(f"  Created {name}: {spec['desc']}")

    bin_actors = {}
    for name, spec in BINS.items():
        bin_actor = actors.build_box(
            scene, half_sizes=[0.045, 0.045, 0.004],
            color=spec["color"], name=name,
            body_type="static",
            initial_pose=sapien.Pose(p=spec["pos"]),
        )
        bin_actors[name] = bin_actor
        print(f"  Created {name}: {spec['label']}")

    return bottle_actors, bin_actors


ABOVE_Z = 0.22

PICK_PLACE_TEMPLATE = [
    ("move_above", "{bottle}"),
    ("lower", "{bottle}"),
    ("grasp", ""),
    ("lift", ""),
    ("move_above", "{bin}"),
    ("lower", "{bin}"),
    ("release", ""),
]


class SortingState:
    def __init__(self, env, base, q_arm, grip, bottle_actors, sub, cam):
        self.env = env
        self.base = base
        self.q_arm = q_arm
        self.grip = grip
        self.bottle_actors = bottle_actors
        self.sub = sub
        self.cam = cam
        self.held_bottle_name = None
        self.sorted_bottles = set()
        self.placed_offsets = {"bin_tall": 0, "bin_short": 0}

    def execute_step(self, act_name, act_tgt, frames, label, step):
        if act_name == "move_above":
            if act_tgt in BOTTLES:
                p_w = np.array(BOTTLES[act_tgt]["pos"])
            elif act_tgt in BINS:
                p_w = np.array(BINS[act_tgt]["pos"])
            else:
                return
            target_b = w2b(p_w, self.base)
            tgt = torch.tensor([target_b[0], target_b[1], ABOVE_Z], dtype=torch.float32)
            q_new, _ = solve_ik(tgt, self.q_arm)
            held = self.bottle_actors.get(self.held_bottle_name)
            execute_motion(self.env, self.q_arm.numpy(), q_new.numpy(),
                          self.grip, self.grip, 90, self.sub, self.cam, frames,
                          label, step, held_bottle=held, base=self.base)
            self.q_arm = q_new

        elif act_name == "lower":
            if act_tgt in BOTTLES:
                p_w = np.array(BOTTLES[act_tgt]["pos"])
                hl = BOTTLES[act_tgt]["half_length"]
                z = p_w[2] + hl + 0.015
            elif act_tgt in BINS:
                p_w = np.array(BINS[act_tgt]["pos"])
                z = 0.12
            else:
                return
            target_b = w2b(p_w, self.base)
            tgt = torch.tensor([target_b[0], target_b[1], z], dtype=torch.float32)
            q_new, _ = solve_ik(tgt, self.q_arm)
            held = self.bottle_actors.get(self.held_bottle_name)
            execute_motion(self.env, self.q_arm.numpy(), q_new.numpy(),
                          self.grip, self.grip, 90, self.sub, self.cam, frames,
                          label, step, held_bottle=held, base=self.base)
            self.q_arm = q_new

        elif act_name == "grasp":
            execute_motion(self.env, self.q_arm.numpy(), self.q_arm.numpy(),
                          GRIPPER_OPEN, GRIPPER_CLOSE, 50,
                          self.sub, self.cam, frames, label, step)
            self.grip = GRIPPER_CLOSE
            tcp_world = get_tcp_pos(self.q_arm) + self.base
            best_name, best_dist = None, float("inf")
            for bname in BOTTLES:
                if bname in self.sorted_bottles:
                    continue
                bp = np.array(BOTTLES[bname]["pos"])
                d = np.linalg.norm(tcp_world[:2] - bp[:2])
                if d < best_dist:
                    best_dist = d; best_name = bname
            if best_name and best_dist < 0.10:
                self.held_bottle_name = best_name
                print(f"    -> Grasped {best_name}", flush=True)

        elif act_name == "lift":
            tcp_b = get_tcp_pos(self.q_arm)
            tgt = torch.tensor([tcp_b[0], tcp_b[1], ABOVE_Z + 0.05], dtype=torch.float32)
            q_new, _ = solve_ik(tgt, self.q_arm)
            held = self.bottle_actors.get(self.held_bottle_name)
            execute_motion(self.env, self.q_arm.numpy(), q_new.numpy(),
                          self.grip, self.grip, 90, self.sub, self.cam, frames,
                          label, step, held_bottle=held, base=self.base)
            self.q_arm = q_new

        elif act_name == "release":
            execute_motion(self.env, self.q_arm.numpy(), self.q_arm.numpy(),
                          GRIPPER_CLOSE, GRIPPER_OPEN, 50,
                          self.sub, self.cam, frames, label, step)
            self.grip = GRIPPER_OPEN
            if self.held_bottle_name:
                self.sorted_bottles.add(self.held_bottle_name)
                tbin = BOTTLES[self.held_bottle_name]["bin"]
                offset = self.placed_offsets[tbin]
                self.placed_offsets[tbin] += 1
                bp = np.array(BINS[tbin]["pos"]).copy()
                bp[1] += (offset - 0.5) * 0.09
                bp[2] = BOTTLES[self.held_bottle_name]["half_length"]
                self.bottle_actors[self.held_bottle_name].set_pose(
                    sapien.Pose(p=bp.tolist(), q=UPRIGHT_Q))
                print(f"    -> Placed {self.held_bottle_name} in {tbin}", flush=True)
                self.held_bottle_name = None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []

    print("Creating environment...", flush=True)
    env = gym.make("PickCube-v1", obs_mode="state_dict", render_mode=None,
                   num_envs=1, sim_backend="cpu", control_mode="pd_joint_pos",
                   max_episode_steps=999999)
    obs, info = env.reset(seed=42)

    robot = env.unwrapped.agent.robot
    base = robot.pose.raw_pose[0, :3].numpy()
    q_arm = torch.tensor(robot.get_qpos()[0].numpy()[:7], dtype=torch.float32)

    env.unwrapped.cube.set_pose(sapien.Pose(p=[10, 10, -10]))
    env.unwrapped.goal_site.set_pose(sapien.Pose(p=[10, 10, -10]))

    scene = env.unwrapped.scene
    sub = scene.sub_scenes[0]

    print("Building sorting scene...", flush=True)
    bottle_actors, bin_actors = build_scene(scene)

    for _ in range(5):
        action = np.append(q_arm.numpy(), GRIPPER_OPEN).astype(np.float32)
        env.step(torch.tensor(action).unsqueeze(0))

    for name, spec in BOTTLES.items():
        bottle_actors[name].set_pose(sapien.Pose(p=spec["pos"], q=UPRIGHT_Q))

    cam = setup_camera(sub)
    st = SortingState(env, base, q_arm, GRIPPER_OPEN, bottle_actors, sub, cam)

    for _ in range(12):
        rgb = snap(sub, cam)
        frames.append(add_label(rgb, "Bottle Sorting + Plan Cache Demo"))

    # ── Sorting loop with plan template caching ──
    action_history = []
    vlm_time_total = 0.0
    cache_hits = 0
    vlm_calls = 0
    step = 0
    template_learned = False

    print("\nStarting VLM sorting loop...\n", flush=True)

    remaining = [n for n in BOTTLES if n not in st.sorted_bottles]
    while remaining and step < 30:
        bottle_name = remaining[0]
        target_bin = BOTTLES[bottle_name]["bin"]

        if not template_learned:
            print(f"  Cycle 1 ({bottle_name}): VLM step-by-step reasoning...", flush=True)
            for _ in range(8):
                rgb = snap(sub, cam)
                frames.append(add_label(rgb, "VLM reasoning...", step))
                tmp = os.path.join(OUT_DIR, "_tmp.png")
                Image.fromarray(rgb).save(tmp)

                rem = [n for n in BOTTLES if n not in st.sorted_bottles]
                status = f"\n\nBottles remaining: {rem}"
                if st.held_bottle_name:
                    status += f"\nCurrently holding: {st.held_bottle_name}"
                hist = ""
                if action_history:
                    hist = "\n\nActions completed so far:\n"
                    for i, a in enumerate(action_history):
                        hist += f"  Step {i}: {a}\n"
                    hist += "\nDo NOT repeat or skip steps."
                prompt = SYSTEM_PROMPT + status + hist
                prompt += f"\nStep {step}. What is the next action?"

                t0 = time.time()
                resp = analyze_image(tmp, prompt, stream=False)
                dt = time.time() - t0
                vlm_time_total += dt
                vlm_calls += 1
                act = parse_action(resp)
                act_name = act.get("action", "?")
                act_tgt = act.get("target", "")
                print(f"  Step {step}: [VLM] {act_name} {act_tgt} ({dt:.1f}s)", flush=True)
                action_history.append(f"{act_name} {act_tgt}".strip())

                if act_name == "done":
                    break

                label = f"VLM: {act_name} {act_tgt}"
                st.execute_step(act_name, act_tgt, frames, label, step)
                step += 1

                if act_name == "release":
                    template_learned = True
                    print(f"  -> Plan template learned from cycle 1!", flush=True)
                    break

        else:
            print(f"  Cycle ({bottle_name}): Using CACHED plan template", flush=True)
            plan = [(a.replace("{bottle}", bottle_name).replace("{bin}", target_bin),
                     t.replace("{bottle}", bottle_name).replace("{bin}", target_bin))
                    for a, t in PICK_PLACE_TEMPLATE]

            for act_name, act_tgt in plan:
                cache_hits += 1
                label = f"CACHED: {act_name} {act_tgt}"
                print(f"  Step {step}: [CACHED] {act_name} {act_tgt} (0.0s)", flush=True)
                action_history.append(f"{act_name} {act_tgt}".strip())
                st.execute_step(act_name, act_tgt, frames, label, step)
                step += 1

        remaining = [n for n in BOTTLES if n not in st.sorted_bottles]

    for _ in range(20):
        rgb = snap(sub, cam)
        frames.append(add_label(rgb, "ALL SORTED! Task complete!", step))

    env.close()

    # ── Performance stats ──
    total_steps = len(action_history)
    avg_vlm = vlm_time_total / max(vlm_calls, 1)
    no_cache_time = total_steps * avg_vlm
    with_cache_time = vlm_time_total
    print(f"\n{'='*55}")
    print(f"  Performance Summary — Plan Template Caching")
    print(f"{'='*55}")
    print(f"  Total action steps:    {total_steps}")
    print(f"  VLM calls (cycle 1):   {vlm_calls}  ({vlm_time_total:.1f}s)")
    print(f"  Cached steps (2 & 3):  {cache_hits}  (0.0s)")
    print(f"  Avg VLM latency:       {avg_vlm:.1f}s/call")
    print(f"  ───────────────────────────────────────")
    print(f"  Without cache:  ~{no_cache_time:.0f}s  ({total_steps} VLM calls)")
    print(f"  With cache:     ~{with_cache_time:.0f}s  ({vlm_calls} VLM calls)")
    print(f"  Speedup:        {no_cache_time/max(with_cache_time,1):.1f}x")
    print(f"  On production line (100 repeats):")
    print(f"    Without cache: ~{100*total_steps*avg_vlm/60:.0f} min")
    print(f"    With cache:    ~{vlm_time_total/60:.1f} min (VLM called only once)")
    print(f"{'='*55}\n")

    # ── Save GIF ──
    gif_path = os.path.join(OUT_DIR, "bottle_sorting.gif")
    print(f"\nSaving GIF ({len(frames)} frames)...", flush=True)
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                       duration=80, loop=0)
    print(f"Saved to {gif_path}", flush=True)

    for i, idx in enumerate([0, len(frames)//3, 2*len(frames)//3, len(frames)-1]):
        Image.fromarray(frames[idx]).save(os.path.join(OUT_DIR, f"key_{i}.png"))

    print("Done!", flush=True)


if __name__ == "__main__":
    main()
