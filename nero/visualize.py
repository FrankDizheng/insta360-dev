"""Matplotlib 3D preview tools for NERO poses and trajectories."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from nero.geometry import ACTIVE_FRAME_ALIGNMENT, ACTIVE_NERO_GEOMETRY, gripper_wireframe_segments, link_capsule_primitives
from nero.kinematics import approximate_reach_m, forward_kinematics
from nero.planning import rad_to_deg
from nero.types import JOINT_LIMITS_DEG, NUM_JOINTS

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D
except ImportError as exc:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    FuncAnimation = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None
    _ACTIVE_ANIMATIONS: list[object] = []


def _require_mpl() -> None:
    if plt is None:
        raise ImportError(
            "nero.visualize requires matplotlib. Install with: pip install matplotlib"
        ) from _IMPORT_ERROR


def _draw_table(
    ax,
    half_extent_m: float = 0.45,
    z: float = ACTIVE_FRAME_ALIGNMENT.table_z_m,
    color: str = "#c4c4c4",
    alpha: float = 0.35,
) -> None:
    xs = np.linspace(-half_extent_m, half_extent_m, 2)
    ys = np.linspace(-half_extent_m, half_extent_m, 2)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.full_like(xx, z, dtype=np.float64)
    ax.plot_surface(xx, yy, zz, color=color, alpha=alpha, linewidth=0, antialiased=True)


def _draw_reach_sphere(ax, reach_m: float | None = None, color: str = "#a8d5ff", alpha: float = 0.08) -> None:
    r = approximate_reach_m() if reach_m is None else float(reach_m)
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 16)
    xs = r * np.outer(np.cos(u), np.sin(v))
    ys = r * np.outer(np.sin(u), np.sin(v))
    zs = r * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xs, ys, zs, color=color, alpha=alpha, linewidth=0)


def _draw_gripper_skeleton(ax, flange_t: np.ndarray) -> None:
    for segment in gripper_wireframe_segments(flange_t, geometry=ACTIVE_NERO_GEOMETRY):
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="#555555", linewidth=1.1, alpha=0.9)


def _draw_link_envelopes(ax, joint_angles_rad: Sequence[float]) -> None:
    for cap in link_capsule_primitives(list(joint_angles_rad), geometry=ACTIVE_NERO_GEOMETRY):
        pts = np.asarray([cap.start_xyz_m, cap.end_xyz_m], dtype=np.float64)
        lw = max(2.0, 220.0 * cap.radius_m)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#7fb3d5", linewidth=lw, alpha=0.10, solid_capstyle="round")


def preview_pose(
    joint_angles_rad: Sequence[float],
    *,
    title: str | None = None,
    show_table: bool = True,
    show_reach: bool = True,
    block: bool = True,
) -> None:
    """Plot one configuration: polyline through link positions + optional table / reach hint."""
    _require_mpl()
    fk = forward_kinematics(joint_angles_rad)
    pts = fk["link_positions"]
    tcp = fk["tcp_position"]
    flange_t = fk["flange_T"]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    _draw_link_envelopes(ax, joint_angles_rad)
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "o-", color="#1f77b4", linewidth=2, markersize=5)
    _draw_gripper_skeleton(ax, flange_t)
    ax.plot([pts[-1, 0], tcp[0]], [pts[-1, 1], tcp[1]], [pts[-1, 2], tcp[2]], "--", color="#666666", linewidth=1.5)
    ax.scatter([pts[-1, 0]], [pts[-1, 1]], [pts[-1, 2]], color="#ff7f0e", s=28, label="flange")
    ax.scatter([tcp[0]], [tcp[1]], [tcp[2]], color="red", s=40, label="tcp")

    if show_table:
        _draw_table(ax)
    if show_reach:
        _draw_reach_sphere(ax)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title or "NERO pose (FK preview)")
    ax.legend(loc="upper right")
    _equal_aspect(ax, pts)
    plt.tight_layout()
    plt.show(block=block)


def preview_move(
    current_angles_rad: Sequence[float],
    target_angles_rad: Sequence[float],
    *,
    title: str | None = None,
    show_table: bool = True,
    show_reach: bool = True,
    block: bool = True,
) -> None:
    """Overlay start (green) and target (blue) arm configurations."""
    _require_mpl()
    fk0 = forward_kinematics(current_angles_rad)
    fk1 = forward_kinematics(target_angles_rad)
    p0, p1 = fk0["link_positions"], fk1["link_positions"]
    tcp0, tcp1 = fk0["tcp_position"], fk1["tcp_position"]
    flange0, flange1 = fk0["flange_T"], fk1["flange_T"]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    _draw_link_envelopes(ax, current_angles_rad)
    _draw_link_envelopes(ax, target_angles_rad)
    ax.plot(p0[:, 0], p0[:, 1], p0[:, 2], "o-", color="#2ca02c", linewidth=2, markersize=4, label="current")
    ax.plot(p1[:, 0], p1[:, 1], p1[:, 2], "o-", color="#1f77b4", linewidth=2, markersize=4, label="target")
    _draw_gripper_skeleton(ax, flange0)
    _draw_gripper_skeleton(ax, flange1)
    ax.plot([p0[-1, 0], tcp0[0]], [p0[-1, 1], tcp0[1]], [p0[-1, 2], tcp0[2]], "--", color="#2ca02c", linewidth=1.2)
    ax.plot([p1[-1, 0], tcp1[0]], [p1[-1, 1], tcp1[1]], [p1[-1, 2], tcp1[2]], "--", color="#1f77b4", linewidth=1.2)
    ax.scatter([tcp0[0]], [tcp0[1]], [tcp0[2]], color="#2ca02c", s=30, label="current_tcp")
    ax.scatter([tcp1[0]], [tcp1[1]], [tcp1[2]], color="#d62728", s=30, label="target_tcp")

    if show_table:
        _draw_table(ax)
    if show_reach:
        _draw_reach_sphere(ax)
    _draw_link_envelopes(ax, path_rad[-1])

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title or "NERO move preview (FK)")
    ax.legend(loc="upper right")
    all_pts = np.vstack([p0, p1])
    _equal_aspect(ax, all_pts)
    plt.tight_layout()
    plt.show(block=block)


def preview_trajectory(
    path_rad: Sequence[Sequence[float]],
    *,
    title: str | None = None,
    target_points: dict[str, Sequence[float]] | None = None,
    show_table: bool = True,
    show_reach: bool = True,
    animate: bool = False,
    interval_ms: int = 120,
    block: bool = True,
    save_path: str | None = None,
) -> None:
    """Visualise one joint trajectory as a static trace or interactive animation."""
    _require_mpl()
    if not path_rad:
        raise ValueError("path_rad must contain at least one waypoint")

    link_sets = []
    tcp_trace = []
    flange_trace = []
    flange_transforms = []
    for waypoint in path_rad:
        fk = forward_kinematics(waypoint)
        pts = fk["link_positions"]
        link_sets.append(pts)
        flange_trace.append(pts[-1])
        tcp_trace.append(fk["tcp_position"])
        flange_transforms.append(fk["flange_T"])

    flange_pts = np.asarray(flange_trace, dtype=np.float64)
    tcp_pts = np.asarray(tcp_trace, dtype=np.float64)
    joint_path_deg = np.asarray([rad_to_deg(waypoint) for waypoint in path_rad], dtype=np.float64)
    frame_ids = np.arange(len(path_rad), dtype=np.float64)
    all_pts = np.vstack(link_sets + [flange_pts, tcp_pts])

    fig = plt.figure(figsize=(13, 7))
    grid = fig.add_gridspec(NUM_JOINTS, 2, width_ratios=[1.7, 1.0], wspace=0.28, hspace=0.15)
    ax = fig.add_subplot(grid[:, 0], projection="3d")
    joint_axes = [fig.add_subplot(grid[idx, 1]) for idx in range(NUM_JOINTS)]

    if show_table:
        _draw_table(ax)
    if show_reach:
        _draw_reach_sphere(ax)

    target_items = target_points or {}
    for label, point in target_items.items():
        arr = np.asarray(point[:3], dtype=np.float64)
        ax.scatter([arr[0]], [arr[1]], [arr[2]], s=48, label=label)

    trace_line, = ax.plot(
        tcp_pts[:, 0],
        tcp_pts[:, 1],
        tcp_pts[:, 2],
        linestyle="--",
        linewidth=1.4,
        color="#444444",
        alpha=0.75,
        label="tcp_trace",
    )
    arm_line, = ax.plot([], [], [], "o-", color="#1f77b4", linewidth=2, markersize=4, label="arm")
    tool_line, = ax.plot([], [], [], "--", color="#666666", linewidth=1.4, label="tool_offset")
    flange_marker = ax.scatter([], [], [], color="#ff7f0e", s=28, label="flange")
    tcp_marker = ax.scatter([], [], [], color="red", s=40, label="tcp")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title or "NERO trajectory")
    _equal_aspect(ax, all_pts)
    ax.legend(loc="upper right")

    joint_lines = []
    joint_markers = []
    limit_lines = []
    for joint_idx, joint_ax in enumerate(joint_axes):
        lo_deg, hi_deg = JOINT_LIMITS_DEG[joint_idx]
        limit_lo = joint_ax.axhline(lo_deg, linestyle="--", linewidth=0.8, color="#cc6666", alpha=0.7)
        limit_hi = joint_ax.axhline(hi_deg, linestyle="--", linewidth=0.8, color="#cc6666", alpha=0.7)
        (joint_line,) = joint_ax.plot([], [], color="#1f77b4", linewidth=1.8)
        (joint_marker,) = joint_ax.plot([], [], "o", color="#ff7f0e", markersize=4)
        limit_lines.extend([limit_lo, limit_hi])
        joint_lines.append(joint_line)
        joint_markers.append(joint_marker)
        joint_ax.set_xlim(0, max(1, len(path_rad) - 1))
        joint_ax.set_ylim(lo_deg - 8.0, hi_deg + 8.0)
        joint_ax.set_ylabel(f"J{joint_idx + 1}\n(deg)", fontsize=8)
        joint_ax.grid(True, alpha=0.25)
        if joint_idx < NUM_JOINTS - 1:
            joint_ax.set_xticklabels([])
        else:
            joint_ax.set_xlabel("Frame")
        joint_ax.set_title(f"Joint {joint_idx + 1}" if joint_idx == 0 else "", fontsize=10)

    if animate:
        if FuncAnimation is None:  # pragma: no cover
            raise ImportError("matplotlib animation support is unavailable") from _IMPORT_ERROR

        def _update(frame_idx: int):
            pts = link_sets[frame_idx]
            tcp = tcp_pts[frame_idx]
            flange_t = flange_transforms[frame_idx]
            arm_line.set_data(pts[:, 0], pts[:, 1])
            arm_line.set_3d_properties(pts[:, 2])
            flange_marker._offsets3d = ([pts[-1, 0]], [pts[-1, 1]], [pts[-1, 2]])
            tcp_marker._offsets3d = ([tcp[0]], [tcp[1]], [tcp[2]])
            tool_line.set_data([pts[-1, 0], tcp[0]], [pts[-1, 1], tcp[1]])
            tool_line.set_3d_properties([pts[-1, 2], tcp[2]])
            # Re-draw the gripper skeleton for the current frame.
            while getattr(ax, "_nero_gripper_lines", []):
                line = ax._nero_gripper_lines.pop()
                line.remove()
            ax._nero_gripper_lines = []
            for segment in gripper_wireframe_segments(flange_t, geometry=ACTIVE_NERO_GEOMETRY):
                line, = ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="#555555", linewidth=1.1, alpha=0.9)
                ax._nero_gripper_lines.append(line)
            ax.set_title((title or "NERO trajectory") + f" [{frame_idx + 1}/{len(link_sets)}]")
            xs = frame_ids[: frame_idx + 1]
            for joint_idx, joint_ax in enumerate(joint_axes):
                ys = joint_path_deg[: frame_idx + 1, joint_idx]
                joint_lines[joint_idx].set_data(xs, ys)
                joint_markers[joint_idx].set_data([frame_idx], [joint_path_deg[frame_idx, joint_idx]])
                joint_ax.set_title(
                    f"Joint 1  current={joint_path_deg[frame_idx, 0]:.1f} deg" if joint_idx == 0 else ""
                )
            return (
                arm_line,
                tool_line,
                flange_marker,
                tcp_marker,
                trace_line,
                *joint_lines,
                *joint_markers,
                *limit_lines,
            )

        animation = FuncAnimation(
            fig,
            _update,
            frames=len(link_sets),
            interval=interval_ms,
            repeat=True,
            blit=False,
        )
        # Keep a reference alive until the figure is closed.
        _ACTIVE_ANIMATIONS.append(animation)
        setattr(fig, "_nero_animation", animation)
    else:
        pts = link_sets[-1]
        tcp = tcp_pts[-1]
        flange_t = flange_transforms[-1]
        arm_line.set_data(pts[:, 0], pts[:, 1])
        arm_line.set_3d_properties(pts[:, 2])
        flange_marker._offsets3d = ([pts[-1, 0]], [pts[-1, 1]], [pts[-1, 2]])
        tcp_marker._offsets3d = ([tcp[0]], [tcp[1]], [tcp[2]])
        tool_line.set_data([pts[-1, 0], tcp[0]], [pts[-1, 1], tcp[1]])
        tool_line.set_3d_properties([pts[-1, 2], tcp[2]])
        ax._nero_gripper_lines = []
        for segment in gripper_wireframe_segments(flange_t, geometry=ACTIVE_NERO_GEOMETRY):
            line, = ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="#555555", linewidth=1.1, alpha=0.9)
            ax._nero_gripper_lines.append(line)
        for joint_idx, joint_ax in enumerate(joint_axes):
            joint_lines[joint_idx].set_data(frame_ids, joint_path_deg[:, joint_idx])
            joint_markers[joint_idx].set_data([frame_ids[-1]], [joint_path_deg[-1, joint_idx]])
            joint_ax.set_title(
                f"Joint 1  final={joint_path_deg[-1, 0]:.1f} deg" if joint_idx == 0 else ""
            )

    if save_path:
        fig.savefig(save_path, dpi=150)
    backend = str(plt.get_backend()).lower()
    noninteractive_backends = {
        "agg",
        "pdf",
        "ps",
        "svg",
        "template",
        "module://matplotlib_inline.backend_inline",
    }
    if backend in noninteractive_backends and not block:
        plt.close(fig)
        return
    if backend in noninteractive_backends:
        plt.close(fig)
        return
    plt.show(block=block)
    if animate:
        try:
            _ACTIVE_ANIMATIONS.remove(getattr(fig, "_nero_animation", None))
        except ValueError:
            pass


def preview_episode_paths(
    point_a_path_rad: Sequence[Sequence[float]],
    point_b_path_rad: Sequence[Sequence[float]],
    *,
    point_a_xyz: Sequence[float] | None = None,
    point_b_xyz: Sequence[float] | None = None,
    animate: bool = False,
    block: bool = True,
    save_path: str | None = None,
) -> None:
    """Visualise a two-stage reach episode: home->A and A->B."""
    combined = list(point_a_path_rad)
    if point_b_path_rad:
        if combined and np.allclose(np.asarray(combined[-1]), np.asarray(point_b_path_rad[0])):
            combined.extend(point_b_path_rad[1:])
        else:
            combined.extend(point_b_path_rad)

    targets: dict[str, Sequence[float]] = {}
    if point_a_xyz is not None:
        targets["point_a"] = point_a_xyz
    if point_b_xyz is not None:
        targets["point_b"] = point_b_xyz

    preview_trajectory(
        combined,
        title="NERO random reach episode",
        target_points=targets,
        animate=animate,
        block=block,
        save_path=save_path,
    )


def preview_expert_rollout_comparison(
    expert_path_rad: Sequence[Sequence[float]],
    rollout_path_rad: Sequence[Sequence[float]],
    *,
    point_a_xyz: Sequence[float] | None = None,
    point_b_xyz: Sequence[float] | None = None,
    block: bool = True,
    save_path: str | None = None,
) -> None:
    """Overlay expert and rollout trajectories to inspect policy drift."""
    _require_mpl()
    if not expert_path_rad:
        raise ValueError("expert_path_rad must not be empty")
    if not rollout_path_rad:
        raise ValueError("rollout_path_rad must not be empty")

    def _collect(path_rad: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
        tcp_pts = []
        joint_deg = []
        for waypoint in path_rad:
            fk = forward_kinematics(waypoint)
            tcp_pts.append(fk["tcp_position"])
            joint_deg.append(rad_to_deg(waypoint))
        return np.asarray(tcp_pts, dtype=np.float64), np.asarray(joint_deg, dtype=np.float64)

    expert_tcp, expert_joint_deg = _collect(expert_path_rad)
    rollout_tcp, rollout_joint_deg = _collect(rollout_path_rad)

    fig = plt.figure(figsize=(13, 7))
    grid = fig.add_gridspec(NUM_JOINTS, 2, width_ratios=[1.7, 1.0], wspace=0.28, hspace=0.15)
    ax = fig.add_subplot(grid[:, 0], projection="3d")
    joint_axes = [fig.add_subplot(grid[idx, 1]) for idx in range(NUM_JOINTS)]

    _draw_table(ax)
    _draw_reach_sphere(ax)
    ax.plot(
        expert_tcp[:, 0],
        expert_tcp[:, 1],
        expert_tcp[:, 2],
        linestyle="--",
        linewidth=2.0,
        color="#2ca02c",
        label="expert_tcp",
    )
    ax.plot(
        rollout_tcp[:, 0],
        rollout_tcp[:, 1],
        rollout_tcp[:, 2],
        linestyle="-",
        linewidth=2.0,
        color="#d62728",
        label="rollout_tcp",
    )
    ax.scatter([expert_tcp[0, 0]], [expert_tcp[0, 1]], [expert_tcp[0, 2]], color="#1f77b4", s=30, label="start")
    ax.scatter([expert_tcp[-1, 0]], [expert_tcp[-1, 1]], [expert_tcp[-1, 2]], color="#2ca02c", s=36, label="expert_end")
    ax.scatter([rollout_tcp[-1, 0]], [rollout_tcp[-1, 1]], [rollout_tcp[-1, 2]], color="#d62728", s=36, label="rollout_end")
    if point_a_xyz is not None:
        p = np.asarray(point_a_xyz[:3], dtype=np.float64)
        ax.scatter([p[0]], [p[1]], [p[2]], color="#9467bd", s=45, label="point_a")
    if point_b_xyz is not None:
        p = np.asarray(point_b_xyz[:3], dtype=np.float64)
        ax.scatter([p[0]], [p[1]], [p[2]], color="#8c564b", s=45, label="point_b")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Expert vs rollout TCP trace")
    _equal_aspect(ax, np.vstack([expert_tcp, rollout_tcp]))
    ax.legend(loc="upper right")

    expert_frames = np.arange(len(expert_path_rad), dtype=np.float64)
    rollout_frames = np.arange(len(rollout_path_rad), dtype=np.float64)
    max_frames = max(len(expert_path_rad), len(rollout_path_rad))
    for joint_idx, joint_ax in enumerate(joint_axes):
        lo_deg, hi_deg = JOINT_LIMITS_DEG[joint_idx]
        joint_ax.axhline(lo_deg, linestyle="--", linewidth=0.8, color="#cc6666", alpha=0.7)
        joint_ax.axhline(hi_deg, linestyle="--", linewidth=0.8, color="#cc6666", alpha=0.7)
        joint_ax.plot(expert_frames, expert_joint_deg[:, joint_idx], color="#2ca02c", linewidth=1.8, label="expert")
        joint_ax.plot(rollout_frames, rollout_joint_deg[:, joint_idx], color="#d62728", linewidth=1.8, label="rollout")
        joint_ax.set_xlim(0, max(1, max_frames - 1))
        joint_ax.set_ylim(lo_deg - 8.0, hi_deg + 8.0)
        joint_ax.set_ylabel(f"J{joint_idx + 1}\n(deg)", fontsize=8)
        joint_ax.grid(True, alpha=0.25)
        if joint_idx < NUM_JOINTS - 1:
            joint_ax.set_xticklabels([])
        else:
            joint_ax.set_xlabel("Frame")
        if joint_idx == 0:
            joint_ax.set_title("Joint comparison", fontsize=10)
            joint_ax.legend(loc="upper right", fontsize=8)

    fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.07, wspace=0.28, hspace=0.18)
    if save_path:
        fig.savefig(save_path, dpi=150)
    backend = str(plt.get_backend()).lower()
    noninteractive_backends = {
        "agg",
        "pdf",
        "ps",
        "svg",
        "template",
        "module://matplotlib_inline.backend_inline",
    }
    if backend in noninteractive_backends:
        plt.close(fig)
        return
    plt.show(block=block)


def _equal_aspect(ax, pts: np.ndarray) -> None:
    """Rough equal scale on 3D axes from point cloud."""
    max_range = float(np.ptp(pts, axis=0).max()) / 2.0 + 1e-6
    mid = pts.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(max(0.0, mid[2] - max_range), mid[2] + max_range)
