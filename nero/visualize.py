"""Matplotlib 3D preview of NERO arm poses (uses local FK from `kinematics`)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from nero.kinematics import approximate_reach_m, forward_kinematics

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D
except ImportError as exc:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _require_mpl() -> None:
    if plt is None:
        raise ImportError(
            "nero.visualize requires matplotlib. Install with: pip install matplotlib"
        ) from _IMPORT_ERROR


def _draw_table(
    ax,
    half_extent_m: float = 0.45,
    z: float = 0.0,
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

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "o-", color="#1f77b4", linewidth=2, markersize=5)
    ax.scatter([pts[-1, 0]], [pts[-1, 1]], [pts[-1, 2]], color="red", s=40, label="flange")

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

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(p0[:, 0], p0[:, 1], p0[:, 2], "o-", color="#2ca02c", linewidth=2, markersize=4, label="current")
    ax.plot(p1[:, 0], p1[:, 1], p1[:, 2], "o-", color="#1f77b4", linewidth=2, markersize=4, label="target")

    if show_table:
        _draw_table(ax)
    if show_reach:
        _draw_reach_sphere(ax)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title or "NERO move preview (FK)")
    ax.legend(loc="upper right")
    all_pts = np.vstack([p0, p1])
    _equal_aspect(ax, all_pts)
    plt.tight_layout()
    plt.show(block=block)


def _equal_aspect(ax, pts: np.ndarray) -> None:
    """Rough equal scale on 3D axes from point cloud."""
    max_range = float(np.ptp(pts, axis=0).max()) / 2.0 + 1e-6
    mid = pts.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(max(0.0, mid[2] - max_range), mid[2] + max_range)
