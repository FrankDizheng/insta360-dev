"""Extract a lightweight geometry alignment summary from the official NERO STEP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cadquery as cq
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def analyze_step(
    step_path: Path,
    *,
    tool_end: str = "max",
    tcp_offset_m: float = 0.132,
    slab_length_m: float = 0.132,
) -> dict[str, object]:
    shape = cq.importers.importStep(str(step_path))
    vertices = np.asarray([[v.X, v.Y, v.Z] for v in shape.vertices().vals()], dtype=np.float64)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    extents = maxs - mins
    primary_axis = int(np.argmax(extents))
    axis_names = ["x", "y", "z"]

    if tool_end not in {"min", "max"}:
        raise ValueError("tool_end must be 'min' or 'max'")

    slab_len = float(slab_length_m * 1000.0)
    coord = vertices[:, primary_axis]
    if tool_end == "max":
        mask = coord >= (coord.max() - slab_len)
        tip_coord = float(coord.max())
    else:
        mask = coord <= (coord.min() + slab_len)
        tip_coord = float(coord.min())
    slab = vertices[mask]
    slab_mins = slab.min(axis=0)
    slab_maxs = slab.max(axis=0)
    slab_extents = slab_maxs - slab_mins

    orth_axes = [idx for idx in range(3) if idx != primary_axis]
    width_axis = orth_axes[int(np.argmax(slab_extents[orth_axes]))]
    height_axis = orth_axes[int(np.argmin(slab_extents[orth_axes]))]

    return {
        "source_note": "Derived from official STEP geometry by vertex-cloud slab analysis.",
        "step_file": str(step_path),
        "primary_axis": axis_names[primary_axis],
        "tool_end": tool_end,
        "tcp_offset_m": tcp_offset_m,
        "table_z_m": float(mins[primary_axis] / 1000.0),
        "base_to_table_offset_m": float(abs(mins[primary_axis]) / 1000.0),
        "bbox": {
            "xmin": float(mins[0] / 1000.0),
            "xmax": float(maxs[0] / 1000.0),
            "ymin": float(mins[1] / 1000.0),
            "ymax": float(maxs[1] / 1000.0),
            "zmin": float(mins[2] / 1000.0),
            "zmax": float(maxs[2] / 1000.0),
            "xlen": float(extents[0] / 1000.0),
            "ylen": float(extents[1] / 1000.0),
            "zlen": float(extents[2] / 1000.0),
        },
        "tool_end_slab": {
            "length_m": float(slab_extents[primary_axis] / 1000.0),
            "width_axis": axis_names[width_axis],
            "width_m": float(slab_extents[width_axis] / 1000.0),
            "height_axis": axis_names[height_axis],
            "height_m": float(slab_extents[height_axis] / 1000.0),
            "tip_coord_m": float(tip_coord / 1000.0),
        },
        "jaw_length_m": float(slab_extents[primary_axis] / 1000.0),
        "jaw_outer_width_m": float(slab_extents[width_axis] / 1000.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract STEP alignment summary for official NERO geometry")
    parser.add_argument(
        "--step",
        default=str(REPO_ROOT / "assets" / "nero_official_3d" / "official_step" / "NERO夹爪版外发.STEP"),
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "assets" / "nero_official_3d" / "step_alignment_gripper.json"),
    )
    parser.add_argument("--tool-end", choices=["min", "max"], default="max")
    parser.add_argument("--tcp-offset-mm", type=float, default=132.0)
    parser.add_argument("--slab-length-mm", type=float, default=132.0)
    args = parser.parse_args()

    result = analyze_step(
        Path(args.step),
        tool_end=args.tool_end,
        tcp_offset_m=args.tcp_offset_mm / 1000.0,
        slab_length_m=args.slab_length_mm / 1000.0,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
