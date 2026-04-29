"""Visualize a generated random reach dataset episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nero.visualize import preview_episode_paths  # noqa: E402


def _load_episode(dataset_path: Path, episode_index: int) -> dict[str, object]:
    with dataset_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == episode_index:
                return json.loads(line)
    raise IndexError(f"Episode index {episode_index} is out of range for {dataset_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one random SimNERO reach episode")
    parser.add_argument(
        "--dataset",
        default=str(REPO_ROOT / "experiments" / "nero_sim" / "outputs" / "smoke_random_reach.jsonl"),
        help="Path to generated JSONL dataset",
    )
    parser.add_argument("--episode", type=int, default=0, help="Zero-based episode index")
    parser.add_argument("--animate", action="store_true", help="Animate the arm along the trajectory")
    parser.add_argument("--save", default="", help="Optional output image path for a static snapshot")
    parser.add_argument("--no-block", action="store_true", help="Show figure without blocking")
    args = parser.parse_args()

    episode = _load_episode(Path(args.dataset), args.episode)
    point_a = episode["point_a"]
    point_b = episode["point_b"]
    preview_episode_paths(
        point_a["plan"]["path_rad"],
        point_b["plan"]["path_rad"],
        point_a_xyz=point_a["xyz_m"],
        point_b_xyz=point_b["xyz_m"],
        animate=args.animate,
        block=not args.no_block,
        save_path=args.save or None,
    )


if __name__ == "__main__":
    main()
