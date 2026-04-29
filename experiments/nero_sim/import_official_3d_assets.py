"""Import and index official NERO 3D assets.

Supported workflows:
1. import a manually downloaded official tarball
2. import an existing local directory containing STEP/STL/OBJ assets

This is useful when the official download link redirects to a login page but
the user already has the official STEP assemblies on disk.
"""

from __future__ import annotations

import argparse
import shutil
import json
from pathlib import Path
import sys
import tarfile


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SUPPORTED_SUFFIXES = {".stl", ".step", ".stp", ".obj", ".iges", ".igs", ".sldprt", ".slas"}


def is_html_stub(path: Path) -> bool:
    try:
        head = path.read_bytes()[:256]
    except OSError:
        return False
    return head.lstrip().startswith(b"<!doctype html") or b"<html" in head.lower()


def build_manifest(extract_dir: Path) -> dict[str, object]:
    meshes = []
    for path in sorted(extract_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            meshes.append(str(path.relative_to(extract_dir)))
    return {
        "extract_dir": str(extract_dir),
        "mesh_count": len(meshes),
        "meshes": meshes,
    }


def import_source_directory(source_dir: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        target = extract_dir / path.name
        shutil.copy2(path, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import official NERO 3D assets from tarball or local STEP directory")
    parser.add_argument(
        "--tar",
        default=str(REPO_ROOT / "assets" / "nero_official_3d" / "NERO_3d_1210.tar"),
        help="Path to the official tarball",
    )
    parser.add_argument(
        "--source-dir",
        default="",
        help="Optional local directory containing official STEP/STL/OBJ assets",
    )
    parser.add_argument(
        "--extract-dir",
        default=str(REPO_ROOT / "assets" / "nero_official_3d" / "imported"),
        help="Directory where assets will be extracted",
    )
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "assets" / "nero_official_3d" / "manifest.json"),
        help="Output manifest JSON path",
    )
    args = parser.parse_args()

    extract_dir = Path(args.extract_dir)
    source_dir = Path(args.source_dir) if args.source_dir else None

    if source_dir and source_dir.exists():
        import_source_directory(source_dir, extract_dir)
    else:
        tar_path = Path(args.tar)
        if not tar_path.exists():
            raise FileNotFoundError(f"Tarball not found: {tar_path}")
        if is_html_stub(tar_path):
            raise RuntimeError(
                "The provided file is an HTML login/redirect page, not the real 3D tarball. "
                "Please download the official package manually from the logged-in browser and "
                f"save it to {tar_path}, or use --source-dir with a local STEP folder."
            )
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path) as tar:
            tar.extractall(extract_dir)

    manifest = build_manifest(extract_dir)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
