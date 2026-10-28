"""Export only verified public files, excluding local private data and logs."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from verify_release import check_files


def export_release(root: Path, destination: Path):
    root = root.resolve()
    destination = destination.resolve()
    if destination.is_relative_to(root):
        raise ValueError("Export must be outside the working tree to avoid including local private files.")
    if destination.exists():
        raise ValueError("Export destination must not already exist.")
    check_files(root)
    manifest = json.loads((root / "release_manifest.json").read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="public-export-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "release"
        staged.mkdir()
        for relative in [*manifest["files"], "release_manifest.json"]:
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, target)
        check_files(staged)
        staged.rename(destination)
    return len(manifest["files"]) + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory outside this working tree")
    args = parser.parse_args()
    count = export_release(Path(__file__).resolve().parents[1], args.out_dir)
    print(f"Exported {count} verified public files to {args.out_dir.resolve()}")
    print("No private data, outputs, model weights, environments, or Git history were copied.")


if __name__ == "__main__":
    main()
