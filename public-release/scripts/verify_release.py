"""Verify the public software and reproduce calculations from aggregate tables."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from build_release_manifest import release_files


ROOT = Path(__file__).resolve().parents[1]


def table_signature(path: Path) -> dict:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    serialized = sorted(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
    return {
        "rows": len(rows),
        "sha256_sorted_rows": hashlib.sha256("\n".join(serialized).encode()).hexdigest(),
    }


def check_files(root: Path = ROOT) -> None:
    manifest = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
    actual_files = {path.relative_to(root).as_posix() for path in release_files(root)}
    expected_files = set(manifest["files"])
    if actual_files != expected_files:
        raise ValueError(
            f"Release file inventory mismatch: extra={sorted(actual_files - expected_files)}, "
            f"missing={sorted(expected_files - actual_files)}"
        )
    for relative, expected in manifest["files"].items():
        path = root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected["sha256"]:
            raise ValueError(f"Release file hash mismatch: {relative}")


def main() -> int:
    if Path.cwd().resolve() != ROOT:
        raise SystemExit("Run this script from the release repository root.")
    import dental_ai

    if Path(dental_ai.__file__).resolve().parent != ROOT / "dental_ai":
        raise SystemExit("The installed dental_ai package comes from another checkout.")
    started = time.monotonic()
    log_dir = ROOT / "outputs/verification"
    log_dir.mkdir(parents=True, exist_ok=True)
    check_files()
    commands = [
        ("tests", ["-m", "pytest", "-q"]),
        ("aggregates", ["scripts/reproduce_aggregate_results.py"]),
    ]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.system(),
        "package_origin": "./dental_ai",
        "gpu_inference_tested": False,
        "checks": {},
    }
    environment = {**os.environ, "MPLCONFIGDIR": str(log_dir / "matplotlib"), "MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"}
    for name, arguments in commands:
        process = subprocess.run([sys.executable, *arguments], cwd=ROOT, env=environment, capture_output=True, text=True)
        (log_dir / f"{name}.log").write_text(process.stdout + process.stderr, encoding="utf-8")
        if process.returncode:
            print((process.stdout + process.stderr)[-8000:], file=sys.stderr)
            raise SystemExit(f"FAILED: {name}; see outputs/verification/{name}.log")
        report["checks"][name] = "passed"
        if name == "tests":
            report["pytest_summary"] = process.stdout.strip().splitlines()[-1]
        print(f"PASS: {name}", flush=True)
    aggregate_report = json.loads((ROOT / "outputs/public_reproduction/report.json").read_text())
    report["aggregate_reproduction"] = aggregate_report
    report["dataset_distributed"] = False
    report["scope"] = "Software tests, aggregate arithmetic and figure regeneration only."
    report["not_reproduced"] = [
        "Collection and source-text checks", "Model inference and original reference retrieval",
        "Post-level evaluation and bootstrap resampling", "Derivation of aggregate tables from private records",
    ]
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    report["dependencies"] = {
        name: importlib.metadata.version(name)
        for name in ["pydantic", "pandas", "PyYAML", "scipy", "networkx", "matplotlib", "pytest"]
    }
    (log_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PASS: software tests, aggregate arithmetic, and figures; original-data analyses NOT reproduced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
