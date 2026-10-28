"""Record distributable release files without generated outputs or model weights."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRECTORIES = {
    ".git", ".venv", "__pycache__", ".pytest_cache", "outputs", "models", "build", "dist",
}
PRIVATE_DIRECTORIES = {".aws", ".agents", ".codex", ".ssh"}
PUBLIC_DATA_FILES = {
    "data/publication_dataset/README.md", "data/publication_dataset/DATA_DICTIONARY.md",
    "data/publication_dataset/metadata.json", "data/publication_dataset/validate.py",
    "results/reference/README.md", "results/reference/corpus_summary.json",
    "results/reference/expected_evaluation.json", "results/reference/expected_ablation_tables.json",
}
PUBLIC_TABLES = {
    "eligibility_gate_metrics.csv", "eligibility_gate_confidence_intervals.csv",
    "unit_validation_metrics.csv", "unit_validation_confidence_intervals.csv",
    "unit_validation_retention.csv", "unit_validation_normalization_metrics.csv",
    "domain_distributions_35_configs.csv", "country_profile_tests.csv",
    "concept_rank_topk_stability.csv", "network_stability_results.csv",
    "domain_profile_stability.csv", "principal_findings_stability.csv",
    "population_denominators_by_country.csv", "standardized_residuals.csv",
}
PUBLIC_DATA_FILES.update(f"results/reference/aggregate_tables/{name}" for name in PUBLIC_TABLES)


def release_files(root: Path):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in EXCLUDED_DIRECTORIES or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Symlink is not allowed in the release: {relative}")
        if not path.is_file() or relative.as_posix() == "release_manifest.json":
            continue
        if any(part in PRIVATE_DIRECTORIES for part in relative.parts):
            raise ValueError(f"Private configuration is not allowed in the release: {relative}")
        if path.suffix.lower() == ".jsonl":
            raise ValueError(f"Post-level JSONL files are not allowed in the public release: {relative}")
        if relative.parts[0] in {"data", "results"} and relative.as_posix() not in PUBLIC_DATA_FILES:
            raise ValueError(f"Unapproved research artifact in public release: {relative}")
        if path.name.startswith(".env") or path.suffix in {".pyc", ".pyo"}:
            raise ValueError(f"Unexpected environment or compiled file: {relative}")
        yield path


def main() -> int:
    files = {
        path.relative_to(ROOT).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in release_files(ROOT)
    }
    manifest = {
        "package_version": "1.0.0",
        "status": "Local release candidate; MIT software and aggregate research results; underlying dataset not publicly released.",
        "scope": "Software, aggregate metadata/results, and schema documentation; no post-level study records. Generated outputs and model weights excluded.",
        "files": files,
    }
    (ROOT / "release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {len(files)} release files ({sum(item['bytes'] for item in files.values()):,} bytes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
