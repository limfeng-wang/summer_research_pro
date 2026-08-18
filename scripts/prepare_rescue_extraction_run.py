#!/usr/bin/env python3
"""Prepare an extraction/judge run for rescued CSM positives.

The script writes:
- rescued_input.jsonl
- checkpoints/classified.jsonl with rescued R1/E1-or-E2/C1-or-C2 labels
- empty annotations/errors/retrieval/extracted files

Then run the normal HF full pipeline with --resume against this out-dir. The
existing runner will skip classification and perform extraction/judge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dental_ai.pipeline import PipelineTrace
from dental_ai.run import _write_output_line
from dental_ai.schemas import ContentFunctionLabel, ExperiencerLabel, ExtractionResult, RelevanceLabel, SourcePost
from dental_ai.validate import validate_hierarchical_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="Candidate JSONL used by rescue classifier")
    parser.add_argument(
        "--decisions",
        action="append",
        required=True,
        help="rescue_decisions.jsonl file or directory containing rescue_decisions.jsonl. Repeat for shards.",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory or sharded OUT_BASE for the rescue extraction run")
    parser.add_argument("--shard-count", type=int, default=1, help="Prepare contiguous shard directories for run_hf_shards.sh")
    args = parser.parse_args()

    candidates = {_post_id(row): row for row in _read_jsonl(Path(args.candidates))}
    decisions = _load_decisions([Path(item) for item in args.decisions])
    rescued = []
    seen: set[str] = set()
    for decision in decisions:
        post_id = _post_id(decision)
        if not post_id or post_id in seen:
            continue
        seen.add(post_id)
        payload = decision.get("decision", {})
        if not payload.get("rescue_csm_eligible"):
            continue
        experiencer = payload.get("rescued_experiencer_label")
        content = payload.get("rescued_content_function")
        if experiencer not in {"E1", "E2"} or content not in {"C1", "C2"}:
            continue
        candidate = candidates.get(post_id)
        if candidate is None:
            raise SystemExit(f"Decision post_id not found in candidates: {post_id}")
        rescued.append((candidate, decision))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.shard_count < 1:
        raise SystemExit("--shard-count must be >= 1")
    input_path = out_dir / "rescued_input.jsonl"
    _write_rescued_input(input_path, rescued)
    if args.shard_count == 1:
        classified_paths = [_prepare_one_output_dir(out_dir, rescued)]
    else:
        classified_paths = []
        for shard_index in range(args.shard_count):
            shard_rows = _select_shard(rescued, shard_count=args.shard_count, shard_index=shard_index)
            shard_dir = out_dir / f"shard_{shard_index:03d}"
            classified_paths.append(_prepare_one_output_dir(shard_dir, shard_rows))

    manifest = {
        "candidates": args.candidates,
        "decisions": args.decisions,
        "out_dir": str(out_dir),
        "rescued_rows": len(rescued),
        "rescued_input": str(input_path),
        "shard_count": args.shard_count,
        "classified_checkpoints": [str(path) for path in classified_paths],
    }
    (out_dir / "rescue_preparation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _write_rescued_input(path: Path, rescued: list[tuple[dict[str, Any], dict[str, Any]]]) -> None:
    path.write_text("", encoding="utf-8")
    with path.open("a", encoding="utf-8") as input_file:
        for candidate, _decision in rescued:
            post = _source_post(candidate)
            input_file.write(json.dumps(post.model_dump(mode="json"), ensure_ascii=False) + "\n")


def _prepare_one_output_dir(out_dir: Path, rescued: list[tuple[dict[str, Any], dict[str, Any]]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    classified_path = checkpoint_dir / "classified.jsonl"
    for path in [
        classified_path,
        out_dir / "annotations.jsonl",
        out_dir / "errors.jsonl",
        out_dir / "retrieval_trace.jsonl",
        checkpoint_dir / "extracted.jsonl",
    ]:
        path.write_text("", encoding="utf-8")
    with classified_path.open("a", encoding="utf-8") as classified_file:
        for candidate, decision in rescued:
            post = _source_post(candidate)
            output = _rescued_checkpoint_output(post, candidate, decision)
            _write_output_line(classified_file, output)
    return classified_path


def _select_shard(
    rows: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    shard_count: int,
    shard_index: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    start = len(rows) * shard_index // shard_count
    end = len(rows) * (shard_index + 1) // shard_count
    return rows[start:end]


def _rescued_checkpoint_output(post: SourcePost, candidate: dict[str, Any], decision: dict[str, Any]) -> Any:
    from dental_ai.pipeline import PipelineOutput

    payload = decision["decision"]
    result = ExtractionResult.empty_for_post(post).model_copy(
        update={
            "relevance_label": RelevanceLabel.R1,
            "experiencer_label": ExperiencerLabel(payload["rescued_experiencer_label"]),
            "content_function": ContentFunctionLabel(payload["rescued_content_function"]),
        }
    )
    validation = validate_hierarchical_result(result, post)
    trace = PipelineTrace(
        stages=["first_pass_classification", "csm_rescue_gate"],
        validation=validation,
        raw_labels={
            "first_pass_relevance_label": candidate.get("first_pass_relevance_label"),
            "first_pass_experiencer_label": candidate.get("first_pass_experiencer_label"),
            "first_pass_content_function": candidate.get("first_pass_content_function"),
            "rescued_experiencer_label": payload["rescued_experiencer_label"],
            "rescued_content_function": payload["rescued_content_function"],
        },
        postprocessing_rules=[
            {
                "rule": "csm_rescue.high_recall_gate",
                "field": "csm_eligibility",
                "before": _first_pass_label(candidate),
                "after": _label_string(result),
                "reason": str(payload.get("reason") or "")[:500],
                "evidence": str(payload.get("rescue_evidence") or "")[:120],
            }
        ],
    )
    return PipelineOutput(result=result, trace=trace)


def _load_decisions(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        decision_path = path / "rescue_decisions.jsonl" if path.is_dir() else path
        rows.extend(_read_jsonl(decision_path))
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _source_post(row: dict[str, Any]) -> SourcePost:
    allowed = set(SourcePost.model_fields)
    payload = {key: value for key, value in row.items() if key in allowed}
    payload["post_id"] = _post_id(row)
    return SourcePost.model_validate(payload)


def _post_id(row: dict[str, Any]) -> str:
    return str(row.get("post_id") or row.get("record_id") or "").strip()


def _first_pass_label(row: dict[str, Any]) -> str:
    return "|".join(
        str(row.get(field))
        for field in [
            "first_pass_relevance_label",
            "first_pass_experiencer_label",
            "first_pass_content_function",
        ]
    )


def _label_string(result: ExtractionResult) -> str:
    return "|".join(
        [
            result.relevance_label.value if result.relevance_label else "None",
            result.experiencer_label.value if result.experiencer_label else "None",
            result.content_function.value if result.content_function else "None",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
