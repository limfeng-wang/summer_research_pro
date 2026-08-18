#!/usr/bin/env python3
"""Run the high-recall CSM rescue classifier on candidate rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dental_ai.local_models import LocalCSMRescueClassifier, local_lm_for_role
from dental_ai.model_config import load_model_stack_config
from dental_ai.schemas import SourcePost


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Candidate JSONL from build_csm_rescue_candidates.py")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default="configs/model_stack.yaml")
    parser.add_argument("--models-root", default="/hdd-storage/lawrencelcty/huggingface/models")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    candidates = _read_jsonl(Path(args.input))
    if args.limit:
        candidates = candidates[: args.limit]
    candidates = _select_shard(candidates, shard_count=args.shard_count, shard_index=args.shard_index)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = out_dir / "rescue_decisions.jsonl"
    errors_path = out_dir / "errors.jsonl"
    if not args.resume:
        decisions_path.write_text("", encoding="utf-8")
        errors_path.write_text("", encoding="utf-8")
    completed = _load_completed_decision_ids(decisions_path) if args.resume else set()

    stack = load_model_stack_config(args.config)
    lm = local_lm_for_role(stack, "classifier", models_root=args.models_root)
    classifier = LocalCSMRescueClassifier(lm)
    attempted = 0
    rescued = 0
    errors = 0
    try:
        with decisions_path.open("a", encoding="utf-8") as decisions_file, errors_path.open(
            "a",
            encoding="utf-8",
        ) as errors_file:
            for index, row in enumerate(_progress(candidates, desc="csm-rescue"), start=1):
                post_id = _post_id(row)
                if post_id in completed:
                    continue
                attempted += 1
                try:
                    post = _source_post(row)
                    first_pass = {
                        "relevance_label": row.get("first_pass_relevance_label"),
                        "experiencer_label": row.get("first_pass_experiencer_label"),
                        "content_function": row.get("first_pass_content_function"),
                        "candidate_reasons": row.get("rescue_candidate_reasons") or [],
                    }
                    decision = classifier.classify(post, first_pass=first_pass)
                    if decision["rescue_csm_eligible"]:
                        rescued += 1
                    payload = {
                        "post_id": post.post_id,
                        "country": post.country.value,
                        "language": post.language.value,
                        "first_pass": first_pass,
                        "decision": {key: value for key, value in decision.items() if key != "raw_text"},
                        "raw_model_text": decision.get("raw_text", ""),
                    }
                    decisions_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    decisions_file.flush()
                except Exception as exc:
                    errors += 1
                    errors_file.write(
                        json.dumps(
                            {
                                "index": index,
                                "post_id": post_id,
                                "country": row.get("country"),
                                "language": row.get("language"),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    errors_file.flush()
    finally:
        lm.close()

    all_decisions = _read_jsonl(decisions_path)
    manifest = {
        "input": args.input,
        "out_dir": str(out_dir),
        "config": args.config,
        "models_root": args.models_root,
        "resume": args.resume,
        "limit": args.limit,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "candidate_rows_in_shard": len(candidates),
        "attempted_this_run": attempted,
        "errors_this_run": errors,
        "decision_rows_total": len(all_decisions),
        "rescued_rows_total": sum(
            1 for row in all_decisions if row.get("decision", {}).get("rescue_csm_eligible")
        ),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def _load_completed_decision_ids(path: Path) -> set[str]:
    return {_post_id(row) for row in _read_jsonl(path) if _post_id(row)}


def _source_post(row: dict[str, Any]) -> SourcePost:
    allowed = set(SourcePost.model_fields)
    payload = {key: value for key, value in row.items() if key in allowed}
    payload["post_id"] = _post_id(row)
    return SourcePost.model_validate(payload)


def _post_id(row: dict[str, Any]) -> str:
    return str(row.get("post_id") or row.get("record_id") or "").strip()


def _select_shard(rows: list[dict[str, Any]], *, shard_count: int, shard_index: int) -> list[dict[str, Any]]:
    if shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < shard-count")
    start = len(rows) * shard_index // shard_count
    end = len(rows) * (shard_index + 1) // shard_count
    return rows[start:end]


def _progress(rows: list[dict[str, Any]], *, desc: str):
    try:
        from tqdm.auto import tqdm

        return tqdm(rows, total=len(rows), desc=desc, unit="post")
    except Exception:
        return rows


if __name__ == "__main__":
    raise SystemExit(main())

