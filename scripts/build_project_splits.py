#!/usr/bin/env python
"""Build leak-checked project data splits.

This script treats the project data roles as mutually exclusive:
- classification few-shot/development gold
- CSM few-shot gold
- held-out manual eval/test
- main raw corpus for automated annotation
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from dental_ai.classification_gold import (
    canonical_post_id,
    classification_gold_summary,
    load_classification_gold_xlsx,
    source_text_fingerprint,
)
from dental_ai.goldset import load_csm_gold_json


DEFAULT_CLASSIFICATION_GOLD_XLSX = Path("data/三语分类金标准-Law.xlsx")
DEFAULT_CLASSIFICATION_GOLD_JSONL = Path("data/classification_gold_172.jsonl")
DEFAULT_CSM_GOLD = Path("data/csm_gold_50E1_10E2.json")
DEFAULT_EVAL = Path("data/raw_eval_holdout_150_no_gold.jsonl")
DEFAULT_EVAL_OUT = Path("data/raw_eval_holdout_150_no_gold.jsonl")
DEFAULT_MAIN_IN = Path("data/raw_main_llm_input_no_gold.jsonl")
DEFAULT_MAIN_OUT = Path("data/raw_main_llm_input_no_gold.jsonl")
DEFAULT_MANIFEST = Path("data/project_split_manifest.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export classification gold and rebuild leak-free main split.")
    parser.add_argument("--classification-gold-xlsx", default=str(DEFAULT_CLASSIFICATION_GOLD_XLSX))
    parser.add_argument("--classification-gold-jsonl", default=str(DEFAULT_CLASSIFICATION_GOLD_JSONL))
    parser.add_argument("--csm-gold", default=str(DEFAULT_CSM_GOLD))
    parser.add_argument("--eval", default=str(DEFAULT_EVAL))
    parser.add_argument("--eval-out", default=str(DEFAULT_EVAL_OUT))
    parser.add_argument("--main-in", default=str(DEFAULT_MAIN_IN))
    parser.add_argument("--main-out", default=str(DEFAULT_MAIN_OUT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args(argv)

    classification = load_classification_gold_xlsx(args.classification_gold_xlsx)
    csm = load_csm_gold_json(args.csm_gold)
    eval_input_path = _existing_input_or_fallback(args.eval, args.eval_out)
    main_input_path = _existing_input_or_fallback(args.main_in, args.main_out)
    eval_rows = _read_jsonl(eval_input_path)
    main_rows = _read_jsonl(main_input_path)
    raw_pool_rows = eval_rows + main_rows

    classification_path = Path(args.classification_gold_jsonl)
    classification_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl((row.to_json() for row in classification), classification_path)

    fewshot_protected = _fewshot_protected_keys(classification, csm)
    repaired_eval_rows, eval_replacements = _repair_eval_holdout(
        eval_rows,
        main_rows,
        protected=fewshot_protected,
    )
    _write_jsonl(repaired_eval_rows, args.eval_out)
    _write_eval_sidecars(repaired_eval_rows, args.eval_out)

    protected = _protected_keys(classification, csm, repaired_eval_rows)
    kept_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, str]] = []

    for row in raw_pool_rows:
        key = _row_key(row)
        reasons = []
        if key["country_canonical_id"] in protected["classification_ids"]:
            reasons.append("classification_gold_id")
        if key["country_canonical_id"] in protected["csm_ids"]:
            reasons.append("csm_gold_id")
        if key["country_canonical_id"] in protected["eval_ids"]:
            reasons.append("eval_holdout_id")
        if key["fingerprint"] in protected["classification_fingerprints"]:
            reasons.append("classification_gold_fingerprint")
        if key["fingerprint"] in protected["csm_fingerprints"]:
            reasons.append("csm_gold_fingerprint")
        if key["fingerprint"] in protected["eval_fingerprints"]:
            reasons.append("eval_holdout_fingerprint")

        if reasons:
            excluded_rows.append(
                {
                    "post_id": str(row.get("record_id") or row.get("post_id")),
                    "country": str(row.get("country", "")),
                    "canonical_id": key["canonical_id"],
                    "reasons": ",".join(sorted(set(reasons))),
                }
            )
        else:
            kept_rows.append(row)

    _write_jsonl(kept_rows, args.main_out)

    classification_keyset = _classification_keys(classification)
    csm_keyset = _csm_keys(csm)
    fewshot_keyset = {
        "ids": classification_keyset["ids"] | csm_keyset["ids"],
        "fingerprints": classification_keyset["fingerprints"] | csm_keyset["fingerprints"],
    }
    gold_internal_overlap = {
        "id_overlap_count": len(classification_keyset["ids"] & csm_keyset["ids"]),
        "fingerprint_overlap_count": len(classification_keyset["fingerprints"] & csm_keyset["fingerprints"]),
    }
    post_audit = _audit_disjointness(
        {
            "fewshot_gold_union": fewshot_keyset,
            "eval_holdout": _source_rows_keys(repaired_eval_rows),
            "main_no_gold": _source_rows_keys(kept_rows),
        }
    )

    if post_audit["overlaps"]:
        raise SystemExit(f"Split leakage remains after rebuild: {json.dumps(post_audit, ensure_ascii=False)}")

    manifest = {
        "classification_gold_xlsx": str(args.classification_gold_xlsx),
        "classification_gold_jsonl": str(classification_path),
        "csm_gold": str(args.csm_gold),
        "eval_holdout_original": str(eval_input_path),
        "eval_holdout_no_gold": str(args.eval_out),
        "main_input_original": str(main_input_path),
        "main_input_no_gold": str(args.main_out),
        "classification_gold": classification_gold_summary(classification),
        "csm_gold_posts": len(csm),
        "eval_holdout_original_rows": len(eval_rows),
        "eval_holdout_no_gold_rows": len(repaired_eval_rows),
        "eval_replacements": eval_replacements,
        "raw_pool_rows": len(raw_pool_rows),
        "main_original_rows": len(main_rows),
        "main_no_gold_rows": len(kept_rows),
        "main_excluded_rows": len(excluded_rows),
        "main_exclusion_reasons": dict(Counter(reason for row in excluded_rows for reason in row["reasons"].split(","))),
        "excluded_main_rows": excluded_rows,
        "gold_internal_overlap_allowed": gold_internal_overlap,
        "disjointness_audit": post_audit,
    }
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in [
        "classification_gold",
        "csm_gold_posts",
        "eval_holdout_original_rows",
        "eval_holdout_no_gold_rows",
        "raw_pool_rows",
        "main_original_rows",
        "main_no_gold_rows",
        "main_excluded_rows",
        "main_exclusion_reasons",
    ]}, ensure_ascii=False, indent=2))
    return 0


def _fewshot_protected_keys(classification: Iterable[Any], csm: Iterable[Any]) -> dict[str, set[str]]:
    classification_rows = list(classification)
    csm_rows = list(csm)
    return {
        "classification_ids": {f"{row.country.value}:{row.canonical_id}" for row in classification_rows},
        "classification_fingerprints": {row.text_fingerprint for row in classification_rows},
        "csm_ids": {f"{row.country.value}:{canonical_post_id(row.post_id)}" for row in csm_rows},
        "csm_fingerprints": {source_text_fingerprint(row.original_title, row.original_text) for row in csm_rows},
    }


def _protected_keys(classification: Iterable[Any], csm: Iterable[Any], eval_rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    protected = _fewshot_protected_keys(classification, csm)
    protected.update(
        {
        "eval_ids": {_row_key(row)["country_canonical_id"] for row in eval_rows},
        "eval_fingerprints": {_row_key(row)["fingerprint"] for row in eval_rows},
        }
    )
    return protected


def _repair_eval_holdout(
    eval_rows: list[dict[str, Any]],
    main_rows: list[dict[str, Any]],
    *,
    protected: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Replace eval rows that overlap few-shot gold with same-stratum rows."""

    protected_ids = protected["classification_ids"] | protected["csm_ids"]
    protected_fingerprints = protected["classification_fingerprints"] | protected["csm_fingerprints"]
    kept_eval: list[dict[str, Any]] = []
    leaked_eval: list[dict[str, Any]] = []
    for row in eval_rows:
        key = _row_key(row)
        if key["country_canonical_id"] in protected_ids or key["fingerprint"] in protected_fingerprints:
            leaked_eval.append(row)
        else:
            kept_eval.append(row)

    kept_eval_ids = {_row_key(row)["country_canonical_id"] for row in kept_eval}
    kept_eval_fingerprints = {_row_key(row)["fingerprint"] for row in kept_eval}
    repaired = list(kept_eval)
    replacements: list[dict[str, str]] = []

    for leaked in leaked_eval:
        leaked_key = _row_key(leaked)
        country = str(leaked.get("country", ""))
        length_band = str(leaked.get("length_band", ""))
        candidate = None
        for row in sorted(main_rows, key=lambda item: str(item.get("record_id") or item.get("post_id"))):
            key = _row_key(row)
            if str(row.get("country", "")) != country:
                continue
            if str(row.get("length_band", "")) != length_band:
                continue
            if key["country_canonical_id"] in protected_ids or key["fingerprint"] in protected_fingerprints:
                continue
            if key["country_canonical_id"] in kept_eval_ids or key["fingerprint"] in kept_eval_fingerprints:
                continue
            candidate = row
            break
        if candidate is None:
            raise ValueError(
                f"No replacement candidate for leaked eval row {leaked_key['post_id']} "
                f"({country}, {length_band})"
            )

        candidate_key = _row_key(candidate)
        repaired.append(candidate)
        kept_eval_ids.add(candidate_key["country_canonical_id"])
        kept_eval_fingerprints.add(candidate_key["fingerprint"])
        replacements.append(
            {
                "removed_post_id": leaked_key["post_id"],
                "removed_country": country,
                "removed_length_band": length_band,
                "replacement_post_id": candidate_key["post_id"],
                "replacement_country": str(candidate.get("country", "")),
                "replacement_length_band": str(candidate.get("length_band", "")),
            }
        )

    return repaired, replacements


def _audit_disjointness(named_keys: dict[str, dict[str, set[str]]]) -> dict[str, object]:
    overlaps: list[dict[str, object]] = []
    names = list(named_keys)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            id_overlap = sorted(named_keys[left]["ids"] & named_keys[right]["ids"])
            fingerprint_overlap = sorted(named_keys[left]["fingerprints"] & named_keys[right]["fingerprints"])
            if id_overlap or fingerprint_overlap:
                overlaps.append(
                    {
                        "left": left,
                        "right": right,
                        "id_overlap_count": len(id_overlap),
                        "fingerprint_overlap_count": len(fingerprint_overlap),
                        "id_overlap_sample": id_overlap[:20],
                        "fingerprint_overlap_sample": fingerprint_overlap[:20],
                    }
                )
    return {"ok": not overlaps, "overlaps": overlaps}


def _classification_keys(records: Iterable[Any]) -> dict[str, set[str]]:
    rows = list(records)
    return {
        "ids": {f"{row.country.value}:{row.canonical_id}" for row in rows},
        "fingerprints": {row.text_fingerprint for row in rows},
    }


def _csm_keys(records: Iterable[Any]) -> dict[str, set[str]]:
    rows = list(records)
    return {
        "ids": {f"{row.country.value}:{canonical_post_id(row.post_id)}" for row in rows},
        "fingerprints": {source_text_fingerprint(row.original_title, row.original_text) for row in rows},
    }


def _source_rows_keys(rows: Iterable[dict[str, Any]]) -> dict[str, set[str]]:
    keys = [_row_key(row) for row in rows]
    return {
        "ids": {key["country_canonical_id"] for key in keys},
        "fingerprints": {key["fingerprint"] for key in keys},
    }


def _row_key(row: dict[str, Any]) -> dict[str, str]:
    post_id = str(row.get("record_id") or row.get("post_id"))
    country = str(row.get("country", ""))
    canonical_id = canonical_post_id(post_id)
    text = str(row.get("text_clean") or "").strip()
    title = str(row.get("original_title") or "").strip()
    original_text = str(row.get("original_text") or row.get("text") or "").strip()
    fingerprint = str(row.get("text_fingerprint") or "").strip()
    if not fingerprint:
        fingerprint = source_text_fingerprint("", text) if text else source_text_fingerprint(title, original_text)
    return {
        "post_id": post_id,
        "country": country,
        "canonical_id": canonical_id,
        "country_canonical_id": f"{country}:{canonical_id}",
        "fingerprint": fingerprint,
    }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _existing_input_or_fallback(path: str | Path, fallback: str | Path) -> Path:
    primary = Path(path)
    if primary.exists():
        return primary
    fallback_path = Path(fallback)
    if fallback_path.exists():
        return fallback_path
    raise FileNotFoundError(f"Neither {primary} nor fallback {fallback_path} exists")


def _write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_eval_sidecars(rows: list[dict[str, Any]], eval_out: str | Path) -> None:
    base = Path(eval_out)
    ids_path = base.with_name(base.stem + "_ids.csv")
    xlsx_path = base.with_suffix(".xlsx")
    manifest_path = base.with_name(base.stem + "_manifest.json")
    ids_path.write_text(
        "record_id,country,language,length_band,text_fingerprint\n"
        + "\n".join(
            ",".join(
                str(row.get(key, ""))
                for key in ["record_id", "country", "language", "length_band", "text_fingerprint"]
            )
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        import pandas as pd

        pd.DataFrame(rows).to_excel(xlsx_path, index=False)
    except Exception:
        pass
    manifest = {
        "rows": len(rows),
        "countries": dict(Counter(str(row.get("country", "")) for row in rows)),
        "length_bands": dict(Counter(str(row.get("length_band", "")) for row in rows)),
        "country_length_bands": dict(
            Counter(f"{row.get('country', '')}:{row.get('length_band', '')}" for row in rows)
        ),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
