"""Load and summarize human-annotated CSM Level-2 gold data."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from dental_ai.schemas import (
    ContentFunctionLabel,
    ExperiencerLabel,
    ExtractionResult,
    RelevanceLabel,
)


def load_csm_gold_json(path: str | Path, *, validate_spans: bool = True) -> list[ExtractionResult]:
    """Load a CSM Level-2 gold JSON file.

    The expected file shape is a list of post-level objects matching
    ExtractionResult. Current human files may use `surface_text_original`;
    the schema accepts it as an alias for `surface_text_working`.
    """

    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")

    results: list[ExtractionResult] = []
    for index, item in enumerate(data):
        try:
            result = ExtractionResult.model_validate(item)
            if validate_spans:
                result.validate_evidence_spans()
        except Exception as exc:
            raise ValueError(f"Invalid CSM gold record in {path} at index {index}: {exc}") from exc
        results.append(result)
    return results


def load_csm_gold_jsons(paths: Iterable[str | Path], *, validate_spans: bool = True) -> list[ExtractionResult]:
    """Load and concatenate multiple CSM Level-2 gold JSON files."""

    results: list[ExtractionResult] = []
    seen_post_ids: set[str] = set()
    duplicates: list[str] = []

    for path in paths:
        for result in load_csm_gold_json(path, validate_spans=validate_spans):
            if result.post_id in seen_post_ids:
                duplicates.append(result.post_id)
            seen_post_ids.add(result.post_id)
            results.append(result)

    if duplicates:
        raise ValueError(f"Duplicate post_id values across gold files: {duplicates!r}")
    return results


def primary_csm_results(results: Iterable[ExtractionResult]) -> list[ExtractionResult]:
    """Return R1 + E1 + (C1 or C2) records."""

    return [result for result in results if result.is_primary_csm_candidate]


def proxy_csm_results(results: Iterable[ExtractionResult]) -> list[ExtractionResult]:
    """Return R1 + E2 + (C1 or C2) records."""

    return [result for result in results if result.is_proxy_csm_candidate]


def summarize_csm_gold(results: Iterable[ExtractionResult]) -> dict[str, object]:
    """Return count summaries for a CSM Level-2 gold set."""

    result_list = list(results)
    unit_count = sum(len(result.units) for result in result_list)
    accepted_units = [unit for result in result_list for unit in result.accepted_units]
    span_ok = 0
    for result in result_list:
        source = result.combined_source_text
        span_ok += sum(unit.is_grounded_in(source) for unit in result.units)

    return {
        "posts": len(result_list),
        "units": unit_count,
        "accepted_units": len(accepted_units),
        "span_exact_matches": span_ok,
        "countries": _counter_values(result.country for result in result_list),
        "languages": _counter_values(result.language for result in result_list),
        "relevance_labels": _counter_optional(result.relevance_label for result in result_list),
        "experiencer_labels": _counter_optional(result.experiencer_label for result in result_list),
        "content_functions": _counter_optional(result.content_function for result in result_list),
        "domains": _counter_values(unit.domain for result in result_list for unit in result.units),
        "accepted_domains": _counter_values(unit.domain for unit in accepted_units),
        "concept_statuses": _counter_values(unit.concept_status for result in result_list for unit in result.units),
        "support_types": _counter_values(unit.support_type for result in result_list for unit in result.units),
        "assertions": _counter_values(unit.assertion for result in result_list for unit in result.units),
        "judge_verdicts": _counter_values(unit.judge_verdict for result in result_list for unit in result.units),
        "primary_posts": len(primary_csm_results(result_list)),
        "proxy_posts": len(proxy_csm_results(result_list)),
    }


def assert_ready_for_rag_seed(
    results: Iterable[ExtractionResult],
    *,
    min_primary_posts: int = 1,
    min_proxy_posts: int = 0,
) -> None:
    """Validate basic requirements for a RAG/few-shot seed set.

    The project seed may include both primary self-report rows and proxy E2
    rows. Primary rows drive the main CSM analysis; proxy rows support
    ablation and boundary handling.
    """

    result_list = list(results)
    primary = primary_csm_results(result_list)
    proxy = proxy_csm_results(result_list)
    if len(primary) < min_primary_posts:
        raise ValueError(
            f"Not enough primary CSM records: found {len(primary)}, required {min_primary_posts}"
        )
    if len(proxy) < min_proxy_posts:
        raise ValueError(f"Not enough proxy CSM records: found {len(proxy)}, required {min_proxy_posts}")

    invalid = [
        result.post_id
        for result in primary + proxy
        if result.relevance_label != RelevanceLabel.R1
        or result.experiencer_label not in {ExperiencerLabel.E1, ExperiencerLabel.E2}
        or result.content_function not in {ContentFunctionLabel.C1, ContentFunctionLabel.C2}
    ]
    if invalid:
        raise ValueError(f"Invalid RAG seed CSM records: {invalid!r}")

    for result in primary + proxy:
        result.validate_evidence_spans()


def _counter_values(values: Iterable[object]) -> dict[str, int]:
    return dict(Counter(value.value for value in values))


def _counter_optional(values: Iterable[object | None]) -> dict[str, int]:
    return dict(Counter(value.value if value is not None else "" for value in values))


__all__ = [
    "assert_ready_for_rag_seed",
    "load_csm_gold_json",
    "load_csm_gold_jsons",
    "primary_csm_results",
    "proxy_csm_results",
    "summarize_csm_gold",
]
