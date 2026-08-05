"""Deterministic validation for hierarchical dental pain annotations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from pydantic import ValidationError

from dental_ai.schemas import (
    CSMDomain,
    ContentFunctionLabel,
    ExperiencerLabel,
    ExtractionResult,
    RelevanceLabel,
    SourcePost,
)


@dataclass(frozen=True)
class ValidationIssue:
    """One deterministic validation issue."""

    code: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    """Validation result with machine-readable issues."""

    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    @classmethod
    def pass_(cls) -> "ValidationReport":
        return cls(ok=True, issues=[])

    @classmethod
    def fail(cls, issues: Iterable[ValidationIssue]) -> "ValidationReport":
        issue_list = list(issues)
        return cls(ok=not issue_list, issues=issue_list)


def validate_hierarchical_result(result: ExtractionResult, post: SourcePost) -> ValidationReport:
    """Validate hierarchy, schema-compatible labels, and source evidence spans."""

    issues: list[ValidationIssue] = []

    if result.post_id != post.post_id:
        issues.append(ValidationIssue("post_id_mismatch", "result post_id does not match source post"))
    if result.country != post.country:
        issues.append(ValidationIssue("country_mismatch", "result country does not match source post"))
    if result.language != post.language:
        issues.append(ValidationIssue("language_mismatch", "result language does not match source post"))

    if result.relevance_label is None:
        issues.append(ValidationIssue("missing_relevance", "relevance_label is required"))
    elif result.relevance_label in {RelevanceLabel.R0, RelevanceLabel.RU}:
        if result.experiencer_label is not None:
            issues.append(ValidationIssue("downstream_after_non_r1", "non-R1 rows must not have experiencer labels"))
        if result.content_function is not None:
            issues.append(ValidationIssue("downstream_after_non_r1", "non-R1 rows must not have content-function labels"))
        if result.units:
            issues.append(ValidationIssue("units_after_non_r1", "non-R1 rows must not have CSM units"))
    elif result.relevance_label == RelevanceLabel.R1:
        if result.experiencer_label is None:
            issues.append(ValidationIssue("missing_experiencer", "R1 rows require experiencer_label"))
        if result.content_function is None:
            issues.append(ValidationIssue("missing_content_function", "R1 rows require content_function"))

    if result.units and not _is_csm_extraction_candidate(result):
        issues.append(
            ValidationIssue(
                "units_for_ineligible_row",
                "CSM units are only expected for R1 + E1/E2 + C1/C2 rows",
            )
        )

    allowed_domains = {domain.value for domain in CSMDomain}
    for unit in result.units:
        if unit.domain.value not in allowed_domains:
            issues.append(ValidationIssue("invalid_csm_domain", f"invalid CSM domain: {unit.domain.value}"))
        if not unit.evidence_span_original:
            issues.append(ValidationIssue("missing_evidence_span", "unit evidence_span_original is required"))
        elif unit.evidence_span_original not in post.combined_source_text:
            issues.append(
                ValidationIssue(
                    "evidence_span_not_found",
                    f"evidence span not found in source text: {unit.evidence_span_original!r}",
                )
            )

    return ValidationReport.fail(issues)


def validate_extraction_payload(payload: object, post: SourcePost) -> tuple[ExtractionResult | None, ValidationReport]:
    """Validate a raw model payload into an ExtractionResult and deterministic report."""

    try:
        result = ExtractionResult.model_validate(payload)
    except ValidationError as exc:
        return None, ValidationReport.fail([ValidationIssue("schema_error", str(exc))])

    return result, validate_hierarchical_result(result, post)


def _is_csm_extraction_candidate(result: ExtractionResult) -> bool:
    return (
        result.relevance_label == RelevanceLabel.R1
        and result.experiencer_label in {ExperiencerLabel.E1, ExperiencerLabel.E2}
        and result.content_function in {ContentFunctionLabel.C1, ContentFunctionLabel.C2}
    )


__all__ = [
    "ValidationIssue",
    "ValidationReport",
    "validate_extraction_payload",
    "validate_hierarchical_result",
]
