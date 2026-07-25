"""Pydantic schemas for evidence-grounded dental pain narrative extraction.

These models implement the contract described in PROJECT.md. They intentionally
cover only the shared data shape and lightweight validation rules; IO, LLM
calls, normalization, judging, and metrics belong in later modules.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class Country(StrEnum):
    """Country/platform context for a source post."""

    CHI = "CHI"
    JPN = "JPN"
    KOR = "KOR"


class Language(StrEnum):
    """Original language of the source post."""

    ZH = "zh"
    JA = "ja"
    KO = "ko"


class CSMDomain(StrEnum):
    """Common-Sense Self-Regulation Model domains used in the project."""

    PERCEIVED_CAUSE = "Perceived Cause"
    SYMPTOM_DESCRIPTION = "Symptom Description"
    PERCEIVED_CONSEQUENCES = "Perceived Consequences"
    COPING_AND_MANAGEMENT = "Coping and Management"
    EMOTIONAL_EXPRESSION = "Emotional Expression"
    SOCIAL_INTERACTION = "Social Interaction"


class ConceptStatus(StrEnum):
    """Whether a normalized concept is already known or newly discovered."""

    EXISTING_DICTIONARY = "existing_dictionary"
    NEW_CANDIDATE = "new_candidate"
    UNMAPPED = "unmapped"


class SupportType(StrEnum):
    """How directly a source span supports the extracted unit."""

    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    UNSUPPORTED = "unsupported"


class AssertionStatus(StrEnum):
    """Whether the extracted concept is affirmed, denied, uncertain, or planned."""

    PRESENT = "present"
    NEGATED = "negated"
    UNCERTAIN = "uncertain"
    PLANNED = "planned"


class Temporality(StrEnum):
    """When the extracted event or experience occurred."""

    PAST = "past"
    CURRENT = "current"
    FUTURE = "future"
    UNKNOWN = "unknown"


class SentimentOrOutcome(StrEnum):
    """Outcome or evaluative signal attached to a unit."""

    EFFECTIVE = "effective"
    INEFFECTIVE = "ineffective"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class JudgeVerdict(StrEnum):
    """Verification outcome from the LLM-as-Judge stage."""

    ACCEPT = "accept"
    REVISE = "revise"
    REJECT = "reject"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


class SourcePost(BaseModel):
    """Minimal source post representation used by the extraction pipeline."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    post_id: str
    country: Country
    language: Language
    text: str = Field(min_length=1)

    @field_validator("post_id", "text", mode="before")
    @classmethod
    def _strip_required_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class NarrativeUnit(BaseModel):
    """One evidence-backed claim extracted from a post."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    unit_id: str = ""
    domain: CSMDomain
    evidence_span: str = Field(min_length=1)
    surface_text: str = Field(min_length=1)
    normalized_concept: str = ""
    concept_status: ConceptStatus
    support_type: SupportType
    assertion: AssertionStatus
    temporality: Temporality = Temporality.UNKNOWN
    sentiment_or_outcome: SentimentOrOutcome = SentimentOrOutcome.UNKNOWN
    confidence: float = Field(ge=0.0, le=1.0)
    judge_verdict: JudgeVerdict

    @field_validator("unit_id", "evidence_span", "surface_text", "normalized_concept", mode="before")
    @classmethod
    def _strip_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def _validate_normalized_concept(self) -> NarrativeUnit:
        if self.concept_status != ConceptStatus.UNMAPPED and not self.normalized_concept:
            raise ValueError("normalized_concept is required unless concept_status is 'unmapped'")
        return self

    @property
    def is_accepted(self) -> bool:
        """Whether this unit is eligible for downstream analysis."""

        return (
            self.judge_verdict == JudgeVerdict.ACCEPT
            and self.support_type != SupportType.UNSUPPORTED
            and self.assertion != AssertionStatus.NEGATED
        )

    def is_grounded_in(self, source_text: str) -> bool:
        """Return True when the evidence span appears verbatim in source text."""

        return self.evidence_span in source_text


class ExtractionResult(BaseModel):
    """Structured extraction output for one post."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    post_id: str
    country: Country
    language: Language
    units: list[NarrativeUnit] = Field(default_factory=list)

    @field_validator("post_id", mode="before")
    @classmethod
    def _strip_post_id(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def _validate_unique_unit_ids(self) -> ExtractionResult:
        unit_ids = [unit.unit_id for unit in self.units if unit.unit_id]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("unit_id values must be unique within a post")
        return self

    @classmethod
    def empty_for_post(cls, post: SourcePost) -> ExtractionResult:
        """Create an empty extraction result that carries source metadata."""

        return cls(post_id=post.post_id, country=post.country, language=post.language, units=[])

    @property
    def accepted_units(self) -> list[NarrativeUnit]:
        """Units accepted by the judge and eligible for downstream analysis."""

        return [unit for unit in self.units if unit.is_accepted]

    @property
    def domain_labels(self) -> dict[CSMDomain, int]:
        """Derive post-level CSM labels from accepted narrative units."""

        accepted_domains = {unit.domain for unit in self.accepted_units}
        return {domain: int(domain in accepted_domains) for domain in CSMDomain}

    def validate_against_post(self, post: SourcePost) -> None:
        """Validate metadata and evidence spans against the original source post.

        Raises:
            ValueError: if post metadata does not match or an evidence span is absent.
        """

        if self.post_id != post.post_id:
            raise ValueError(f"post_id mismatch: extraction={self.post_id!r}, post={post.post_id!r}")
        if self.country != post.country:
            raise ValueError(f"country mismatch: extraction={self.country!r}, post={post.country!r}")
        if self.language != post.language:
            raise ValueError(f"language mismatch: extraction={self.language!r}, post={post.language!r}")

        missing_spans = [
            unit.evidence_span for unit in self.units if not unit.is_grounded_in(post.text)
        ]
        if missing_spans:
            raise ValueError(f"evidence_span not found in source text: {missing_spans!r}")

    def with_assigned_unit_ids(self) -> ExtractionResult:
        """Return a copy with deterministic unit IDs filled where missing."""

        used_ids = {unit.unit_id for unit in self.units if unit.unit_id}
        updated_units: list[NarrativeUnit] = []
        next_index = 1

        for unit in self.units:
            if unit.unit_id:
                updated_units.append(unit)
                continue

            while True:
                candidate = f"{self.post_id}_u{next_index:03d}"
                next_index += 1
                if candidate not in used_ids:
                    used_ids.add(candidate)
                    updated_units.append(unit.model_copy(update={"unit_id": candidate}))
                    break

        return self.model_copy(update={"units": updated_units})


def flatten_units(results: Iterable[ExtractionResult], accepted_only: bool = True) -> list[dict[str, object]]:
    """Flatten extraction results into unit-level records for CSV/Excel export."""

    rows: list[dict[str, object]] = []
    for result in results:
        units = result.accepted_units if accepted_only else result.units
        for unit in units:
            rows.append(
                {
                    "post_id": result.post_id,
                    "country": result.country.value,
                    "language": result.language.value,
                    "unit_id": unit.unit_id,
                    "domain": unit.domain.value,
                    "evidence_span": unit.evidence_span,
                    "surface_text": unit.surface_text,
                    "normalized_concept": unit.normalized_concept,
                    "concept_status": unit.concept_status.value,
                    "support_type": unit.support_type.value,
                    "assertion": unit.assertion.value,
                    "temporality": unit.temporality.value,
                    "sentiment_or_outcome": unit.sentiment_or_outcome.value,
                    "confidence": unit.confidence,
                    "judge_verdict": unit.judge_verdict.value,
                }
            )
    return rows


__all__ = [
    "AssertionStatus",
    "CSMDomain",
    "ConceptStatus",
    "Country",
    "ExtractionResult",
    "JudgeVerdict",
    "Language",
    "NarrativeUnit",
    "SentimentOrOutcome",
    "SourcePost",
    "SupportType",
    "Temporality",
    "ValidationError",
    "flatten_units",
]
