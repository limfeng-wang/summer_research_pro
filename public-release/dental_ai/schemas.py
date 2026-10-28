"""Pydantic schemas for evidence-grounded dental pain narrative extraction.

These models implement the current project contract:

- upstream corpus labels: relevance, experiencer, and content function
- five CSM domains for first-person illness-narrative extraction
- source-language evidence spans preserved exactly
- annotator working-language surface labels allowed and explicitly marked
- English normalized concepts for cross-language analysis

IO, LLM calls, normalization, judging, metrics, and network analysis belong in
separate modules.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Iterable

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


class Country(StrEnum):
    """Country/platform corpus."""

    CHI = "CHI"
    JPN = "JPN"
    KOR = "KOR"


class Language(StrEnum):
    """Original source language."""

    ZH = "zh"
    JA = "ja"
    KO = "ko"


class RelevanceLabel(StrEnum):
    """Toothache relevance label for upstream filtering."""

    R1 = "R1"
    R0 = "R0"
    RU = "RU"


class ExperiencerLabel(StrEnum):
    """Whose toothache-related experience is described."""

    E1 = "E1"
    E2 = "E2"
    E3 = "E3"


class ContentFunctionLabel(StrEnum):
    """Primary communicative function of a toothache-related post."""

    C1 = "C1"
    C2 = "C2"
    C3 = "C3"
    C4 = "C4"
    C5 = "C5"


class CSMDomain(StrEnum):
    """Five CSM domains used for illness-narrative concept extraction."""

    PERCEIVED_CAUSE = "Perceived Cause"
    SYMPTOM_DESCRIPTION = "Symptom Description"
    PERCEIVED_CONSEQUENCES = "Perceived Consequences"
    COPING_AND_MANAGEMENT = "Coping and Management"
    EMOTIONAL_EXPRESSION = "Emotional Expression"


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
    """Verification outcome from the LLM-as-Judge stage or accepted human gold."""

    ACCEPT = "accept"
    REVISE = "revise"
    REJECT = "reject"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


class SourcePost(BaseModel):
    """Minimal source post representation used by the pipeline."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, populate_by_name=True)

    post_id: str = Field(validation_alias=AliasChoices("post_id", "record_id"))
    country: Country
    language: Language
    platform: str = ""
    original_title: str = ""
    original_text: str = Field(default="", validation_alias=AliasChoices("original_text", "text"))
    text_clean: str = ""
    text_length_chars: int | None = None
    text_fingerprint: str = ""
    analysis_text_en: str = ""
    source_file: str = ""
    source_sheet: str = ""
    source_row_number: int | None = None
    length_band: str = ""
    country_length_tertile_cutpoints: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "post_id",
        "platform",
        "original_title",
        "original_text",
        "text_clean",
        "text_fingerprint",
        "analysis_text_en",
        "source_file",
        "source_sheet",
        "length_band",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def _validate_has_text(self) -> SourcePost:
        if not self.combined_source_text:
            raise ValueError("SourcePost requires original_title or original_text")
        return self

    @property
    def combined_source_text(self) -> str:
        """Source text used for prompting and evidence containment checks."""

        if self.text_clean:
            return self.text_clean
        return "\n".join(part for part in [self.original_title, self.original_text] if part).strip()

    @property
    def text(self) -> str:
        """Backward-compatible alias for the source text used by old helpers."""

        return self.combined_source_text


class NarrativeUnit(BaseModel):
    """One evidence-backed CSM claim extracted from a post."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, populate_by_name=True)

    unit_id: str = ""
    domain: CSMDomain
    evidence_span_original: str = Field(
        min_length=1,
        validation_alias=AliasChoices("evidence_span_original", "evidence_span"),
    )
    surface_text_working: str = Field(
        min_length=1,
        validation_alias=AliasChoices("surface_text_working", "surface_text_original", "surface_text"),
    )
    working_language: Language = Language.ZH
    normalized_concept_en: str = Field(
        default="",
        validation_alias=AliasChoices("normalized_concept_en", "normalized_concept"),
    )
    concept_status: ConceptStatus
    support_type: SupportType
    assertion: AssertionStatus
    temporality: Temporality = Temporality.UNKNOWN
    sentiment_or_outcome: SentimentOrOutcome = SentimentOrOutcome.UNKNOWN
    confidence: float = Field(ge=0.0, le=1.0)
    judge_verdict: JudgeVerdict

    @field_validator(
        "unit_id",
        "evidence_span_original",
        "surface_text_working",
        "normalized_concept_en",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def _validate_normalized_concept(self) -> NarrativeUnit:
        if self.concept_status != ConceptStatus.UNMAPPED and not self.normalized_concept_en:
            raise ValueError("normalized_concept_en is required unless concept_status is 'unmapped'")
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

        return self.evidence_span_original in source_text


class ExtractionResult(BaseModel):
    """Structured CSM extraction output for one post."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, populate_by_name=True)

    post_id: str = Field(validation_alias=AliasChoices("post_id", "record_id"))
    country: Country
    language: Language
    platform: str = ""
    original_title: str = ""
    original_text: str = ""
    text_clean: str = ""
    text_length_chars: int | None = None
    text_fingerprint: str = ""
    analysis_text_en: str = ""
    source_file: str = ""
    source_sheet: str = ""
    source_row_number: int | None = None
    length_band: str = ""
    country_length_tertile_cutpoints: dict[str, Any] = Field(default_factory=dict)
    relevance_label: RelevanceLabel | None = None
    experiencer_label: ExperiencerLabel | None = None
    content_function: ContentFunctionLabel | None = None
    units: list[NarrativeUnit] = Field(default_factory=list)

    @field_validator(
        "post_id",
        "platform",
        "original_title",
        "original_text",
        "text_clean",
        "text_fingerprint",
        "analysis_text_en",
        "source_file",
        "source_sheet",
        "length_band",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: object) -> object:
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

        return cls(
            post_id=post.post_id,
            country=post.country,
            language=post.language,
            platform=post.platform,
            original_title=post.original_title,
            original_text=post.original_text,
            text_clean=post.text_clean,
            text_length_chars=post.text_length_chars,
            text_fingerprint=post.text_fingerprint,
            analysis_text_en=post.analysis_text_en,
            source_file=post.source_file,
            source_sheet=post.source_sheet,
            source_row_number=post.source_row_number,
            length_band=post.length_band,
            country_length_tertile_cutpoints=post.country_length_tertile_cutpoints,
            units=[],
        )

    @property
    def combined_source_text(self) -> str:
        """Title and body joined for evidence containment checks."""

        if self.text_clean:
            return self.text_clean
        return "\n".join(part for part in [self.original_title, self.original_text] if part).strip()

    @property
    def accepted_units(self) -> list[NarrativeUnit]:
        """Units accepted by the judge and eligible for downstream analysis."""

        return [unit for unit in self.units if unit.is_accepted]

    @property
    def domain_labels(self) -> dict[CSMDomain, int]:
        """Derive post-level CSM labels from accepted narrative units."""

        accepted_domains = {unit.domain for unit in self.accepted_units}
        return {domain: int(domain in accepted_domains) for domain in CSMDomain}

    @property
    def is_primary_csm_candidate(self) -> bool:
        """Whether the post belongs to the primary self-report CSM corpus."""

        return (
            self.relevance_label == RelevanceLabel.R1
            and self.experiencer_label == ExperiencerLabel.E1
            and self.content_function in {ContentFunctionLabel.C1, ContentFunctionLabel.C2}
        )

    @property
    def is_proxy_csm_candidate(self) -> bool:
        """Whether the post belongs to the secondary proxy-narrative corpus."""

        return (
            self.relevance_label == RelevanceLabel.R1
            and self.experiencer_label == ExperiencerLabel.E2
            and self.content_function in {ContentFunctionLabel.C1, ContentFunctionLabel.C2}
        )

    def validate_against_post(self, post: SourcePost) -> None:
        """Validate metadata and evidence spans against the original source post."""

        if self.post_id != post.post_id:
            raise ValueError(f"post_id mismatch: extraction={self.post_id!r}, post={post.post_id!r}")
        if self.country != post.country:
            raise ValueError(f"country mismatch: extraction={self.country!r}, post={post.country!r}")
        if self.language != post.language:
            raise ValueError(f"language mismatch: extraction={self.language!r}, post={post.language!r}")
        self.validate_evidence_spans(post.combined_source_text)

    def validate_evidence_spans(self, source_text: str | None = None) -> None:
        """Validate all evidence spans against source text.

        If `source_text` is omitted, this method uses the result's stored
        original title and text.
        """

        source = source_text if source_text is not None else self.combined_source_text
        missing_spans = [
            unit.evidence_span_original for unit in self.units if not unit.is_grounded_in(source)
        ]
        if missing_spans:
            raise ValueError(f"evidence_span_original not found in source text: {missing_spans!r}")

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
                    "relevance_label": result.relevance_label.value if result.relevance_label else "",
                    "experiencer_label": result.experiencer_label.value if result.experiencer_label else "",
                    "content_function": result.content_function.value if result.content_function else "",
                    "unit_id": unit.unit_id,
                    "domain": unit.domain.value,
                    "evidence_span_original": unit.evidence_span_original,
                    "surface_text_working": unit.surface_text_working,
                    "working_language": unit.working_language.value,
                    "normalized_concept_en": unit.normalized_concept_en,
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
    "ContentFunctionLabel",
    "Country",
    "ExperiencerLabel",
    "ExtractionResult",
    "JudgeVerdict",
    "Language",
    "NarrativeUnit",
    "RelevanceLabel",
    "SentimentOrOutcome",
    "SourcePost",
    "SupportType",
    "Temporality",
    "ValidationError",
    "flatten_units",
]
