"""Hierarchical annotation orchestration.

This module intentionally contains no Hugging Face loading code. It defines
the pipeline contract so local model clients, mocks, or batch runners can be
plugged in without changing the hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from dental_ai.schemas import (
    ContentFunctionLabel,
    ExperiencerLabel,
    ExtractionResult,
    RelevanceLabel,
    SourcePost,
)
from dental_ai.validate import ValidationReport, validate_hierarchical_result


class RelevanceClassifier(Protocol):
    """Model client for Level-0 relevance screening."""

    def classify_relevance(self, post: SourcePost) -> RelevanceLabel:
        """Return R1, R0, or RU."""


class R1Classifier(Protocol):
    """Model client for R1-only experiencer and content-function labels."""

    def classify_r1(self, post: SourcePost) -> tuple[ExperiencerLabel, ContentFunctionLabel]:
        """Return experiencer and primary content-function labels for an R1 post."""


class CSMExtractor(Protocol):
    """Model client for evidence-grounded CSM extraction."""

    def extract_csm(self, post: SourcePost, seed_examples: list[ExtractionResult]) -> ExtractionResult:
        """Return a post-level ExtractionResult with CSM units."""


class RAGRetriever(Protocol):
    """Retriever over human gold examples."""

    def retrieve(self, post: SourcePost, *, k: int) -> list[ExtractionResult]:
        """Return similar human-adjudicated examples."""


class LLMJudge(Protocol):
    """Judge client for post-hoc validation of extraction output."""

    def judge(self, post: SourcePost, result: ExtractionResult) -> ExtractionResult:
        """Return result with judge verdicts updated or preserved."""


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime-independent hierarchy configuration."""

    extract_proxy_csm: bool = True
    rag_k: int = 5


@dataclass(frozen=True)
class PipelineTrace:
    """Small trace explaining which stages ran for a post."""

    stages: list[str] = field(default_factory=list)
    validation: ValidationReport = field(default_factory=ValidationReport.pass_)


@dataclass(frozen=True)
class PipelineOutput:
    """Final annotation and trace for one post."""

    result: ExtractionResult
    trace: PipelineTrace


@dataclass
class HierarchicalAnnotator:
    """Run the project hierarchy over one post at a time."""

    relevance_classifier: RelevanceClassifier
    r1_classifier: R1Classifier
    csm_extractor: CSMExtractor
    retriever: RAGRetriever
    judge: LLMJudge | None = None
    config: PipelineConfig = field(default_factory=PipelineConfig)

    def annotate_post(self, post: SourcePost) -> PipelineOutput:
        stages: list[str] = []

        relevance = self.relevance_classifier.classify_relevance(post)
        stages.append("relevance")

        result = ExtractionResult.empty_for_post(post).model_copy(update={"relevance_label": relevance})
        if relevance != RelevanceLabel.R1:
            validation = validate_hierarchical_result(result, post)
            return PipelineOutput(result=result, trace=PipelineTrace(stages=stages, validation=validation))

        experiencer, content_function = self.r1_classifier.classify_r1(post)
        stages.append("r1_classification")
        result = result.model_copy(
            update={
                "experiencer_label": experiencer,
                "content_function": content_function,
            }
        )

        if not self._should_extract_csm(experiencer, content_function):
            validation = validate_hierarchical_result(result, post)
            return PipelineOutput(result=result, trace=PipelineTrace(stages=stages, validation=validation))

        seed_examples = self.retriever.retrieve(post, k=self.config.rag_k)
        stages.append("rag_retrieval")
        result = self.csm_extractor.extract_csm(post, seed_examples)
        stages.append("csm_extraction")

        result = result.model_copy(
            update={
                "relevance_label": relevance,
                "experiencer_label": experiencer,
                "content_function": content_function,
            }
        ).with_assigned_unit_ids()

        if self.judge is not None:
            result = self.judge.judge(post, result)
            stages.append("judge")

        validation = validate_hierarchical_result(result, post)
        return PipelineOutput(result=result, trace=PipelineTrace(stages=stages, validation=validation))

    def _should_extract_csm(
        self,
        experiencer: ExperiencerLabel,
        content_function: ContentFunctionLabel,
    ) -> bool:
        if content_function not in {ContentFunctionLabel.C1, ContentFunctionLabel.C2}:
            return False
        if experiencer == ExperiencerLabel.E1:
            return True
        return self.config.extract_proxy_csm and experiencer == ExperiencerLabel.E2


__all__ = [
    "CSMExtractor",
    "HierarchicalAnnotator",
    "LLMJudge",
    "PipelineConfig",
    "PipelineOutput",
    "PipelineTrace",
    "RAGRetriever",
    "R1Classifier",
    "RelevanceClassifier",
]
