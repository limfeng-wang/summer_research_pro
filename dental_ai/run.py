"""Batch runner for the hierarchical annotation pipeline.

The current implementation supports a safe mock backend for end-to-end IO and
hierarchy checks. Real local Hugging Face backends should implement the same
pipeline protocols in a later module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from dental_ai.data_io import read_posts_jsonl, write_extractions_jsonl
from dental_ai.pipeline import HierarchicalAnnotator
from dental_ai.schemas import (
    AssertionStatus,
    CSMDomain,
    ConceptStatus,
    ContentFunctionLabel,
    ExperiencerLabel,
    ExtractionResult,
    JudgeVerdict,
    NarrativeUnit,
    RelevanceLabel,
    SourcePost,
    SupportType,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run hierarchical dental pain annotation.")
    parser.add_argument("--input", required=True, help="Input source-post JSONL")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--backend", choices=("mock", "hf"), default="mock")
    parser.add_argument("--config", default="configs/model_stack.yaml", help="Model stack config")
    parser.add_argument("--models-root", default="/hdd-storage/lawrencelcty/huggingface/models")
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows for smoke tests")
    parser.add_argument(
        "--hf-stage",
        choices=("classify", "full"),
        default="classify",
        help="HF backend stage. 'classify' avoids loading extractor/judge.",
    )
    args = parser.parse_args(argv)

    posts = read_posts_jsonl(args.input)
    if args.limit:
        posts = posts[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "mock":
        outputs = [_mock_annotator().annotate_post(post) for post in posts]
    else:
        outputs = _run_hf(posts, config_path=args.config, models_root=args.models_root, stage=args.hf_stage)
    results = [output.result for output in outputs]

    write_extractions_jsonl(results, out_dir / "annotations.jsonl")
    _write_manifest(outputs, out_dir / "run_manifest.json", input_path=args.input, backend=args.backend)
    return 0


def _mock_annotator() -> HierarchicalAnnotator:
    return HierarchicalAnnotator(
        relevance_classifier=_MockRelevance(),
        r1_classifier=_MockR1(),
        csm_extractor=_MockExtractor(),
        retriever=_MockRetriever(),
        judge=_MockJudge(),
    )


def _run_hf(posts: list[SourcePost], *, config_path: str, models_root: str, stage: str):
    if stage != "classify":
        raise SystemExit("HF full extraction backend is not implemented yet. Use --hf-stage classify.")

    from dental_ai.local_models import LocalR1Classifier, LocalRelevanceClassifier, local_lm_for_role
    from dental_ai.model_config import load_model_stack_config
    from dental_ai.pipeline import PipelineOutput, PipelineTrace
    from dental_ai.validate import validate_hierarchical_result

    stack = load_model_stack_config(config_path)
    classifier_lm = local_lm_for_role(stack, "classifier", models_root=models_root)
    relevance = LocalRelevanceClassifier(classifier_lm)
    r1 = LocalR1Classifier(classifier_lm)
    outputs = []
    try:
        for post in posts:
            stages = ["relevance"]
            relevance_label = relevance.classify_relevance(post)
            result = ExtractionResult.empty_for_post(post).model_copy(update={"relevance_label": relevance_label})
            if relevance_label == RelevanceLabel.R1:
                experiencer, content_function = r1.classify_r1(post)
                stages.append("r1_classification")
                result = result.model_copy(
                    update={
                        "experiencer_label": experiencer,
                        "content_function": content_function,
                    }
                )
            validation = validate_hierarchical_result(result, post)
            outputs.append(PipelineOutput(result=result, trace=PipelineTrace(stages=stages, validation=validation)))
    finally:
        classifier_lm.close()
    return outputs


class _MockRelevance:
    def classify_relevance(self, post: SourcePost) -> RelevanceLabel:
        return RelevanceLabel.R1 if _contains_toothache_hint(post.combined_source_text) else RelevanceLabel.R0


class _MockR1:
    def classify_r1(self, post: SourcePost) -> tuple[ExperiencerLabel, ContentFunctionLabel]:
        text = post.combined_source_text
        experiencer = ExperiencerLabel.E1
        if any(token in text for token in ["妈妈", "爸爸", "娘", "旦那", "친구", "엄마", "아빠"]):
            experiencer = ExperiencerLabel.E2
        content_function = ContentFunctionLabel.C2 if any(token in text for token in ["?", "？", "怎么办", "どうした", "어떡"]) else ContentFunctionLabel.C1
        return experiencer, content_function


class _MockRetriever:
    def retrieve(self, post: SourcePost, *, k: int) -> list[ExtractionResult]:
        return []


class _MockExtractor:
    def extract_csm(self, post: SourcePost, seed_examples: list[ExtractionResult]) -> ExtractionResult:
        span = _first_toothache_span(post.combined_source_text)
        units = []
        if span:
            units.append(
                NarrativeUnit(
                    domain=CSMDomain.SYMPTOM_DESCRIPTION,
                    evidence_span_original=span,
                    surface_text_working="牙痛",
                    normalized_concept_en="Toothache",
                    concept_status=ConceptStatus.NEW_CANDIDATE,
                    support_type=SupportType.EXPLICIT,
                    assertion=AssertionStatus.PRESENT,
                    confidence=0.1,
                    judge_verdict=JudgeVerdict.NEEDS_HUMAN_REVIEW,
                )
            )
        return ExtractionResult.empty_for_post(post).model_copy(update={"units": units})


class _MockJudge:
    def judge(self, post: SourcePost, result: ExtractionResult) -> ExtractionResult:
        units = [unit.model_copy(update={"judge_verdict": JudgeVerdict.ACCEPT}) for unit in result.units]
        return result.model_copy(update={"units": units})


def _contains_toothache_hint(text: str) -> bool:
    return any(token in text for token in ["牙疼", "牙痛", "歯が痛", "歯痛", "치통", "이가 아"])


def _first_toothache_span(text: str) -> str:
    for token in ["牙疼", "牙痛", "歯が痛", "歯痛", "치통", "이가 아"]:
        index = text.find(token)
        if index >= 0:
            return text[index : index + len(token)]
    return ""


def _write_manifest(outputs: Iterable[object], path: Path, *, input_path: str, backend: str) -> None:
    output_list = list(outputs)
    manifest = {
        "input": input_path,
        "backend": backend,
        "rows": len(output_list),
        "validation_ok": sum(1 for output in output_list if output.trace.validation.ok),
        "validation_failed": sum(1 for output in output_list if not output.trace.validation.ok),
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
