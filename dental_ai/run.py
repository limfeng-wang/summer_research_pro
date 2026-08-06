"""Batch runner for the hierarchical annotation pipeline.

The current implementation supports a safe mock backend for end-to-end IO and
hierarchy checks. Real local Hugging Face backends should implement the same
pipeline protocols in a later module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, TextIO

from dental_ai.data_io import read_posts_jsonl
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
    annotations_path = out_dir / "annotations.jsonl"
    errors_path = out_dir / "errors.jsonl"
    retrieval_trace_path = out_dir / "retrieval_trace.jsonl"
    annotations_path.write_text("", encoding="utf-8")
    errors_path.write_text("", encoding="utf-8")
    retrieval_trace_path.write_text("", encoding="utf-8")

    with annotations_path.open("a", encoding="utf-8") as annotations_file, errors_path.open(
        "a",
        encoding="utf-8",
    ) as errors_file, retrieval_trace_path.open("a", encoding="utf-8") as retrieval_trace_file:
        if args.backend == "mock":
            outputs, errors = _run_mock(posts, annotations_file=annotations_file, errors_file=errors_file)
        else:
            outputs, errors = _run_hf(
                posts,
                config_path=args.config,
                models_root=args.models_root,
                stage=args.hf_stage,
                annotations_file=annotations_file,
                errors_file=errors_file,
                retrieval_trace_file=retrieval_trace_file,
            )

    _write_manifest(
        outputs,
        out_dir / "run_manifest.json",
        input_path=args.input,
        backend=args.backend,
        stage=args.hf_stage if args.backend == "hf" else "mock",
        config_path=args.config if args.backend == "hf" else "",
        models_root=args.models_root if args.backend == "hf" else "",
        attempted=len(posts),
        errors=errors,
    )
    return 0


def _mock_annotator() -> HierarchicalAnnotator:
    return HierarchicalAnnotator(
        relevance_classifier=_MockRelevance(),
        r1_classifier=_MockR1(),
        csm_extractor=_MockExtractor(),
        retriever=_MockRetriever(),
        judge=_MockJudge(),
    )


def _run_mock(posts: list[SourcePost], *, annotations_file: TextIO, errors_file: TextIO):
    annotator = _mock_annotator()
    outputs = []
    errors = []
    for index, post in enumerate(_progress(posts, desc="annotating"), start=1):
        try:
            output = annotator.annotate_post(post)
            outputs.append(output)
            _write_result_line(annotations_file, output.result)
        except Exception as exc:
            error = _error_record(post, index=index, stage="mock", exc=exc)
            errors.append(error)
            _write_jsonl_line(errors_file, error)
    return outputs, errors


def _run_hf(
    posts: list[SourcePost],
    *,
    config_path: str,
    models_root: str,
    stage: str,
    annotations_file: TextIO,
    errors_file: TextIO,
    retrieval_trace_file: TextIO,
):
    from dental_ai.classification_gold import load_classification_gold_jsonl
    from dental_ai.local_models import (
        LocalCSMExtractor,
        LocalJudge,
        LocalR1Classifier,
        LocalRelevanceClassifier,
        local_lm_for_role,
    )
    from dental_ai.model_config import load_model_stack_config
    from dental_ai.pipeline import PipelineConfig, PipelineOutput, PipelineTrace
    from dental_ai.rag import GoldRAGRetriever, retrieval_trace_rows
    from dental_ai.validate import validate_hierarchical_result

    stack = load_model_stack_config(config_path)
    classifier_lm = local_lm_for_role(stack, "classifier", models_root=models_root)
    classification_examples = []
    classification_gold_path = stack.paths.get("classification_gold", "")
    if classification_gold_path:
        classification_examples = load_classification_gold_jsonl(classification_gold_path)
    fewshot_k = int(stack.runtime.get("classification_fewshot_k", 8))
    relevance = LocalRelevanceClassifier(classifier_lm)
    r1 = LocalR1Classifier(
        classifier_lm,
        classification_examples=classification_examples,
        fewshot_k=fewshot_k,
    )
    outputs = []
    errors = []
    try:
        for index, post in enumerate(_progress(posts, desc="hf-classify"), start=1):
            current_stage = "relevance"
            try:
                stages = ["relevance"]
                relevance_label = relevance.classify_relevance(post)
                result = ExtractionResult.empty_for_post(post).model_copy(update={"relevance_label": relevance_label})
                if relevance_label == RelevanceLabel.R1:
                    current_stage = "r1_classification"
                    experiencer, content_function = r1.classify_r1(post)
                    stages.append("r1_classification")
                    result = result.model_copy(
                        update={
                            "experiencer_label": experiencer,
                            "content_function": content_function,
                        }
                    )
                current_stage = "validation"
                validation = validate_hierarchical_result(result, post)
                output = PipelineOutput(result=result, trace=PipelineTrace(stages=stages, validation=validation))
                outputs.append(output)
                if stage == "classify":
                    _write_result_line(annotations_file, output.result)
            except Exception as exc:
                error = _error_record(post, index=index, stage=current_stage, exc=exc)
                errors.append(error)
                _write_jsonl_line(errors_file, error)
    finally:
        classifier_lm.close()

    if stage == "classify":
        return outputs, errors
    if stage != "full":
        raise SystemExit(f"Unsupported HF stage: {stage}")

    output_by_post_id = {output.result.post_id: output for output in outputs}
    rag_k = int(stack.runtime.get("default_rag_k", 5))
    config = PipelineConfig(
        extract_proxy_csm=bool(stack.runtime.get("extract_proxy_csm", True)),
        rag_k=rag_k,
    )
    retriever = GoldRAGRetriever.from_config(stack, models_root=models_root)

    extractor_lm = local_lm_for_role(stack, "extractor", models_root=models_root)
    extractor = LocalCSMExtractor(extractor_lm)
    try:
        for index, post in enumerate(_progress(posts, desc="hf-extract"), start=1):
            output = output_by_post_id.get(post.post_id)
            if output is None:
                continue
            result = output.result
            if not _should_extract_csm_result(result, config):
                continue
            current_stage = "rag_retrieval"
            try:
                retrieved = retriever.retrieve_with_scores(post, k=config.rag_k)
                for trace_row in retrieval_trace_rows(post, retrieved):
                    _write_jsonl_line(retrieval_trace_file, trace_row)
                current_stage = "csm_extraction"
                extracted = extractor.extract_csm(post, [item.result for item in retrieved])
                extracted = ExtractionResult.empty_for_post(post).model_copy(
                    update={
                        "units": extracted.units,
                        "relevance_label": result.relevance_label,
                        "experiencer_label": result.experiencer_label,
                        "content_function": result.content_function,
                    }
                ).with_assigned_unit_ids()
                validation = validate_hierarchical_result(extracted, post)
                stages = output.trace.stages + ["rag_retrieval", "csm_extraction"]
                output_by_post_id[post.post_id] = PipelineOutput(
                    result=extracted,
                    trace=PipelineTrace(stages=stages, validation=validation),
                )
            except Exception as exc:
                error = _error_record(post, index=index, stage=current_stage, exc=exc)
                errors.append(error)
                _write_jsonl_line(errors_file, error)
    finally:
        extractor_lm.close()

    judge_lm = local_lm_for_role(stack, "judge", models_root=models_root)
    judge = LocalJudge(judge_lm)
    try:
        for index, post in enumerate(_progress(posts, desc="hf-judge"), start=1):
            output = output_by_post_id.get(post.post_id)
            if output is None or not output.result.units:
                continue
            try:
                judged = judge.judge(post, output.result)
                validation = validate_hierarchical_result(judged, post)
                output_by_post_id[post.post_id] = PipelineOutput(
                    result=judged,
                    trace=PipelineTrace(stages=output.trace.stages + ["judge"], validation=validation),
                )
            except Exception as exc:
                error = _error_record(post, index=index, stage="judge", exc=exc)
                errors.append(error)
                _write_jsonl_line(errors_file, error)
    finally:
        judge_lm.close()

    final_outputs = [output_by_post_id[post.post_id] for post in posts if post.post_id in output_by_post_id]
    for output in final_outputs:
        _write_result_line(annotations_file, output.result)
    return final_outputs, errors


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


def _progress(posts: list[SourcePost], *, desc: str):
    try:
        from tqdm.auto import tqdm

        return tqdm(posts, total=len(posts), desc=desc, unit="post")
    except Exception:
        return posts


def _write_result_line(file: TextIO, result: ExtractionResult) -> None:
    _write_jsonl_line(file, result.model_dump(mode="json"))


def _write_jsonl_line(file: TextIO, payload: dict[str, Any]) -> None:
    file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    file.flush()


def _error_record(post: SourcePost, *, index: int, stage: str, exc: Exception) -> dict[str, Any]:
    return {
        "index": index,
        "post_id": post.post_id,
        "country": post.country.value,
        "language": post.language.value,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _write_manifest(
    outputs: Iterable[object],
    path: Path,
    *,
    input_path: str,
    backend: str,
    stage: str,
    config_path: str,
    models_root: str,
    attempted: int,
    errors: list[dict[str, Any]],
) -> None:
    output_list = list(outputs)
    manifest = {
        "input": input_path,
        "backend": backend,
        "stage": stage,
        "rows_attempted": attempted,
        "rows": len(output_list),
        "rows_succeeded": len(output_list),
        "rows_failed": len(errors),
        "errors_path": "errors.jsonl",
        "validation_ok": sum(1 for output in output_list if output.trace.validation.ok),
        "validation_failed": sum(1 for output in output_list if not output.trace.validation.ok),
    }
    if backend == "hf":
        manifest["hf_config"] = _hf_manifest_config(config_path=config_path, models_root=models_root)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _hf_manifest_config(*, config_path: str, models_root: str) -> dict[str, Any]:
    from dental_ai.model_config import load_model_stack_config

    stack = load_model_stack_config(config_path)
    roles = ["classifier", "extractor", "judge", "retriever", "reranker"]
    models = {}
    for role in roles:
        if role not in stack.specs:
            continue
        spec = stack.spec(role)
        models[role] = {
            "model_id": spec.model_id,
            "backend": spec.backend,
            "local_path": str(spec.local_path(models_root)),
        }
    runtime_keys = [
        "default_rag_k",
        "classification_fewshot_k",
        "use_reranker",
        "reranker_backend",
        "reranker_batch_size",
        "reranker_max_length",
        "reranker_use_fp16",
        "reranker_device",
    ]
    return {
        "config_path": config_path,
        "models_root": models_root,
        "models": models,
        "runtime": {key: stack.runtime.get(key) for key in runtime_keys if key in stack.runtime},
        "paths": {
            "classification_gold": stack.paths.get("classification_gold", ""),
            "rag_gold": stack.paths.get("rag_gold", ""),
        },
    }


def _should_extract_csm_result(result: ExtractionResult, config: object) -> bool:
    if result.relevance_label != RelevanceLabel.R1:
        return False
    if result.content_function not in {ContentFunctionLabel.C1, ContentFunctionLabel.C2}:
        return False
    if result.experiencer_label == ExperiencerLabel.E1:
        return True
    return bool(getattr(config, "extract_proxy_csm", True)) and result.experiencer_label == ExperiencerLabel.E2


if __name__ == "__main__":
    raise SystemExit(main())
