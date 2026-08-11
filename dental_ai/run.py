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
from dental_ai.validate import ValidationIssue, ValidationReport


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run hierarchical dental pain annotation.")
    parser.add_argument("--input", required=True, help="Input source-post JSONL")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--backend", choices=("mock", "hf"), default="mock")
    parser.add_argument("--config", default="configs/model_stack.yaml", help="Model stack config")
    parser.add_argument("--models-root", default="/hdd-storage/lawrencelcty/huggingface/models")
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows for smoke tests")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint/output files")
    parser.add_argument("--shard-count", type=int, default=1, help="Number of contiguous input shards")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based contiguous shard index")
    parser.add_argument(
        "--classification-mode",
        choices=("combined", "separate"),
        default="combined",
        help="Use one combined R/E/C classifier call per row, or the legacy two-call classifier.",
    )
    parser.add_argument(
        "--hf-stage",
        choices=("classify", "full"),
        default="classify",
        help="HF backend stage. 'classify' avoids loading extractor/judge.",
    )
    args = parser.parse_args(argv)

    input_posts = read_posts_jsonl(args.input)
    if args.limit:
        input_posts = input_posts[: args.limit]
    posts = _select_shard(input_posts, shard_count=args.shard_count, shard_index=args.shard_index)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = out_dir / "annotations.jsonl"
    errors_path = out_dir / "errors.jsonl"
    retrieval_trace_path = out_dir / "retrieval_trace.jsonl"
    classified_checkpoint_path = checkpoint_dir / "classified.jsonl"
    extracted_checkpoint_path = checkpoint_dir / "extracted.jsonl"
    if not args.resume:
        annotations_path.write_text("", encoding="utf-8")
        errors_path.write_text("", encoding="utf-8")
        retrieval_trace_path.write_text("", encoding="utf-8")
        classified_checkpoint_path.write_text("", encoding="utf-8")
        extracted_checkpoint_path.write_text("", encoding="utf-8")

    with annotations_path.open("a", encoding="utf-8") as annotations_file, errors_path.open(
        "a",
        encoding="utf-8",
    ) as errors_file, retrieval_trace_path.open("a", encoding="utf-8") as retrieval_trace_file, classified_checkpoint_path.open(
        "a",
        encoding="utf-8",
    ) as classified_checkpoint_file, extracted_checkpoint_path.open(
        "a",
        encoding="utf-8",
    ) as extracted_checkpoint_file:
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
                classified_checkpoint_file=classified_checkpoint_file,
                extracted_checkpoint_file=extracted_checkpoint_file,
                annotations_path=annotations_path,
                errors_path=errors_path,
                classified_checkpoint_path=classified_checkpoint_path,
                extracted_checkpoint_path=extracted_checkpoint_path,
                resume=args.resume,
                classification_mode=args.classification_mode,
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
        input_rows=len(input_posts),
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        resume=args.resume,
        classification_mode=args.classification_mode if args.backend == "hf" else "",
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


def _select_shard(posts: list[SourcePost], *, shard_count: int, shard_index: int) -> list[SourcePost]:
    if shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < shard-count")
    start = len(posts) * shard_index // shard_count
    end = len(posts) * (shard_index + 1) // shard_count
    return posts[start:end]


def _load_output_map(path: Path) -> dict[str, "PipelineOutput"]:
    from dental_ai.pipeline import PipelineOutput, PipelineTrace

    outputs: dict[str, PipelineOutput] = {}
    if not path.exists():
        return outputs
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if "result" in payload:
            result = ExtractionResult.model_validate(payload["result"])
            trace_payload = payload.get("trace", {})
            validation_payload = trace_payload.get("validation", {})
            issues = [
                ValidationIssue(code=str(issue.get("code", "")), message=str(issue.get("message", "")))
                for issue in validation_payload.get("issues", [])
                if isinstance(issue, dict)
            ]
            validation = ValidationReport(
                ok=bool(validation_payload.get("ok", not issues)),
                issues=issues,
            )
            trace = PipelineTrace(
                stages=list(trace_payload.get("stages", [])),
                validation=validation,
                raw_labels=dict(trace_payload.get("raw_labels", {})),
                postprocessing_rules=list(trace_payload.get("postprocessing_rules", [])),
            )
        else:
            result = ExtractionResult.model_validate(payload)
            trace = PipelineTrace(stages=["loaded_final"], validation=ValidationReport.pass_())
        outputs[result.post_id] = PipelineOutput(result=result, trace=trace)
    return outputs


def _load_error_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _run_hf(
    posts: list[SourcePost],
    *,
    config_path: str,
    models_root: str,
    stage: str,
    annotations_file: TextIO,
    errors_file: TextIO,
    retrieval_trace_file: TextIO,
    classified_checkpoint_file: TextIO,
    extracted_checkpoint_file: TextIO,
    annotations_path: Path,
    errors_path: Path,
    classified_checkpoint_path: Path,
    extracted_checkpoint_path: Path,
    resume: bool,
    classification_mode: str,
):
    from dental_ai.classification_gold import load_classification_gold_jsonl
    from dental_ai.local_models import (
        LocalCombinedClassifier,
        LocalCSMExtractor,
        LocalJudge,
        LocalR1Classifier,
        LocalRelevanceClassifier,
        local_lm_for_role,
    )
    from dental_ai.classification_postprocess import apply_rec_postprocessing
    from dental_ai.model_config import load_model_stack_config
    from dental_ai.pipeline import PipelineConfig, PipelineOutput, PipelineTrace
    from dental_ai.rag import GoldRAGRetriever, retrieval_trace_rows
    from dental_ai.validate import validate_hierarchical_result

    stack = load_model_stack_config(config_path)
    final_outputs_by_post_id = _load_output_map(annotations_path) if resume else {}
    classified_by_post_id = _load_output_map(classified_checkpoint_path) if resume else {}
    extracted_by_post_id = _load_output_map(extracted_checkpoint_path) if resume else {}
    final_post_ids = set(final_outputs_by_post_id)
    errors = _load_error_records(errors_path) if resume else []

    classifier_lm = local_lm_for_role(stack, "classifier", models_root=models_root)
    classification_examples = []
    classification_gold_path = stack.paths.get("classification_gold", "")
    if classification_gold_path:
        classification_examples = load_classification_gold_jsonl(classification_gold_path)
    fewshot_k = int(stack.runtime.get("classification_fewshot_k", 8))
    combined = LocalCombinedClassifier(
        classifier_lm,
        classification_examples=classification_examples,
        fewshot_k=fewshot_k,
    )
    relevance = LocalRelevanceClassifier(classifier_lm)
    r1 = LocalR1Classifier(
        classifier_lm,
        classification_examples=classification_examples,
        fewshot_k=fewshot_k,
    )
    outputs_by_post_id = {**classified_by_post_id, **final_outputs_by_post_id}
    try:
        for index, post in enumerate(_progress(posts, desc="hf-classify"), start=1):
            if post.post_id in classified_by_post_id or post.post_id in final_post_ids:
                continue
            current_stage = "relevance"
            try:
                if classification_mode == "combined":
                    current_stage = "combined_classification"
                    relevance_label, experiencer, content_function = combined.classify(post)
                    stages = ["combined_classification"]
                    raw_labels = dict(combined.last_raw_labels)
                    postprocessing_rules = list(combined.last_postprocess_rules)
                else:
                    stages = ["relevance"]
                    relevance_label = relevance.classify_relevance(post)
                    experiencer = None
                    content_function = None
                    raw_labels = dict(relevance.last_raw_labels)
                    postprocessing_rules = list(relevance.last_postprocess_rules)
                result = ExtractionResult.empty_for_post(post).model_copy(update={"relevance_label": relevance_label})
                if relevance_label == RelevanceLabel.R1:
                    if classification_mode == "separate":
                        current_stage = "r1_classification"
                        experiencer, content_function = r1.classify_r1(post)
                        stages.append("r1_classification")
                        raw_labels.update(r1.last_raw_labels)
                        postprocessing_rules.extend(r1.last_postprocess_rules)
                    result = result.model_copy(
                        update={
                            "experiencer_label": experiencer,
                            "content_function": content_function,
                        }
                    )
                current_stage = "validation"
                validation = validate_hierarchical_result(result, post)
                output = PipelineOutput(
                    result=result,
                    trace=PipelineTrace(
                        stages=stages,
                        validation=validation,
                        raw_labels=raw_labels,
                        postprocessing_rules=postprocessing_rules,
                    ),
                )
                classified_by_post_id[post.post_id] = output
                outputs_by_post_id[post.post_id] = output
                _write_output_line(classified_checkpoint_file, output)
                if stage == "classify":
                    _write_final_output(
                        annotations_file,
                        output,
                        final_outputs_by_post_id,
                        final_post_ids,
                    )
            except Exception as exc:
                error = _error_record(post, index=index, stage=current_stage, exc=exc)
                errors.append(error)
                _write_jsonl_line(errors_file, error)
    finally:
        classifier_lm.close()

    if stage == "classify":
        return list(outputs_by_post_id.values()), errors
    if stage != "full":
        raise SystemExit(f"Unsupported HF stage: {stage}")

    output_by_post_id = dict(outputs_by_post_id)
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
            if post.post_id in final_post_ids:
                continue
            output = output_by_post_id.get(post.post_id)
            if output is None:
                continue
            output = _apply_pre_extraction_classification_safeguards(
                post,
                output,
                apply_rec_postprocessing=apply_rec_postprocessing,
                validate_hierarchical_result=validate_hierarchical_result,
            )
            output_by_post_id[post.post_id] = output
            result = output.result
            if not _should_extract_csm_result(result, config):
                _write_final_output(
                    annotations_file,
                    output,
                    final_outputs_by_post_id,
                    final_post_ids,
                )
                continue
            if post.post_id in extracted_by_post_id:
                output_by_post_id[post.post_id] = extracted_by_post_id[post.post_id]
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
                    trace=_trace_with(output.trace, stages=stages, validation=validation),
                )
                extracted_by_post_id[post.post_id] = output_by_post_id[post.post_id]
                _write_output_line(extracted_checkpoint_file, output_by_post_id[post.post_id])
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
            if post.post_id in final_post_ids:
                continue
            output = output_by_post_id.get(post.post_id)
            if output is None:
                continue
            if not output.result.units:
                _write_final_output(
                    annotations_file,
                    output,
                    final_outputs_by_post_id,
                    final_post_ids,
                )
                continue
            try:
                judged = judge.judge(post, output.result)
                validation = validate_hierarchical_result(judged, post)
                output_by_post_id[post.post_id] = PipelineOutput(
                    result=judged,
                    trace=_trace_with(output.trace, stages=output.trace.stages + ["judge"], validation=validation),
                )
                _write_final_output(
                    annotations_file,
                    output_by_post_id[post.post_id],
                    final_outputs_by_post_id,
                    final_post_ids,
                )
            except Exception as exc:
                error = _error_record(post, index=index, stage="judge", exc=exc)
                errors.append(error)
                _write_jsonl_line(errors_file, error)
    finally:
        judge_lm.close()

    return [final_outputs_by_post_id[post.post_id] for post in posts if post.post_id in final_outputs_by_post_id], errors


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


def _write_output_line(file: TextIO, output: object) -> None:
    result = getattr(output, "result")
    trace = getattr(output, "trace")
    validation = trace.validation
    _write_jsonl_line(
        file,
        {
            "result": result.model_dump(mode="json"),
            "trace": {
                "stages": list(trace.stages),
                "raw_labels": dict(getattr(trace, "raw_labels", {})),
                "postprocessing_rules": list(getattr(trace, "postprocessing_rules", [])),
                "validation": {
                    "ok": validation.ok,
                    "issues": [
                        {"code": issue.code, "message": issue.message}
                        for issue in validation.issues
                    ],
                },
            },
        },
    )


def _write_final_output(
    file: TextIO,
    output: object,
    outputs_by_post_id: dict[str, object],
    final_post_ids: set[str],
) -> None:
    result = getattr(output, "result")
    if result.post_id in final_post_ids:
        return
    _write_result_line(file, result)
    outputs_by_post_id[result.post_id] = output
    final_post_ids.add(result.post_id)


def _write_jsonl_line(file: TextIO, payload: dict[str, Any]) -> None:
    file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    file.flush()


def _trace_with(trace: object, *, stages: list[str], validation: ValidationReport) -> object:
    return trace.__class__(
        stages=stages,
        validation=validation,
        raw_labels=dict(getattr(trace, "raw_labels", {})),
        postprocessing_rules=list(getattr(trace, "postprocessing_rules", [])),
    )


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
    input_rows: int,
    shard_count: int,
    shard_index: int,
    resume: bool,
    classification_mode: str,
) -> None:
    output_list = list(outputs)
    completed_post_ids = {
        getattr(getattr(output, "result", None), "post_id", "")
        for output in output_list
    }
    completed_post_ids.discard("")
    error_post_ids = {str(error.get("post_id", "")) for error in errors if error.get("post_id")}
    manifest = {
        "input": input_path,
        "backend": backend,
        "stage": stage,
        "input_rows_after_limit": input_rows,
        "shard_count": shard_count,
        "shard_index": shard_index,
        "resume": resume,
        "classification_mode": classification_mode,
        "rows_attempted": attempted,
        "rows": len(output_list),
        "rows_completed": len(output_list),
        "rows_succeeded": len(output_list),
        "rows_failed": max(attempted - len(completed_post_ids), 0),
        "stage_errors": len(errors),
        "rows_with_stage_errors": len(error_post_ids),
        "errors_path": "errors.jsonl",
        "validation_ok": sum(1 for output in output_list if output.trace.validation.ok),
        "validation_failed": sum(1 for output in output_list if not output.trace.validation.ok),
        "postprocessing_rule_counts": _postprocessing_rule_counts(output_list),
    }
    if backend == "hf":
        manifest["hf_config"] = _hf_manifest_config(config_path=config_path, models_root=models_root)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _postprocessing_rule_counts(outputs: Iterable[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for output in outputs:
        trace = getattr(output, "trace", None)
        if trace is None:
            continue
        for rule in getattr(trace, "postprocessing_rules", []):
            if not isinstance(rule, dict):
                continue
            name = str(rule.get("rule", ""))
            if not name:
                continue
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _hf_manifest_config(*, config_path: str, models_root: str) -> dict[str, Any]:
    from dental_ai.model_config import OPTIONAL_MODEL_ROLES, active_model_roles, load_model_stack_config

    stack = load_model_stack_config(config_path)
    active_roles = active_model_roles(stack)
    optional_roles = [
        role
        for role in active_model_roles(stack, include_optional=True)
        if role in OPTIONAL_MODEL_ROLES and role not in active_roles
    ]
    models = {}
    for role in active_roles:
        spec = stack.spec(role)
        models[role] = {
            "role": spec.role,
            "model_id": spec.model_id,
            "backend": spec.backend,
            "quantization": spec.quantization,
            "device_policy": spec.device_policy,
            "local_path": str(spec.local_path(models_root)),
        }
    disabled_optional_models = {}
    for role in optional_roles:
        spec = stack.spec(role)
        disabled_optional_models[role] = {
            "role": spec.role,
            "model_id": spec.model_id,
            "backend": spec.backend,
            "quantization": spec.quantization,
            "device_policy": spec.device_policy,
            "local_path": str(spec.local_path(models_root)),
            "disabled_by": "runtime.use_reranker=false" if role == "reranker" else "runtime",
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
        "active_model_roles": active_roles,
        "disabled_optional_models": disabled_optional_models,
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


def _apply_pre_extraction_classification_safeguards(
    post: SourcePost,
    output: object,
    *,
    apply_rec_postprocessing: object,
    validate_hierarchical_result: object,
) -> object:
    result = output.result
    if result.relevance_label != RelevanceLabel.R1 or not result.experiencer_label or not result.content_function:
        return output
    processed = apply_rec_postprocessing(
        post,
        result.relevance_label,
        result.experiencer_label,
        result.content_function,
    )
    experiencer = processed.experiencer_label
    content_function = processed.content_function
    if experiencer is None or content_function is None:
        return output
    existing_rules = list(getattr(output.trace, "postprocessing_rules", []))
    new_rules = [
        rule
        for rule in processed.rule_dicts
        if rule not in existing_rules
    ]
    if experiencer == result.experiencer_label and content_function == result.content_function:
        return output
    result = result.model_copy(
        update={
            "experiencer_label": experiencer,
            "content_function": content_function,
        }
    )
    return output.__class__(
        result=result,
        trace=output.trace.__class__(
            stages=output.trace.stages + ["classification_safeguards"],
            validation=validate_hierarchical_result(result, post),
            raw_labels=dict(getattr(output.trace, "raw_labels", {})),
            postprocessing_rules=existing_rules + new_rules,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
