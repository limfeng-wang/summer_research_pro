"""Small real-model smoke test; run through test_gpu_smoke.sh."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    import torch
    from huggingface_hub import snapshot_download
    from dental_ai.data_io import read_posts_jsonl
    from dental_ai.local_models import LocalCSMExtractor, LocalJudge, local_lm_for_role
    from dental_ai.model_config import active_model_roles, load_model_stack_config
    from dental_ai.publication import read_jsonl
    from dental_ai.rag import GoldRAGRetriever
    from dental_ai.run import main as run
    from dental_ai.schemas import ExtractionResult

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Private evaluation inputs; not distributed")
    parser.add_argument("--predictions", type=Path, required=True, help="Private archived predictions; not distributed")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assert Path.cwd() == root, "Run from the release root"
    assert torch.cuda.is_available(), "CUDA unavailable"
    assert torch.cuda.device_count() > 0
    stack = load_model_stack_config(args.config)
    for path in [args.input, args.predictions, Path(stack.paths.get("rag_gold", "")),
                 Path(stack.paths.get("classification_gold", ""))]:
        if not path.is_file():
            parser.error(f"Missing private input: {path}; study records are not distributed.")
    inventory = {}
    try:
        for role in active_model_roles(stack):
            spec = stack.spec(role)
            path = spec.local_path(args.models_root)
            weights = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
            action = "existing local files"
            if not (path / "config.json").exists() or not weights:
                action = "downloaded/resumed missing model files; retained in shared model directory"
                snapshot_download(repo_id=spec.model_id, local_dir=str(path))
            inventory[role] = {"model_id": spec.model_id, "path": str(path), "action": action}
    finally:
        (args.out_dir / "models.json").write_text(json.dumps(inventory, indent=2) + "\n")

    import traceback
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(
            str(stack.spec("judge").local_path(args.models_root)),
            local_files_only=True,
            trust_remote_code=True,
        )
        del processor
    except Exception:
        (args.out_dir / "judge_processor_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise

    posts = read_posts_jsonl(args.input)
    archived = {row["post_id"]: row for row in read_jsonl(args.predictions)}
    selected = []
    for language in ["zh", "ja", "ko"]:
        candidates = [post for post in posts if post.language.value == language and archived[post.post_id].get("units")]
        assert candidates, f"No archived extraction example for {language}"
        selected.append(min(candidates, key=lambda post: len(post.combined_source_text)))
    input_path = args.out_dir / "input.jsonl"
    input_path.write_text("".join(post.model_dump_json() + "\n" for post in selected), encoding="utf-8")
    retriever = GoldRAGRetriever.from_config(stack, models_root=args.models_root)
    retrieval = {}
    for post in selected:
        scores = retriever._embedding_scores(post)
        assert scores is not None, "Dense retrieval failed; lexical fallback is not a dense retrieval pass"
        hits = retriever.retrieve_with_scores(post, k=5)
        assert len(hits) == 5
        retrieval[post.post_id] = [{"post_id": hit.result.post_id, "score": hit.score} for hit in hits]
    (args.out_dir / "retrieval.json").write_text(json.dumps(retrieval, indent=2) + "\n")
    del retriever
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    output = args.out_dir / "full"
    arguments = ["--input", str(input_path), "--out-dir", str(output), "--backend", "hf", "--hf-stage", "full", "--models-root", str(args.models_root), "--config", str(args.config)]
    assert run(arguments) == 0
    records = read_jsonl(output / "annotations.jsonl")
    assert not (output / "errors.jsonl").read_text().strip(), "Inspect full/errors.jsonl"
    assert len(records) == 3 and {row["post_id"] for row in records} == {post.post_id for post in selected}, "Missing or duplicate final annotations; inspect full/ checkpoints and logs"
    for row in records:
        ExtractionResult.model_validate(row)
    before = hashlib.sha256((output / "annotations.jsonl").read_bytes()).hexdigest()
    assert run([*arguments, "--resume"]) == 0
    assert hashlib.sha256((output / "annotations.jsonl").read_bytes()).hexdigest() == before
    assert not (output / "errors.jsonl").read_text().strip()

    probes = {}
    for role, client_class in [("extractor", LocalCSMExtractor), ("judge", LocalJudge)]:
        model = local_lm_for_role(stack, role, models_root=args.models_root)
        try:
            client = client_class(model)
            for post in selected:
                reference = ExtractionResult.model_validate(archived[post.post_id])
                if role == "extractor":
                    result = client.extract_csm(post, [])
                    result.validate_evidence_spans(post.combined_source_text)
                else:
                    result = client.judge(post, reference)
                    assert len(result.units) == len(reference.units)
                    assert any(unit.judge_verdict.value != "needs_human_review" for unit in result.units), "Judge returned no usable verdict"
                probes[f"{role}:{post.language.value}"] = result.model_dump(mode="json")
        finally:
            model.close()
    (args.out_dir / "stage_probes.json").write_text(json.dumps(probes, ensure_ascii=False, indent=2) + "\n")
    report = {
        "result": "PASS",
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "checks": ["CUDA", "model loading", "dense retrieval without fallback", "three-language full pipeline", "exact resume stability", "direct extraction and judge probes in every language"],
        "scope": "Execution smoke test, not accuracy validation or exact historical prediction reproduction. Direct stage probes ensure coverage even if the new classifier filters out inputs.",
        "models": inventory,
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
