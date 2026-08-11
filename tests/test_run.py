import json

from dental_ai.pipeline import PipelineOutput, PipelineTrace
from dental_ai.classification_postprocess import apply_rec_postprocessing
from dental_ai.run import (
    _apply_pre_extraction_classification_safeguards,
    _hf_manifest_config,
    _load_output_map,
    _select_shard,
    _should_extract_csm_result,
    _write_output_line,
    main as run_main,
)
from dental_ai.schemas import ContentFunctionLabel, Country, ExperiencerLabel, ExtractionResult, Language, RelevanceLabel, SourcePost
from dental_ai.validate import validate_hierarchical_result


def test_hf_manifest_lists_only_active_models_when_reranker_disabled(tmp_path):
    config = tmp_path / "model_stack.json"
    config.write_text(
        json.dumps(
            {
                "model_stack": {
                    "classifier": {
                        "role": "hierarchical relevance and R1 classification",
                        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
                        "backend": "transformers",
                        "quantization": "8bit",
                        "device_policy": "auto",
                    },
                    "judge": {
                        "role": "LLM-as-Judge",
                        "model_id": "google/gemma-4-E2B-it",
                        "backend": "transformers-gemma4",
                        "quantization": "8bit",
                        "device_policy": "cuda:1",
                    },
                    "retriever": {
                        "role": "multilingual dense retrieval",
                        "model_id": "BAAI/bge-m3",
                        "backend": "sentence-transformers",
                    },
                    "reranker": {
                        "model_id": "BAAI/bge-reranker-v2-m3",
                        "backend": "FlagEmbedding",
                    },
                },
                "runtime": {"use_reranker": False},
                "paths": {},
            }
        ),
        encoding="utf-8",
    )

    manifest = _hf_manifest_config(config_path=str(config), models_root="/models")

    assert "reranker" not in manifest["models"]
    assert manifest["active_model_roles"] == ["classifier", "judge", "retriever"]
    assert manifest["models"]["classifier"]["role"] == "hierarchical relevance and R1 classification"
    assert manifest["models"]["classifier"]["quantization"] == "8bit"
    assert manifest["models"]["classifier"]["device_policy"] == "auto"
    assert manifest["models"]["judge"]["role"] == "LLM-as-Judge"
    assert manifest["models"]["judge"]["quantization"] == "8bit"
    assert manifest["models"]["judge"]["device_policy"] == "cuda:1"
    assert manifest["models"]["retriever"]["role"] == "multilingual dense retrieval"
    assert manifest["models"]["retriever"]["quantization"] == ""
    assert manifest["models"]["retriever"]["device_policy"] == "auto"
    assert manifest["disabled_optional_models"]["reranker"]["model_id"] == "BAAI/bge-reranker-v2-m3"
    assert manifest["disabled_optional_models"]["reranker"]["role"] == "reranker"
    assert manifest["disabled_optional_models"]["reranker"]["quantization"] == ""
    assert manifest["disabled_optional_models"]["reranker"]["device_policy"] == "auto"


def test_hf_manifest_lists_reranker_as_active_when_enabled(tmp_path):
    config = tmp_path / "model_stack.json"
    config.write_text(
        json.dumps(
            {
                "model_stack": {
                    "retriever": {
                        "role": "multilingual dense retrieval",
                        "model_id": "BAAI/bge-m3",
                        "backend": "sentence-transformers",
                    },
                    "reranker": {
                        "role": "multilingual reranking",
                        "model_id": "BAAI/bge-reranker-v2-m3",
                        "backend": "FlagEmbedding",
                        "quantization": "fp16",
                        "device_policy": "cuda:0",
                    },
                },
                "runtime": {"use_reranker": True},
                "paths": {},
            }
        ),
        encoding="utf-8",
    )

    manifest = _hf_manifest_config(config_path=str(config), models_root="/models")

    assert "reranker" in manifest["models"]
    assert manifest["active_model_roles"] == ["retriever", "reranker"]
    assert manifest["models"]["reranker"]["role"] == "multilingual reranking"
    assert manifest["models"]["reranker"]["quantization"] == "fp16"
    assert manifest["models"]["reranker"]["device_policy"] == "cuda:0"
    assert manifest["disabled_optional_models"] == {}


def test_select_shard_uses_contiguous_zero_based_partitions():
    posts = [
        SourcePost(post_id=f"p{i}", country=Country.CHI, language=Language.ZH, text_clean="牙疼")
        for i in range(10)
    ]

    assert [post.post_id for post in _select_shard(posts, shard_count=3, shard_index=0)] == ["p0", "p1", "p2"]
    assert [post.post_id for post in _select_shard(posts, shard_count=3, shard_index=1)] == ["p3", "p4", "p5"]
    assert [post.post_id for post in _select_shard(posts, shard_count=3, shard_index=2)] == ["p6", "p7", "p8", "p9"]


def test_output_checkpoint_roundtrip_preserves_trace(tmp_path):
    post = SourcePost(post_id="p1", country=Country.CHI, language=Language.ZH, text_clean="牙疼")
    result = ExtractionResult.empty_for_post(post).model_copy(update={"relevance_label": RelevanceLabel.R1})
    output = PipelineOutput(
        result=result,
        trace=PipelineTrace(
            stages=["combined_classification"],
            validation=validate_hierarchical_result(result, post),
            raw_labels={"relevance_label": "R1"},
            postprocessing_rules=[
                {
                    "rule": "relevance.toothache_evidence_required",
                    "field": "relevance_label",
                    "before": "R1",
                    "after": "R0",
                    "reason": "test",
                }
            ],
        ),
    )
    checkpoint = tmp_path / "classified.jsonl"

    with checkpoint.open("a", encoding="utf-8") as file:
        _write_output_line(file, output)
    loaded = _load_output_map(checkpoint)

    assert loaded["p1"].result.relevance_label == RelevanceLabel.R1
    assert loaded["p1"].trace.stages == ["combined_classification"]
    assert loaded["p1"].trace.raw_labels == {"relevance_label": "R1"}
    assert loaded["p1"].trace.postprocessing_rules[0]["rule"] == "relevance.toothache_evidence_required"
    assert not loaded["p1"].trace.validation.ok
    assert loaded["p1"].trace.validation.issues[0].code == "missing_experiencer"


def test_mock_run_supports_sharding_and_manifest_fields(tmp_path):
    input_path = tmp_path / "input.jsonl"
    out_dir = tmp_path / "out"
    rows = [
        SourcePost(post_id=f"p{i}", country=Country.CHI, language=Language.ZH, text_clean="牙疼").model_dump(mode="json")
        for i in range(4)
    ]
    input_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    assert run_main(
        [
            "--input",
            str(input_path),
            "--out-dir",
            str(out_dir),
            "--backend",
            "mock",
            "--shard-count",
            "2",
            "--shard-index",
            "1",
        ]
    ) == 0

    annotations = [json.loads(line) for line in (out_dir / "annotations.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert [row["post_id"] for row in annotations] == ["p2", "p3"]
    assert manifest["input_rows_after_limit"] == 4
    assert manifest["rows_attempted"] == 2
    assert manifest["shard_count"] == 2
    assert manifest["shard_index"] == 1


def test_pre_extraction_safeguard_blocks_generic_wisdom_tooth_logistics():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean=(
            "两种阻生齿,看看需要多少钱。拔牙×2,第一次手术费975,第二次手术费1300左右。"
            "作者本人的是阻生齿,有龋齿,不痛,没发炎。首先,关于挂号的小tips:挂号一定要趁早。"
            "其次,过程:挂号、签到,等待叫号。拔牙后的注意事项:拔完牙24h内可以冰敷一下。"
        ),
    )
    result = ExtractionResult.empty_for_post(post).model_copy(
        update={
            "relevance_label": RelevanceLabel.R1,
            "experiencer_label": ExperiencerLabel.E1,
            "content_function": ContentFunctionLabel.C1,
        }
    )
    output = PipelineOutput(result=result, trace=PipelineTrace(stages=["relevance", "r1_classification"]))

    guarded = _apply_pre_extraction_classification_safeguards(
        post,
        output,
        apply_rec_postprocessing=apply_rec_postprocessing,
        validate_hierarchical_result=validate_hierarchical_result,
    )

    assert guarded.result.content_function == ContentFunctionLabel.C3
    assert "classification_safeguards" in guarded.trace.stages
    assert guarded.trace.postprocessing_rules[0]["rule"] == "content.generic_procedure_demote"
    assert not _should_extract_csm_result(guarded.result, object())
