import json

from dental_ai.pipeline import PipelineOutput, PipelineTrace
from dental_ai.local_models import apply_classification_safeguards
from dental_ai.run import _apply_pre_extraction_classification_safeguards, _hf_manifest_config, _should_extract_csm_result
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
        apply_classification_safeguards=apply_classification_safeguards,
        validate_hierarchical_result=validate_hierarchical_result,
    )

    assert guarded.result.content_function == ContentFunctionLabel.C3
    assert "classification_safeguards" in guarded.trace.stages
    assert not _should_extract_csm_result(guarded.result, object())
