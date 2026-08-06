import json

from dental_ai.run import _hf_manifest_config


def test_hf_manifest_lists_only_active_models_when_reranker_disabled(tmp_path):
    config = tmp_path / "model_stack.json"
    config.write_text(
        json.dumps(
            {
                "model_stack": {
                    "classifier": {
                        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
                        "backend": "transformers",
                    },
                    "judge": {
                        "model_id": "google/gemma-4-E2B-it",
                        "backend": "transformers-gemma4",
                    },
                    "retriever": {
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
    assert manifest["disabled_optional_models"]["reranker"]["model_id"] == "BAAI/bge-reranker-v2-m3"


def test_hf_manifest_lists_reranker_as_active_when_enabled(tmp_path):
    config = tmp_path / "model_stack.json"
    config.write_text(
        json.dumps(
            {
                "model_stack": {
                    "retriever": {
                        "model_id": "BAAI/bge-m3",
                        "backend": "sentence-transformers",
                    },
                    "reranker": {
                        "model_id": "BAAI/bge-reranker-v2-m3",
                        "backend": "FlagEmbedding",
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
    assert manifest["disabled_optional_models"] == {}
