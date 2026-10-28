import json

from dental_ai.model_config import active_model_roles, check_model_paths, load_model_stack_config


def write_config(tmp_path):
    path = tmp_path / "model_stack.json"
    path.write_text(
        json.dumps(
            {
                "model_stack": {
                    "classifier": {
                        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
                        "backend": "transformers",
                    },
                    "extractor": {
                        "model_id": "Qwen/Qwen3-8B",
                        "backend": "transformers",
                    },
                    "reranker": {
                        "model_id": "BAAI/bge-reranker-v2-m3",
                        "backend": "transformers",
                    },
                },
                "runtime": {"use_reranker": False},
                "paths": {},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_model_config_resolves_hf_repo_paths_under_models_root(tmp_path):
    config = load_model_stack_config(write_config(tmp_path))

    report = check_model_paths(config, tmp_path)

    assert report["extractor"]["path"] == str(tmp_path / "Qwen" / "Qwen3-8B")
    assert report["classifier"]["path"] == str(tmp_path / "Qwen" / "Qwen3-4B-Instruct-2507")
    assert "reranker" not in report


def test_model_path_check_reports_existing_paths(tmp_path):
    config = load_model_stack_config(write_config(tmp_path))
    for role in active_model_roles(config):
        spec = config.spec(role)
        path = tmp_path / spec.model_id
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}", encoding="utf-8")

    report = check_model_paths(config, tmp_path)

    assert all(item["exists"] for item in report.values())
    assert all(item["has_config_json"] for item in report.values())


def test_optional_reranker_role_is_only_active_when_enabled(tmp_path):
    config = load_model_stack_config(write_config(tmp_path))

    assert "reranker" not in active_model_roles(config)
    assert "reranker" in active_model_roles(config, include_optional=True)

    enabled_path = write_config(tmp_path)
    payload = json.loads(enabled_path.read_text(encoding="utf-8"))
    payload["runtime"]["use_reranker"] = True
    enabled_path.write_text(json.dumps(payload), encoding="utf-8")
    enabled = load_model_stack_config(enabled_path)

    assert "reranker" in active_model_roles(enabled)
