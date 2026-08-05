"""Download configured Hugging Face models without running inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("configs/model_stack.yaml")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download configured Hugging Face model snapshots.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Model stack YAML config")
    parser.add_argument("--cache-dir", default="", help="Optional Hugging Face cache directory")
    parser.add_argument("--local-dir", default="", help="Optional directory for local snapshots")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
        import yaml
    except Exception as exc:
        raise SystemExit(f"Missing dependency. Run scripts/setup_h_ramos_env.sh first. Details: {exc}")

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    models = _model_ids(config)
    downloads: dict[str, str] = {}

    for role, model_id in models.items():
        kwargs: dict[str, Any] = {}
        if args.cache_dir:
            kwargs["cache_dir"] = args.cache_dir
        if args.local_dir:
            role_dir = Path(args.local_dir) / role
            role_dir.mkdir(parents=True, exist_ok=True)
            kwargs["local_dir"] = str(role_dir)
        path = snapshot_download(repo_id=model_id, **kwargs)
        downloads[role] = path
        print(f"{role}: {model_id} -> {path}")

    print(json.dumps(downloads, ensure_ascii=False, indent=2))
    return 0


def _model_ids(config: dict[str, Any]) -> dict[str, str]:
    stack = config.get("model_stack", {})
    return {
        role: spec["model_id"]
        for role, spec in stack.items()
        if isinstance(spec, dict) and spec.get("model_id")
    }


if __name__ == "__main__":
    raise SystemExit(main())
