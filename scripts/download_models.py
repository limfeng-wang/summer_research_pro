"""Download configured Hugging Face models without running inference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("configs/model_stack.yaml")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download configured Hugging Face model snapshots.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Model stack YAML config")
    parser.add_argument("--cache-dir", default="", help="Optional Hugging Face cache directory")
    parser.add_argument(
        "--models-root",
        default="",
        help="Optional root directory. Each model is stored as <models-root>/<org>/<repo>.",
    )
    parser.add_argument("--token", default="", help="Optional Hugging Face token; otherwise HF_TOKEN is used")
    parser.add_argument(
        "--use-xet",
        action="store_true",
        help="Use Hugging Face Xet storage backend. Disabled by default to avoid CAS auth failures.",
    )
    args = parser.parse_args()

    if not args.use_xet:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

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
        if args.token:
            kwargs["token"] = args.token
        if args.models_root:
            model_dir = Path(args.models_root) / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            kwargs["local_dir"] = str(model_dir)
            kwargs["local_dir_use_symlinks"] = False
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
