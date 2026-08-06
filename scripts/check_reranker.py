#!/usr/bin/env python
"""Isolated diagnostic for the configured pairwise reranker."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dental_ai.model_config import DEFAULT_MODELS_ROOT, load_model_stack_config
from dental_ai.rag import RerankerConfig, build_pair_reranker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check BGE reranker scoring in isolation.")
    parser.add_argument("--config", default="configs/model_stack.yaml")
    parser.add_argument("--models-root", default=str(DEFAULT_MODELS_ROOT))
    parser.add_argument("--backend", choices=("transformers", "flagembedding"), default="transformers")
    parser.add_argument("--device", default="", help="Override reranker device, e.g. cuda or cpu")
    parser.add_argument("--query", default="半夜牙疼睡不着, 吃了布洛芬还是疼")
    parser.add_argument(
        "--passage",
        action="append",
        default=None,
        help="Candidate passage; may be repeated.",
    )
    args = parser.parse_args(argv)

    stack = load_model_stack_config(args.config)
    model_path = stack.spec("reranker").local_path(args.models_root)
    reranker_config = RerankerConfig(
        backend=args.backend,
        batch_size=int(stack.runtime.get("reranker_batch_size", 1)),
        max_length=int(stack.runtime.get("reranker_max_length", 512)),
        use_fp16=bool(stack.runtime.get("reranker_use_fp16", True)),
        device=args.device or str(stack.runtime.get("reranker_device", "auto")),
    )
    if not Path(model_path).exists():
        raise SystemExit(f"reranker model path does not exist: {model_path}")

    start = time.perf_counter()
    reranker = build_pair_reranker(model_path, reranker_config)
    passages = args.passage or [
        "作者半夜牙疼, 无法入睡, 尝试布洛芬止痛。",
        "智齿拔除价格和挂号流程分享。",
    ]
    scores = reranker.score(args.query, passages)
    elapsed = time.perf_counter() - start
    print(
        json.dumps(
            {
                "model_path": str(model_path),
                "backend": reranker_config.backend,
                "batch_size": reranker_config.batch_size,
                "max_length": reranker_config.max_length,
                "use_fp16": reranker_config.use_fp16,
                "device": reranker_config.device,
                "elapsed_seconds": round(elapsed, 3),
                "query": args.query,
                "passages": passages,
                "scores": scores,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
