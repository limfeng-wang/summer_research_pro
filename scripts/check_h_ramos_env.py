"""Check the h-ramos environment before local model integration.

This script does not download or load project model weights. It only reports
Python/package/CUDA availability so setup issues are separated from pipeline
logic.
"""

from __future__ import annotations

import importlib
import json
import platform
import sys


PACKAGES = [
    "pydantic",
    "pandas",
    "openpyxl",
    "pytest",
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "sentence_transformers",
    "FlagEmbedding",
    "faiss",
]


def main() -> int:
    report = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {},
        "cuda": {},
    }

    for package in PACKAGES:
        report["packages"][package] = _package_status(package)

    torch_status = report["packages"].get("torch", {})
    if torch_status.get("available"):
        import torch

        report["cuda"] = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_gb": round(torch.cuda.get_device_properties(index).total_memory / 1024**3, 2),
                }
                for index in range(torch.cuda.device_count())
            ],
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    missing = [name for name, status in report["packages"].items() if not status["available"]]
    return 1 if missing else 0


def _package_status(package: str) -> dict[str, object]:
    try:
        module = importlib.import_module(package)
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    return {"available": True, "version": getattr(module, "__version__", "")}


if __name__ == "__main__":
    raise SystemExit(main())
