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

    if report["packages"].get("transformers", {}).get("available"):
        import transformers

        report["transformers_features"] = {
            "has_AutoProcessor": hasattr(transformers, "AutoProcessor"),
            "has_AutoModelForMultimodalLM": hasattr(transformers, "AutoModelForMultimodalLM"),
        }

    torch_status = report["packages"].get("torch", {})
    if torch_status.get("available"):
        import torch

        devices = []
        cuda_available = False
        device_count = 0
        cuda_error = ""
        try:
            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count()
            for index in range(device_count):
                try:
                    properties = torch.cuda.get_device_properties(index)
                    devices.append(
                        {
                            "index": index,
                            "name": properties.name,
                            "total_memory_gb": round(properties.total_memory / 1024**3, 2),
                        }
                    )
                except Exception as exc:
                    devices.append({"index": index, "error": str(exc)})
        except Exception as exc:
            cuda_error = str(exc)

        report["cuda"] = {
            "available": cuda_available,
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "device_count": device_count,
            "devices": devices,
        }
        if cuda_error:
            report["cuda"]["error"] = cuda_error

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
