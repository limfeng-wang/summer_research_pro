"""Model stack configuration and local path resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path("configs/model_stack.yaml")
DEFAULT_MODELS_ROOT = Path("/hdd-storage/lawrencelcty/huggingface/models")
OPTIONAL_MODEL_ROLES = {"reranker"}


@dataclass(frozen=True)
class ModelSpec:
    """One model role from the stack config."""

    role: str
    model_id: str
    backend: str
    quantization: str = ""
    device_policy: str = "auto"

    def local_path(self, models_root: str | Path = DEFAULT_MODELS_ROOT) -> Path:
        return Path(models_root) / self.model_id


@dataclass(frozen=True)
class ModelStackConfig:
    """Loaded model stack and runtime paths."""

    specs: dict[str, ModelSpec]
    runtime: dict[str, Any]
    paths: dict[str, str]

    def spec(self, role: str) -> ModelSpec:
        try:
            return self.specs[role]
        except KeyError as exc:
            raise ValueError(f"Unknown model role: {role}") from exc


def load_model_stack_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ModelStackConfig:
    """Load model stack YAML or JSON configuration."""

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("PyYAML is required to load YAML model stack config") from exc
        payload = yaml.safe_load(text)
    specs = {
        role: ModelSpec(
            role=spec.get("role", role),
            model_id=spec["model_id"],
            backend=spec.get("backend", ""),
            quantization=spec.get("quantization", ""),
            device_policy=spec.get("device_policy", "auto"),
        )
        for role, spec in payload.get("model_stack", {}).items()
    }
    return ModelStackConfig(
        specs=specs,
        runtime=payload.get("runtime", {}),
        paths=payload.get("paths", {}),
    )


def check_model_paths(config: ModelStackConfig, models_root: str | Path = DEFAULT_MODELS_ROOT) -> dict[str, dict[str, object]]:
    """Return existence checks for all configured local model paths."""

    report: dict[str, dict[str, object]] = {}
    for role in active_model_roles(config):
        spec = config.spec(role)
        path = spec.local_path(models_root)
        report[role] = {
            "model_id": spec.model_id,
            "path": str(path),
            "exists": path.exists(),
            "has_config_json": (path / "config.json").exists(),
        }
    return report


def active_model_roles(config: ModelStackConfig, *, include_optional: bool = False) -> list[str]:
    """Return model roles required by the active runtime configuration."""

    roles = []
    for role in config.specs:
        if role in OPTIONAL_MODEL_ROLES and not include_optional and not _optional_role_enabled(config, role):
            continue
        roles.append(role)
    return roles


def _optional_role_enabled(config: ModelStackConfig, role: str) -> bool:
    if role == "reranker":
        return bool(config.runtime.get("use_reranker", False))
    return True


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_MODELS_ROOT",
    "ModelSpec",
    "ModelStackConfig",
    "active_model_roles",
    "check_model_paths",
    "load_model_stack_config",
]
