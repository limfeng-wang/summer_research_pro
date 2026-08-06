"""Local Hugging Face model clients for the hierarchical pipeline."""

from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dental_ai.model_config import DEFAULT_MODELS_ROOT, ModelStackConfig
from dental_ai.prompts import CSM_EXTRACTION_PROMPT, JUDGE_PROMPT, R1_CLASSIFICATION_PROMPT, RELEVANCE_PROMPT
from dental_ai.schemas import (
    ContentFunctionLabel,
    ExperiencerLabel,
    ExtractionResult,
    RelevanceLabel,
    SourcePost,
)


@dataclass
class GenerationConfig:
    """Small generation defaults for structured outputs."""

    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0


class LocalCausalLM:
    """Lazy local causal LM wrapper.

    The model is loaded only when `generate_json_text` is called. Call `close`
    before loading another large model on small GPUs.
    """

    def __init__(self, model_path: str | Path, *, quantization: str = "4bit", device_map: str = "auto"):
        self.model_path = str(model_path)
        self.quantization = quantization
        self.device_map = device_map
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def generate_json_text(self, system_prompt: str, user_payload: dict[str, Any], config: GenerationConfig) -> str:
        self._ensure_loaded()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        kwargs: dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.temperature > 0,
            "temperature": config.temperature if config.temperature > 0 else None,
            "top_p": config.top_p if config.temperature > 0 else None,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        outputs = self._model.generate(**inputs, **kwargs)
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quantization_config = None
        if self.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            device_map=self.device_map,
            quantization_config=quantization_config,
        )


class LocalRelevanceClassifier:
    def __init__(self, lm: LocalCausalLM):
        self.lm = lm

    def classify_relevance(self, post: SourcePost) -> RelevanceLabel:
        text = self.lm.generate_json_text(
            RELEVANCE_PROMPT,
            _post_payload(post),
            GenerationConfig(max_new_tokens=64),
        )
        payload = _extract_json_object(text)
        return RelevanceLabel(payload["relevance_label"])


class LocalR1Classifier:
    def __init__(self, lm: LocalCausalLM):
        self.lm = lm

    def classify_r1(self, post: SourcePost) -> tuple[ExperiencerLabel, ContentFunctionLabel]:
        text = self.lm.generate_json_text(
            R1_CLASSIFICATION_PROMPT,
            _post_payload(post),
            GenerationConfig(max_new_tokens=128),
        )
        payload = _extract_json_object(text)
        return ExperiencerLabel(payload["experiencer_label"]), ContentFunctionLabel(payload["content_function"])


class LocalCSMExtractor:
    def __init__(self, lm: LocalCausalLM):
        self.lm = lm

    def extract_csm(self, post: SourcePost, seed_examples: list[ExtractionResult]) -> ExtractionResult:
        text = self.lm.generate_json_text(
            CSM_EXTRACTION_PROMPT,
            {
                "post": _post_payload(post),
                "seed_examples": [example.model_dump(mode="json") for example in seed_examples],
            },
            GenerationConfig(max_new_tokens=2048),
        )
        payload = _extract_json_object(text)
        return ExtractionResult.model_validate(payload)


class LocalJudge:
    def __init__(self, lm: LocalCausalLM):
        self.lm = lm

    def judge(self, post: SourcePost, result: ExtractionResult) -> ExtractionResult:
        text = self.lm.generate_json_text(
            JUDGE_PROMPT,
            {
                "post": _post_payload(post),
                "candidate_annotation": result.model_dump(mode="json"),
            },
            GenerationConfig(max_new_tokens=2048),
        )
        payload = _extract_json_object(text)
        return ExtractionResult.model_validate(payload)


def local_lm_for_role(
    stack: ModelStackConfig,
    role: str,
    *,
    models_root: str | Path = DEFAULT_MODELS_ROOT,
) -> LocalCausalLM:
    spec = stack.spec(role)
    return LocalCausalLM(
        spec.local_path(models_root),
        quantization=spec.quantization or "4bit",
        device_map=spec.device_policy,
    )


def _post_payload(post: SourcePost) -> dict[str, Any]:
    return {
        "post_id": post.post_id,
        "country": post.country.value,
        "language": post.language.value,
        "platform": post.platform,
        "original_title": post.original_title,
        "original_text": post.original_text,
        "text_clean": post.text_clean,
        "analysis_text_en": post.analysis_text_en,
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Model output did not contain a JSON object: {text[:500]!r}")
    return json.loads(match.group(0))


__all__ = [
    "GenerationConfig",
    "LocalCSMExtractor",
    "LocalCausalLM",
    "LocalJudge",
    "LocalR1Classifier",
    "LocalRelevanceClassifier",
    "local_lm_for_role",
]
