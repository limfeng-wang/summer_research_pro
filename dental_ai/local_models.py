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
        payload = _extract_label_payload(text, ["relevance_label"])
        return RelevanceLabel(payload["relevance_label"])


class LocalR1Classifier:
    def __init__(self, lm: LocalCausalLM):
        self.lm = lm

    def classify_r1(self, post: SourcePost) -> tuple[ExperiencerLabel, ContentFunctionLabel]:
        text = self.lm.generate_json_text(
            R1_CLASSIFICATION_PROMPT,
            _post_payload(post),
            GenerationConfig(max_new_tokens=192),
        )
        payload = _extract_label_payload(text, ["experiencer_label", "content_function"])
        experiencer = ExperiencerLabel(payload["experiencer_label"])
        content_function = ContentFunctionLabel(payload["content_function"])
        return apply_classification_safeguards(post, experiencer, content_function)


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


def _extract_label_payload(text: str, required_keys: list[str]) -> dict[str, Any]:
    """Extract small classification labels from complete or truncated JSON.

    Local LMs sometimes finish after emitting the two labels but before closing
    long evidence strings. For classification-only smoke tests, the labels are
    the contract; evidence fields are diagnostic and should not crash the run.
    """

    try:
        payload = _extract_json_object(text)
    except (json.JSONDecodeError, ValueError):
        payload = {}
        for key in required_keys:
            match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]+)"', text)
            if match:
                payload[key] = match.group(1)

    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f"Model output missing required label(s) {missing}: {text[:500]!r}")
    return payload


def apply_classification_safeguards(
    post: SourcePost,
    experiencer: ExperiencerLabel,
    content_function: ContentFunctionLabel,
) -> tuple[ExperiencerLabel, ContentFunctionLabel]:
    """Apply deterministic, auditable safeguards to common boundary errors."""

    content_function = apply_commercial_safeguard(post, content_function)
    content_function = apply_weak_commercial_demote_safeguard(post, content_function)
    content_function = apply_help_seeking_safeguard(post, content_function)
    experiencer = apply_generic_knowledge_experiencer_safeguard(post, experiencer, content_function)
    return experiencer, content_function


def has_strong_commercial_evidence(post: SourcePost) -> bool:
    """Return whether source text has enough evidence for C4.

    C4 needs a promotional target plus promotional stance. This protects
    ordinary care logistics such as fees, registration, and hospital process
    sharing from being mislabeled as advertising.
    """

    text = post.combined_source_text
    lower_text = text.lower()

    product_targets = [
        "芬必得",
        "布洛芬",
        "对乙酰氨基酚",
        "novashine",
        "医疗器械",
        "药",
        "颗粒",
        "止痛药",
        "ibuprofen",
        "acetaminophen",
    ]
    product_promo_stance = [
        "不愧是",
        "大品牌",
        "全家都很信任",
        "外卖了",
        "每包含有",
        "口味",
        "包装",
        "购买",
        "下单",
        "推荐",
        "trusted",
        "recommend",
    ]

    clinic_targets = [
        "口腔门诊",
        "口腔诊所",
        "口腔医院",
        "口腔医学中心",
        "牙科",
        "치과",
        "歯科",
        "クリニック",
        "clinic",
        "dental",
    ]
    clinic_promo_stance = [
        "预约咨询",
        "咨询预约",
        "私信",
        "预约",
        "咨询",
        "到院",
        "门店",
        "套餐",
        "优惠",
        "折扣",
        "看牙",
        "booking",
        "book",
        "consult",
        "discount",
    ]

    explicit_ad = any(cue in lower_text for cue in ["广告", "推广", "合作", "赞助", "ad", "sponsored", "pr"])
    product_ad = any(cue.lower() in lower_text for cue in product_targets) and any(
        cue.lower() in lower_text for cue in product_promo_stance
    )
    clinic_account_targets = clinic_targets + ["口腔"]
    clinic_account_ad = "@" in text and any(cue.lower() in lower_text for cue in clinic_account_targets)
    clinic_conversion_ad = any(cue.lower() in lower_text for cue in clinic_targets) and any(
        cue.lower() in lower_text for cue in clinic_promo_stance
    )
    promo_hashtags = text.count("#") >= 3 and any(
        cue in text for cue in ["#芬必得", "#牙痛止痛药", "#上海看牙", "#看牙", "#口腔护理", "#novashine", "#Novashine"]
    )

    return explicit_ad or product_ad or clinic_account_ad or clinic_conversion_ad or promo_hashtags


def apply_commercial_safeguard(post: SourcePost, label: ContentFunctionLabel) -> ContentFunctionLabel:
    """Promote obvious promotional posts to C4.

    This deliberately does not demote C4. It only catches strong commercial
    signals that are common in Xiaohongshu advertorials and clinic/service
    posts, where models often confuse C4 with C1/C3.
    """

    if label not in {ContentFunctionLabel.C1, ContentFunctionLabel.C3}:
        return label
    return ContentFunctionLabel.C4 if has_strong_commercial_evidence(post) else label


def apply_weak_commercial_demote_safeguard(post: SourcePost, label: ContentFunctionLabel) -> ContentFunctionLabel:
    """Demote weak C4 guesses when the source lacks commercial evidence."""

    if label != ContentFunctionLabel.C4 or has_strong_commercial_evidence(post):
        return label

    text = post.combined_source_text
    first_person_cues = ["我", "我的", "本人", "作者本人", "第一次", "第二次", "私", "僕", "俺", "나는", "제가", "내 "]
    care_logistics_cues = ["挂号", "签到", "拍片", "缴费", "手术费", "医保", "报销", "麻醉", "拔完", "拔牙后", "价格"]
    general_knowledge_cues = ["什么是", "常见病因", "症状", "护理", "治疗", "方法", "适用于", "建议"]

    if any(cue in text for cue in first_person_cues) and any(cue in text for cue in care_logistics_cues):
        return ContentFunctionLabel.C1
    if any(cue in text for cue in general_knowledge_cues):
        return ContentFunctionLabel.C3
    return label


def apply_help_seeking_safeguard(post: SourcePost, label: ContentFunctionLabel) -> ContentFunctionLabel:
    """Promote genuine advice requests to C2 unless commercial content dominates."""

    if label == ContentFunctionLabel.C4:
        return label
    text = post.combined_source_text
    question_mark = "?" in text or "？" in text
    advice_cues = [
        "怎么办",
        "怎么缓解",
        "吃什么药",
        "要不要",
        "该不该",
        "求助",
        "救救",
        "有没有人",
        "どうしたら",
        "どうすれば",
        "해야",
        "어떡",
        "추천",
    ]
    rhetorical_title_cues = ["办法", "方法", "科普", "一篇说清楚", "急救办法"]
    if any(cue in text for cue in advice_cues) and question_mark:
        if not any(cue in post.original_title for cue in rhetorical_title_cues):
            return ContentFunctionLabel.C2
    return label


def apply_generic_knowledge_experiencer_safeguard(
    post: SourcePost,
    experiencer: ExperiencerLabel,
    content_function: ContentFunctionLabel,
) -> ExperiencerLabel:
    """Demote obvious general knowledge/commercial posts without a specific experiencer to E3."""

    if experiencer == ExperiencerLabel.E2:
        return experiencer
    if content_function not in {ContentFunctionLabel.C3, ContentFunctionLabel.C4}:
        return experiencer
    text = post.combined_source_text
    first_person_cues = ["我", "我的", "本人", "うち", "私", "僕", "俺", "나는", "제가", "내 "]
    specific_other_cues = ["妈妈", "爸爸", "孩子", "女儿", "儿子", "娘", "旦那", "친구", "엄마", "아빠"]
    general_cues = ["什么是", "常见病因", "症状", "护理", "治疗", "方法", "tips", "小妙招", "适用于", "建议"]
    if any(cue in text for cue in first_person_cues + specific_other_cues):
        return experiencer
    if any(cue in text for cue in general_cues):
        return ExperiencerLabel.E3
    return experiencer


__all__ = [
    "GenerationConfig",
    "LocalCSMExtractor",
    "LocalCausalLM",
    "LocalJudge",
    "LocalR1Classifier",
    "LocalRelevanceClassifier",
    "apply_classification_safeguards",
    "apply_commercial_safeguard",
    "apply_generic_knowledge_experiencer_safeguard",
    "apply_help_seeking_safeguard",
    "apply_weak_commercial_demote_safeguard",
    "has_strong_commercial_evidence",
    "local_lm_for_role",
]
