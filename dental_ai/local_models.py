"""Local Hugging Face model clients for the hierarchical pipeline."""

from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dental_ai.classification_gold import ClassificationGoldRecord, canonical_post_id
from dental_ai.model_config import DEFAULT_MODELS_ROOT, ModelStackConfig
from dental_ai.prompts import CSM_EXTRACTION_PROMPT, JUDGE_PROMPT, R1_CLASSIFICATION_PROMPT, RELEVANCE_PROMPT
from dental_ai.schemas import (
    CSMDomain,
    ContentFunctionLabel,
    ExperiencerLabel,
    ExtractionResult,
    JudgeVerdict,
    RelevanceLabel,
    SupportType,
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
        self._processor: Any | None = None
        self._model: Any | None = None

    def generate_json_text(self, system_prompt: str, user_payload: dict[str, Any], config: GenerationConfig) -> str:
        self._ensure_loaded()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        if self._processor is not None:
            return self._generate_with_processor(messages, config)

        prompt = self._tokenizer_chat_prompt(messages)
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
        self._processor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _ensure_loaded(self) -> None:
        if self._model is not None and (self._tokenizer is not None or self._processor is not None):
            return

        if self._uses_processor_loader():
            self._ensure_processor_model_loaded()
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quantization_config = self._quantization_config()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            device_map=self.device_map,
            quantization_config=quantization_config,
        )

    def _ensure_processor_model_loaded(self) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        quantization_config = self._quantization_config()
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        model_kwargs = {
            "local_files_only": True,
            "trust_remote_code": True,
            "device_map": self.device_map,
            "quantization_config": quantization_config,
        }
        try:
            from transformers import AutoModelForMultimodalLM

            model_cls = AutoModelForMultimodalLM
        except ImportError:
            model_cls = AutoModelForCausalLM
        except AttributeError:
            model_cls = AutoModelForCausalLM

        try:
            self._model = model_cls.from_pretrained(self.model_path, **model_kwargs)
        except ValueError as exc:
            if model_cls is AutoModelForCausalLM:
                raise
            self._model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        except TypeError as exc:
            if "quantization_config" not in str(exc):
                raise
            model_kwargs.pop("quantization_config", None)
            self._model = model_cls.from_pretrained(self.model_path, **model_kwargs)

    def _generate_with_processor(self, messages: list[dict[str, str]], config: GenerationConfig) -> str:
        prefix = self._processor_chat_prefix(messages)
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        try:
            inputs = self._processor.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            inputs = self._processor.apply_chat_template(messages, **kwargs)
        inputs = inputs.to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.temperature > 0,
            "temperature": config.temperature if config.temperature > 0 else None,
            "top_p": config.top_p if config.temperature > 0 else None,
        }
        generate_kwargs = {key: value for key, value in generate_kwargs.items() if value is not None}
        outputs = self._model.generate(**inputs, **generate_kwargs)
        response = self._processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        if hasattr(self._processor, "parse_response"):
            try:
                parsed = self._processor.parse_response(response, prefix=prefix)
                if isinstance(parsed, str):
                    return parsed.strip()
                if isinstance(parsed, dict):
                    for key in ("content", "text", "response"):
                        value = parsed.get(key)
                        if isinstance(value, str):
                            return value.strip()
                return str(parsed).strip()
            except Exception:
                return response.strip()
        return response.strip()

    def _processor_chat_prefix(self, messages: list[dict[str, str]]) -> str:
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        try:
            prefix = self._processor.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            prefix = self._processor.apply_chat_template(messages, **kwargs)
        if isinstance(prefix, str):
            return prefix
        return str(prefix)

    def _tokenizer_chat_prompt(self, messages: list[dict[str, str]]) -> str:
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        try:
            return self._tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self._tokenizer.apply_chat_template(messages, **kwargs)

    def _uses_processor_loader(self) -> bool:
        model_path = Path(self.model_path)
        return (model_path / "processor_config.json").exists() or "gemma-4" in self.model_path.lower()

    def _quantization_config(self) -> Any | None:
        from transformers import BitsAndBytesConfig

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
        return quantization_config


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
        return apply_relevance_safeguard(post, RelevanceLabel(payload["relevance_label"]))


class LocalR1Classifier:
    def __init__(
        self,
        lm: LocalCausalLM,
        *,
        classification_examples: list[ClassificationGoldRecord] | None = None,
        fewshot_k: int = 8,
    ):
        self.lm = lm
        self.classification_examples = classification_examples or []
        self.fewshot_k = fewshot_k

    def classify_r1(self, post: SourcePost) -> tuple[ExperiencerLabel, ContentFunctionLabel]:
        text = self.lm.generate_json_text(
            R1_CLASSIFICATION_PROMPT,
            {
                "post": _post_payload(post),
                "classification_gold_examples": _classification_fewshot_payload(
                    post,
                    self.classification_examples,
                    k=self.fewshot_k,
                ),
            },
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
        payload = _extract_csm_payload(text, post)
        result = ExtractionResult.model_validate(payload)
        return mark_units_needing_human_review(result)


class LocalJudge:
    def __init__(self, lm: LocalCausalLM):
        self.lm = lm

    def judge(self, post: SourcePost, result: ExtractionResult) -> ExtractionResult:
        text = self.lm.generate_json_text(
            JUDGE_PROMPT,
            {
                "post": _post_payload(post),
                "candidate_units": _judge_units_payload(result),
            },
            GenerationConfig(max_new_tokens=768),
        )
        payload = _extract_judge_verdict_payload(text)
        judged = apply_judge_verdict_payload(result, payload)
        return apply_deterministic_csm_safeguards(judged)


def mark_units_needing_human_review(result: ExtractionResult) -> ExtractionResult:
    """Ensure extractor-only units are not counted as judge-accepted units."""

    units = [
        unit.model_copy(update={"judge_verdict": JudgeVerdict.NEEDS_HUMAN_REVIEW})
        for unit in result.units
    ]
    return result.model_copy(update={"units": units})


def apply_judge_verdict_payload(result: ExtractionResult, payload: dict[str, Any]) -> ExtractionResult:
    """Apply compact judge verdicts to an extraction result."""

    verdicts = {}
    for item in payload.get("unit_verdicts", []):
        if not isinstance(item, dict):
            continue
        unit_id = item.get("unit_id")
        verdict = item.get("judge_verdict")
        if not unit_id or not verdict:
            continue
        try:
            verdicts[str(unit_id)] = JudgeVerdict(str(verdict))
        except ValueError:
            continue

    units = [
        unit.model_copy(update={"judge_verdict": verdicts.get(unit.unit_id, JudgeVerdict.NEEDS_HUMAN_REVIEW)})
        for unit in result.units
    ]
    return result.model_copy(update={"units": units})


def apply_deterministic_csm_safeguards(result: ExtractionResult) -> ExtractionResult:
    """Reject rule-detectable non-CSM units after model judging."""

    units = []
    for unit in result.units:
        if _is_rule_rejected_csm_unit(unit):
            units.append(
                unit.model_copy(
                    update={
                        "judge_verdict": JudgeVerdict.REJECT,
                        "support_type": SupportType.UNSUPPORTED,
                    }
                )
            )
        else:
            units.append(unit)
    return result.model_copy(update={"units": units})


def _is_rule_rejected_csm_unit(unit: Any) -> bool:
    text = " ".join(
        [
            unit.evidence_span_original,
            unit.surface_text_working,
            unit.normalized_concept_en,
        ]
    ).lower()
    if _has_negated_no_pain_signal(text):
        return True
    if _is_pure_admin_or_cost_unit(text):
        return True
    if unit.domain == CSMDomain.PERCEIVED_CAUSE and _has_diagnosis_without_pain_link(text):
        return True
    return False


def _has_negated_no_pain_signal(text: str) -> bool:
    return any(
        cue in text
        for cue in [
            "不痛",
            "不疼",
            "无痛",
            "没发炎",
            "没有发炎",
            "no pain",
            "no inflammation",
            "痛みなし",
            "痛くない",
            "아프지",
            "통증 없음",
        ]
    )


def _is_pure_admin_or_cost_unit(text: str) -> bool:
    admin_or_cost = [
        "挂号",
        "预约",
        "签到",
        "加号",
        "检查费",
        "手术费",
        "费用",
        "价格",
        "医保",
        "报销",
        "牙片",
        "拍片",
        "appointment",
        "registration",
        "insurance",
        "fee",
        "cost",
        "price",
        "x-ray",
        "予約",
        "受付",
        "費用",
        "料金",
        "保険",
        "예약",
        "접수",
        "비용",
        "가격",
        "보험",
    ]
    pain_or_barrier = [
        "牙疼",
        "牙痛",
        "疼痛",
        "痛",
        "发炎",
        "肿",
        "止痛",
        "镇痛",
        "忍不了",
        "睡不着",
        "吃不了",
        "负担",
        "太贵",
        "承担不起",
        "barrier",
        "burden",
        "pain",
        "swelling",
        "inflammation",
        "痛み",
        "腫れ",
        "치통",
        "통증",
        "붓",
    ]
    return any(cue in text for cue in admin_or_cost) and not any(cue in text for cue in pain_or_barrier)


def _has_diagnosis_without_pain_link(text: str) -> bool:
    diagnosis = [
        "阻生齿",
        "阻生智齿",
        "龋齿",
        "impacted",
        "caries",
        "埋伏",
        "虫歯",
        "매복",
        "충치",
    ]
    pain_link = [
        "牙疼",
        "牙痛",
        "疼",
        "痛",
        "发炎",
        "肿",
        "pain",
        "inflammation",
        "swelling",
        "痛み",
        "腫れ",
        "통증",
        "아프",
        "붓",
    ]
    return any(cue in text for cue in diagnosis) and not any(cue in text for cue in pain_link)


def _extract_judge_verdict_payload(text: str) -> dict[str, Any]:
    """Extract compact judge verdicts from strict or mildly malformed JSON."""

    try:
        return _extract_json_object(text)
    except json.JSONDecodeError:
        recovered = _recover_judge_verdicts(text)
        if recovered:
            return {"unit_verdicts": recovered}
        raise


def _recover_judge_verdicts(text: str) -> list[dict[str, str]]:
    unit_matches = list(re.finditer(r'"unit_id"\s*:\s*"([^"]+)"', text))
    recovered = []
    for index, match in enumerate(unit_matches):
        start = match.start()
        end = unit_matches[index + 1].start() if index + 1 < len(unit_matches) else min(len(text), start + 1200)
        window = text[start:end]
        verdict_match = re.search(
            r'"judge_verdict"\s*:\s*"(accept|revise|reject|needs_human_review)"',
            window,
        )
        if not verdict_match:
            continue
        recovered.append(
            {
                "unit_id": match.group(1),
                "judge_verdict": verdict_match.group(1),
            }
        )
    return recovered


def _extract_csm_payload(text: str, post: SourcePost) -> dict[str, Any]:
    """Extract CSM payload from strict or mildly malformed model JSON."""

    try:
        return _extract_json_object(text)
    except json.JSONDecodeError:
        recovered_units = _recover_csm_units(text)
        if recovered_units:
            return {
                "post_id": post.post_id,
                "country": post.country.value,
                "language": post.language.value,
                "units": recovered_units,
            }
        raise


def _recover_csm_units(text: str) -> list[dict[str, Any]]:
    text = _strip_thinking_text(text)
    unit_payloads = []
    for obj in _iter_json_objects(text):
        if _looks_like_csm_unit(obj):
            unit_payloads.append(obj)
    return unit_payloads


def _iter_json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects = []
    for start in [match.start() for match in re.finditer(r"\{", text)]:
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            objects.append(obj)
    return objects


def _looks_like_csm_unit(obj: dict[str, Any]) -> bool:
    required_keys = {
        "domain",
        "evidence_span_original",
        "surface_text_working",
        "normalized_concept_en",
        "concept_status",
        "support_type",
        "assertion",
        "confidence",
    }
    return required_keys.issubset(obj)


def _judge_units_payload(result: ExtractionResult) -> list[dict[str, Any]]:
    return [
        {
            "unit_id": unit.unit_id,
            "domain": unit.domain.value,
            "evidence_span_original": unit.evidence_span_original,
            "surface_text_working": unit.surface_text_working,
            "normalized_concept_en": unit.normalized_concept_en,
            "support_type": unit.support_type.value,
            "assertion": unit.assertion.value,
            "temporality": unit.temporality.value,
            "sentiment_or_outcome": unit.sentiment_or_outcome.value,
        }
        for unit in result.units
    ]


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


def _classification_fewshot_payload(
    post: SourcePost,
    examples: list[ClassificationGoldRecord],
    *,
    k: int,
) -> list[dict[str, Any]]:
    """Select compact classification gold examples without using eval/main rows."""

    if not examples or k <= 0:
        return []

    post_canonical_id = canonical_post_id(post.post_id)
    same_language = [
        example
        for example in examples
        if example.language == post.language
        and not (example.country == post.country and example.canonical_id == post_canonical_id)
    ]
    fallback = [
        example
        for example in examples
        if example not in same_language
        and not (example.country == post.country and example.canonical_id == post_canonical_id)
    ]

    selected: list[ClassificationGoldRecord] = []
    seen_combos: set[tuple[ExperiencerLabel, ContentFunctionLabel]] = set()
    for pool in [same_language, fallback]:
        for example in pool:
            combo = (example.experiencer_label, example.content_function)
            if combo in seen_combos:
                continue
            selected.append(example)
            seen_combos.add(combo)
            if len(selected) >= k:
                return [_classification_example_payload(example) for example in selected]
        for example in pool:
            if example in selected:
                continue
            selected.append(example)
            if len(selected) >= k:
                return [_classification_example_payload(example) for example in selected]

    return [_classification_example_payload(example) for example in selected]


def _classification_example_payload(example: ClassificationGoldRecord) -> dict[str, Any]:
    text = example.combined_source_text
    return {
        "country": example.country.value,
        "language": example.language.value,
        "source_excerpt": text[:900],
        "experiencer_label": example.experiencer_label.value,
        "content_function": example.content_function.value,
        "evidence_excerpt": example.evidence_excerpt[:500],
        "rationale": example.rationale[:300],
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    text = _strip_thinking_text(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Model output did not contain a JSON object: {text[:500]!r}")
    return json.loads(match.group(0))


def _strip_thinking_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^\s*<think>.*?(?=\{|\[|$)", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


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

    experiencer = apply_specific_experiencer_safeguard(post, experiencer)
    content_function = apply_commercial_safeguard(post, content_function)
    content_function = apply_weak_commercial_demote_safeguard(post, content_function)
    content_function = apply_help_seeking_safeguard(post, content_function)
    content_function = apply_personal_narrative_safeguard(post, experiencer, content_function)
    experiencer = apply_generic_knowledge_experiencer_safeguard(post, experiencer, content_function)
    return experiencer, content_function


def apply_relevance_safeguard(post: SourcePost, label: RelevanceLabel) -> RelevanceLabel:
    """Protect R1 from oral-ulcer-only and generic oral-health leakage."""

    if label != RelevanceLabel.R1:
        return label
    return RelevanceLabel.R1 if has_toothache_relevance_evidence(post) else RelevanceLabel.R0


def has_toothache_relevance_evidence(post: SourcePost) -> bool:
    """Return whether the text concerns tooth/dental pain or pain-related care."""

    text = post.combined_source_text
    pain_cues = [
        "牙疼",
        "牙痛",
        "牙龈肿痛",
        "智齿痛",
        "咬物痛",
        "咬合痛",
        "冷热刺激痛",
        "自发痛",
        "夜间痛",
        "歯が痛",
        "歯痛",
        "치통",
        "이가 아",
    ]
    pain_related_care_cues = [
        "牙髓炎",
        "根管",
        "拔牙",
        "拔智齿",
        "补牙",
        "蛀牙",
        "龋",
        "智齿发炎",
        "冠周炎",
        "干槽症",
        "牙神经",
        "止痛",
        "止疼",
        "麻醉",
        "根尖",
    ]
    oral_ulcer_cues = ["口腔溃疡", "溃疡", "口内炎", "구내염"]
    if any(cue in text for cue in oral_ulcer_cues) and not any(cue in text for cue in pain_cues + pain_related_care_cues):
        return False
    return any(cue in text for cue in pain_cues + pain_related_care_cues)


def apply_specific_experiencer_safeguard(post: SourcePost, label: ExperiencerLabel) -> ExperiencerLabel:
    """Promote missed specific experiencers when explicit case evidence exists."""

    if label != ExperiencerLabel.E3:
        return label
    text = post.combined_source_text
    if has_specific_other_case(text):
        return ExperiencerLabel.E2
    if has_specific_author_case(text):
        return ExperiencerLabel.E1
    return label


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
        "欧乐b",
        "欧乐B",
        "美团",
        "喷剂",
        "保健液",
        "电动牙刷",
        "磁波刷",
        "树脂补牙",
        "补牙615",
        "医疗器械",
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
        "入手",
        "抢",
        "使用方式",
        "味道温和",
        "平价",
        "好用",
        "智能",
        "自动",
        "更干净",
        "新科技",
        "护龈",
        "分享给大家",
        "价格透明",
        "无隐形消费",
        "限指定门店",
        "trusted",
        "recommend",
    ]

    clinic_targets = [
        "口腔门诊",
        "口腔诊所",
        "口腔医院",
        "口腔医学中心",
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

    explicit_ad = any(cue in lower_text for cue in ["广告", "推广", "合作", "赞助", "医广", "ad", "sponsored", "pr"])
    product_ad = any(cue.lower() in lower_text for cue in product_targets) and any(
        cue.lower() in lower_text for cue in product_promo_stance
    )
    clinic_account_targets = clinic_targets + ["口腔"]
    clinic_account_ad = "@" in text and any(cue.lower() in lower_text for cue in clinic_account_targets)
    named_clinic = has_named_clinic_or_service_tag(text)
    clinic_conversion_ad = named_clinic and any(
        cue.lower() in lower_text for cue in clinic_promo_stance
    )
    dense_named_service_hashtags = named_clinic and text.count("#") >= 5 and any(
        cue in text for cue in ["#看牙", "#深圳看牙", "#上海看牙", "#洁牙", "#补牙", "#种牙", "#美团医疗"]
    )
    promo_hashtags = text.count("#") >= 3 and any(
        cue in text for cue in ["#芬必得", "#牙痛止痛药", "#上海看牙", "#看牙", "#novashine", "#Novashine", "#美团医疗"]
    )

    return explicit_ad or product_ad or clinic_account_ad or clinic_conversion_ad or dense_named_service_hashtags or promo_hashtags


def has_named_clinic_or_service_tag(text: str) -> bool:
    """Detect named clinic/service promotion without broad health hashtag leakage."""

    broad_tags = {
        "口腔健康",
        "口腔护理",
        "口腔管理",
        "口腔清洁",
        "口腔健康科普",
        "牙齿护理",
        "牙齿修复攻略",
        "牙齿保护计划",
        "天然牙保护",
        "公立私立牙科",
        "儿童口腔",
        "口腔日常护理",
        "口腔",
        "成都口腔",
        "口腔挂号攻略",
        "看牙医",
        "口腔医学生",
        "医生日常",
    }
    for tag in re.findall(r"[#@]([^#@\s]+)", text):
        normalized = tag.strip()
        if normalized in broad_tags:
            continue
        if "口腔" in normalized:
            prefix = normalized.split("口腔", 1)[0]
            if len(prefix) >= 3:
                return True
            continue
        if any(cue in normalized for cue in ["牙美家", "牙科", "歯科", "치과", "clinic", "dental"]):
            return True
    return False


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
    first_person_cues = ["我", "我的", "本人", "作者本人", "第一次", "第二次", "나는", "제가", "내 "]
    care_logistics_cues = ["挂号", "签到", "拍片", "缴费", "手术费", "医保", "报销", "麻醉", "拔完", "拔牙后", "价格"]
    general_knowledge_cues = [
        "什么是",
        "常见病因",
        "症状",
        "护理",
        "治疗",
        "方法",
        "适用于",
        "建议",
        "预防",
        "保护牙齿",
        "刷牙",
        "牙线",
    ]

    if any(cue in text for cue in first_person_cues) and any(cue in text for cue in care_logistics_cues):
        return ContentFunctionLabel.C1
    if any(cue in text for cue in general_knowledge_cues):
        return ContentFunctionLabel.C3
    return label


def apply_help_seeking_safeguard(post: SourcePost, label: ContentFunctionLabel) -> ContentFunctionLabel:
    """Promote genuine advice requests to C2 and demote rhetorical C2."""

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
    rhetorical_title_cues = ["办法", "方法", "科普", "一篇说清楚", "急救办法", "一张图看懂", "指南", "攻略"]
    genuine_request = any(cue in text for cue in advice_cues) and question_mark and not any(
        cue in post.original_title for cue in rhetorical_title_cues
    )
    if genuine_request:
        return ContentFunctionLabel.C2
    if label == ContentFunctionLabel.C2:
        return ContentFunctionLabel.C3
    return label


def apply_personal_narrative_safeguard(
    post: SourcePost,
    experiencer: ExperiencerLabel,
    label: ContentFunctionLabel,
) -> ContentFunctionLabel:
    """Keep personal treatment/cost/process accounts in C1 unless promotional."""

    if label != ContentFunctionLabel.C3 or experiencer not in {ExperiencerLabel.E1, ExperiencerLabel.E2}:
        return label
    if has_strong_commercial_evidence(post):
        return ContentFunctionLabel.C4
    text = post.combined_source_text
    logistics = ["挂号", "签到", "拍片", "缴费", "手术费", "医保", "报销", "麻醉", "拔完", "拔牙后", "总共", "费用"]
    narrative = ["经历", "经验", "全程", "第一次", "第二次", "我去", "我上周", "前段时间我", "作者本人"]
    if any(cue in text for cue in logistics) and any(cue in text for cue in narrative):
        return ContentFunctionLabel.C1
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


def has_specific_author_case(text: str) -> bool:
    """Detect explicit first-person illness/dental-care events."""

    author_case_cues = [
        "我自己",
        "本人",
        "作者本人",
        "我上周",
        "我去",
        "我真的",
        "我坚持",
        "前段时间我",
        "根据我从小到大",
        "나는",
        "제가",
        "내 ",
    ]
    event_cues = [
        "牙疼",
        "牙痛",
        "牙龈",
        "智齿",
        "拔",
        "补牙",
        "根管",
        "发炎",
        "恢复",
        "治疗",
        "检查",
        "复查",
        "疼",
        "痛",
    ]
    return any(cue in text for cue in author_case_cues) and any(cue in text for cue in event_cues)


def has_specific_other_case(text: str) -> bool:
    """Detect explicit proxy experiencers."""

    other_cues = ["我妈", "我爸", "妈妈", "爸爸", "女儿", "儿子", "我家孩子", "家人牙", "朋友牙", "娘", "旦那", "친구", "엄마", "아빠"]
    event_cues = ["牙疼", "牙痛", "牙龈", "智齿", "拔", "补牙", "根管", "发炎", "疼", "痛"]
    return any(cue in text for cue in other_cues) and any(cue in text for cue in event_cues)


__all__ = [
    "GenerationConfig",
    "LocalCSMExtractor",
    "LocalCausalLM",
    "LocalJudge",
    "LocalR1Classifier",
    "LocalRelevanceClassifier",
    "apply_relevance_safeguard",
    "apply_classification_safeguards",
    "apply_commercial_safeguard",
    "apply_generic_knowledge_experiencer_safeguard",
    "apply_help_seeking_safeguard",
    "apply_personal_narrative_safeguard",
    "apply_specific_experiencer_safeguard",
    "apply_weak_commercial_demote_safeguard",
    "has_named_clinic_or_service_tag",
    "has_strong_commercial_evidence",
    "has_toothache_relevance_evidence",
    "local_lm_for_role",
]
