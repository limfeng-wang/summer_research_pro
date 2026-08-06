"""Frozen prompt templates for source-first multilingual annotation."""

RELEVANCE_PROMPT = """\
You are screening multilingual social-media posts for a toothache narrative study.

Use the original source text as authoritative. Label only relevance:
- R1: explicitly concerns toothache, tooth/gingival pain, pain-related treatment, or response to a specific toothache experience.
- R0: unrelated keyword noise, unrelated reposting, hashtag stuffing, or ambiguous non-toothache use.
- RU: relevance cannot be determined reliably from the source text.

Boundary rules:
- Oral-ulcer-only, mouth-sore-only, generic oral hygiene, cosmetic dentistry, or general dental prevention posts are R0 unless they explicitly concern tooth/gingival pain or pain-related dental care.
- Dental procedures can be R1 when pain-related care is central: root canal, extraction, wisdom-tooth inflammation, pulpitis, dry socket, painful caries, pain relief, anesthesia, or painful recovery.

Return strict JSON:
{"relevance_label":"R1|R0|RU"}
"""


R1_CLASSIFICATION_PROMPT = """\
You are applying the frozen codebook for a multilingual toothache narrative study.
The post is already R1 toothache-relevant. Use original source text as authoritative.

The user payload contains:
- post: the target post to classify.
- classification_gold_examples: human-adjudicated R/E/C examples from the development/few-shot gold set. Use them only to calibrate boundary decisions. Do not copy their labels unless the target post has the same function/evidence pattern.

Return one experiencer label and one primary content-function label.
Do not extract CSM units in this task.

EXPERIENCER LABEL
- E1 Author: the author describes their own current/past toothache, symptoms, dental visit, treatment, recovery, or intended care. Omitted subject can still be E1 when context clearly points to the author.
- E2 Specific other person: a family member, friend, patient, spouse, child, or directly addressed person has the toothache/dental-pain experience.
- E3 No specific experiencer: general knowledge, product/service information, institutional content, jokes/general statements, or broad "patients/everyone/people" statements.

E-label safeguards:
- An @mention alone is not E2.
- Generic "patients", "everyone", "you", "people" is E3 unless a specific person is described.
- An advertorial written in first person can still be E1 if it claims the author's own pain/use; commercial status changes only C label, not E label.

CONTENT FUNCTION LABEL
- C1 Experience-sharing narrative: what happened to a specific experiencer is the main content: symptoms, timing, treatment, management, recovery, or outcome.
- C2 Question/help-seeking: the author genuinely asks readers for explanation, judgement, recommendation, or actionable advice affecting a decision.
- C3 Health knowledge sharing: reusable information about causes, symptoms, prevention, management, treatment, or care. It stands independently of one person's story.
- C4 Advertising/commercial: primary purpose is promotion or conversion for an identifiable commercial target: product, brand, drug, device, clinic, doctor/team, appointment, consultation, paid service, purchase channel, discount, package, or account traffic. Includes advertorials disguised as personal experience.
- C5 Other: toothache-related but low-information reaction, joke, wish, slogan, or brief interaction.

PRIMARY FUNCTION PRIORITY
C4 > C2 > C3 > C1 > C5.
Apply this priority only to content_function; do not change experiencer_label.

C4 requires BOTH:
1. a commercial target: named product/brand/drug/device/clinic/service/doctor account, purchase or booking channel, paid package, discount, consultation, or identifiable institutional account; AND
2. a promotional stance: recommendation, trust/quality claim, product features, dosage/taste/packaging, price/package selling point, appointment/booking/conversion language, account tagging, or service/location marketing.

C4 MUST override C1/C3 only when this commercial evidence is strong:
- repeated brand/product/clinic name plus favorable or purchase-oriented claims;
- product features, dose, taste, packaging, price, purchase channel, "big brand", "trusted", "recommended";
- clinic/service/location/account promotion with appointment, consultation, booking, conversion, or account-tag evidence;
- dense product/service hashtags or account tags that point to a brand/clinic/service;
- educational content posted mainly to promote a clinic/product.

C4 MUST NOT be used for ordinary care logistics by itself:
- cost-sharing, hospital registration, appointment process, X-ray fee, surgery fee, insurance reimbursement, or "which department/doctor" details are NOT C4 unless a specific clinic/product/service is being promoted.
- A personal treatment or extraction story with practical tips remains C1 when anchored in the author's own case and no commercial target is promoted.
- Broad health hashtags alone are NOT C4: #口腔健康, #口腔护理, #牙齿护理, #刷牙, #牙齿保护计划, generic city+口腔 tags.
- A named clinic/service/product hashtag or @account can support C4 only when paired with service/product/booking/selling/traffic evidence.

C2 must be genuine help-seeking:
- "怎么办/どうしたら/어떡해/should I..." plus a real request for advice -> C2.
- Rhetorical question in a title followed by education/ad/promotional content is C3 or C4, not C2.
- "一张图看懂", "指南", "攻略", "一篇说清楚", and "why/how" explainer titles are usually C3/C4, not C2, unless the author is truly asking readers for help.

C3 vs C1:
- If the post mainly tells the author's/specific person's own sequence of symptoms/actions/outcomes -> C1.
- If the post mainly teaches generally reusable information, even with "you/大家" language -> C3.
- Personal cost/process sharing from the author's own dental visit is C1 unless it becomes general unanchored education (C3) or promotion (C4).

C5:
- Use only after C1-C4 do not fit. Do not use C5 for irrelevant content; irrelevant content should have been R0 upstream.

COMPACT EXAMPLES
- "喝了一包芬必得小绿盒, 不愧是大品牌, #芬必得 #牙痛止痛药" -> E1 + C4.
- "牙痛急救办法: 冰敷、盐水漱口、及时就医" -> E3 + C3.
- "我牙疼三天睡不着, 明天去拔牙" -> E1 + C1.
- "我拔智齿两次, 第一次手术费975, 第二次1300, 分享挂号和术后注意事项" -> E1 + C1, not C4, because cost/process sharing alone is not promotion.
- "牙疼怎么办? 现在要不要去急诊?" -> E1 + C2.
- "我妈牙疼到吃不了饭" -> E2 + C1.
- "智齿冠周炎症状和治疗, #上海看牙 @某口腔" -> E3 + C4 when service/account promotion dominates.
- "牙痛痛痛!!!" -> E1 + C5 if it is only a brief reaction with no meaningful narrative or help request.

Return strict JSON only:
{
  "experiencer_label": "E1|E2|E3",
  "content_function": "C1|C2|C3|C4|C5",
  "experiencer_evidence": "short exact source phrase supporting E label, or empty for E3",
  "content_function_evidence": "short exact source phrase supporting C label"
}
"""


CSM_EXTRACTION_PROMPT = """\
You extract source-supported CSM narrative units from multilingual toothache posts.

Use original source text as authoritative. Chinese working-language glosses are allowed.
Every evidence_span_original must be copied exactly from the source text. No evidence span, no extraction.
Extract only claim units that describe a toothache/dental-pain CSM narrative.

CSM domains:
- Perceived Cause: stated or implied trigger/cause of the pain or painful dental condition.
- Symptom Description: pain sensations, swelling, inflammation, location, severity, timing, recurrence.
- Perceived Consequences: impact of pain/condition on sleep, eating, work, emotion, money, delay, function, or daily life.
- Coping and Management: actions taken/planned/recommended to manage pain/condition, including medicine, dental visits, procedures, self-care.
- Emotional Expression: fear, anxiety, frustration, relief, regret, trust, anger, embarrassment tied to the pain/condition.

Negative rules:
- Do not extract pure cost, registration, booking, insurance, hospital workflow, location, price, package, account, or appointment logistics unless the span explicitly frames it as a barrier, consequence, or coping decision for the painful condition.
- Do not put procedure cost or administrative information under Symptom Description.
- Do not put a diagnosis/tooth type under Perceived Cause unless the source links it to pain, inflammation, swelling, or treatment need.
- Do not extract generic tips from a mixed post when they are not tied to the specific author's/specific person's pain experience.
- If the post has no source-supported CSM unit, return "units": [].

For surface_text_working, write a short Chinese annotator-facing gloss.
Set working_language to "zh".
Set normalized_concept_en to a concise English concept. Do not introduce diagnoses or causal claims absent from the source text.
The extractor is not the judge: always set judge_verdict to "needs_human_review".

Return strict JSON matching:
{
  "post_id": "string",
  "country": "CHI|JPN|KOR",
  "language": "zh|ja|ko",
  "units": [
    {
      "domain": "...",
      "evidence_span_original": "exact source span",
      "surface_text_working": "Chinese working gloss",
      "working_language": "zh",
      "normalized_concept_en": "English concept",
      "concept_status": "existing_dictionary|new_candidate|unmapped",
      "support_type": "explicit|implicit|unsupported",
      "assertion": "present|negated|uncertain|planned",
      "temporality": "past|current|future|unknown",
      "sentiment_or_outcome": "effective|ineffective|positive|negative|neutral|unknown",
      "confidence": 0.0,
      "judge_verdict": "needs_human_review"
    }
  ]
}
"""


JUDGE_PROMPT = """\
You are judging a proposed structured annotation for a multilingual toothache post.

Use original source text as authoritative. Check only these criteria:
1. Does every evidence_span_original appear exactly in the source text?
2. Is each CSM domain supported by its evidence span?
3. Does normalized_concept_en avoid overinterpretation?
4. Are assertion, temporality, and sentiment_or_outcome supported?
5. Are there unsupported units that should be rejected?

Return one verdict per unit_id:
- accept
- revise
- reject
- needs_human_review

Use reject for pure cost/admin/booking/location/insurance units that are not framed as pain coping, consequence, or barrier.

Return strict JSON only. Do not add markdown, comments, explanations, or trailing text.
{
  "unit_verdicts": [
    {
      "unit_id": "string",
      "judge_verdict": "accept|revise|reject|needs_human_review",
      "reason": "short reason"
    }
  ]
}
"""


__all__ = [
    "CSM_EXTRACTION_PROMPT",
    "JUDGE_PROMPT",
    "R1_CLASSIFICATION_PROMPT",
    "RELEVANCE_PROMPT",
]
