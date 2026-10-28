"""Auditable post-processing for R/E/C classification labels.

This module is intentionally small and explicit: model clients produce raw
labels, this layer applies stable boundary rules, and the pipeline can record
which rules changed a label. The rule predicates still live in local_models for
backward compatibility while we stabilize the research pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dental_ai.schemas import ContentFunctionLabel, ExperiencerLabel, RelevanceLabel, SourcePost


@dataclass(frozen=True)
class RuleApplication:
    """One auditable post-processing change."""

    rule: str
    field: str
    before: str | None
    after: str | None
    reason: str

    def as_dict(self) -> dict[str, str | None]:
        return {
            "rule": self.rule,
            "field": self.field,
            "before": self.before,
            "after": self.after,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ClassificationPostprocessResult:
    """Final labels plus raw labels and applied post-processing rules."""

    relevance_label: RelevanceLabel
    experiencer_label: ExperiencerLabel | None = None
    content_function: ContentFunctionLabel | None = None
    raw_relevance_label: RelevanceLabel | None = None
    raw_experiencer_label: ExperiencerLabel | None = None
    raw_content_function: ContentFunctionLabel | None = None
    rules: list[RuleApplication] = field(default_factory=list)

    @property
    def rule_dicts(self) -> list[dict[str, str | None]]:
        return [rule.as_dict() for rule in self.rules]


def apply_relevance_postprocessing(post: SourcePost, label: RelevanceLabel) -> ClassificationPostprocessResult:
    """Apply relevance-only rules and record any change."""

    from dental_ai import local_models as lm

    final_label = lm.apply_relevance_safeguard(post, label)
    rules = []
    if final_label != label:
        rules.append(
            RuleApplication(
                rule="relevance.toothache_evidence_required",
                field="relevance_label",
                before=label.value,
                after=final_label.value,
                reason="R1 requires explicit toothache, tooth/gingival pain, or pain-related dental-care evidence.",
            )
        )
    return ClassificationPostprocessResult(
        relevance_label=final_label,
        raw_relevance_label=label,
        rules=rules,
    )


def apply_rec_postprocessing(
    post: SourcePost,
    relevance_label: RelevanceLabel,
    experiencer_label: ExperiencerLabel | None,
    content_function: ContentFunctionLabel | None,
) -> ClassificationPostprocessResult:
    """Apply R/E/C post-processing rules in a named, auditable order."""

    if relevance_label != RelevanceLabel.R1:
        return ClassificationPostprocessResult(
            relevance_label=relevance_label,
            raw_relevance_label=relevance_label,
            raw_experiencer_label=experiencer_label,
            raw_content_function=content_function,
        )
    if experiencer_label is None or content_function is None:
        return ClassificationPostprocessResult(
            relevance_label=relevance_label,
            raw_relevance_label=relevance_label,
            raw_experiencer_label=experiencer_label,
            raw_content_function=content_function,
        )

    from dental_ai import local_models as lm

    rules: list[RuleApplication] = []
    experiencer = experiencer_label
    content = content_function

    def apply_e_rule(rule: str, reason: str, fn: Any) -> None:
        nonlocal experiencer
        before = experiencer
        after = fn(post, before)
        if after != before:
            rules.append(
                RuleApplication(
                    rule=rule,
                    field="experiencer_label",
                    before=before.value,
                    after=after.value,
                    reason=reason,
                )
            )
            experiencer = after

    def apply_c_rule(rule: str, reason: str, fn: Any) -> None:
        nonlocal content
        before = content
        after = fn(post, before)
        if after != before:
            rules.append(
                RuleApplication(
                    rule=rule,
                    field="content_function",
                    before=before.value,
                    after=after.value,
                    reason=reason,
                )
            )
            content = after

    def apply_ec_rule(rule: str, reason: str, fn: Any) -> None:
        nonlocal experiencer, content
        before_e = experiencer
        before_c = content
        after_c = fn(post, before_e, before_c)
        if after_c != before_c:
            rules.append(
                RuleApplication(
                    rule=rule,
                    field="content_function",
                    before=before_c.value,
                    after=after_c.value,
                    reason=reason,
                )
            )
            content = after_c

    apply_e_rule(
        "experiencer.specific_case",
        "Explicit author/other-person case evidence overrides generic E3.",
        lm.apply_specific_experiencer_safeguard,
    )
    apply_c_rule(
        "content.commercial_promote",
        "Strong product, clinic, service, or account-promotion evidence promotes C4.",
        lm.apply_commercial_safeguard,
    )
    apply_c_rule(
        "content.weak_commercial_demote",
        "Weak C4 without strong commercial evidence is demoted to narrative or knowledge.",
        lm.apply_weak_commercial_demote_safeguard,
    )
    apply_c_rule(
        "content.help_seeking_boundary",
        "C2 requires genuine advice-seeking, not rhetorical education/advice titles.",
        lm.apply_help_seeking_safeguard,
    )
    apply_ec_rule(
        "content.personal_narrative",
        "Personal treatment/cost/process accounts with a specific experiencer remain C1 unless promotional.",
        lm.apply_personal_narrative_safeguard,
    )
    apply_ec_rule(
        "content.generic_procedure_demote",
        "Procedure-heavy logistics/aftercare without lived burden is C3.",
        lm.apply_generic_procedure_demote_safeguard,
    )
    apply_ec_rule(
        "content.short_lived_pain_narrative",
        "Short but meaningful lived pain episodes are C1 rather than low-information C3/C5.",
        lm.apply_short_lived_pain_narrative_safeguard,
    )
    apply_e_rule(
        "experiencer.generic_knowledge",
        "Generic knowledge/commercial posts without a specific experiencer are E3.",
        lambda current_post, current_e: lm.apply_generic_knowledge_experiencer_safeguard(
            current_post,
            current_e,
            content,
        ),
    )

    return ClassificationPostprocessResult(
        relevance_label=relevance_label,
        experiencer_label=experiencer,
        content_function=content,
        raw_relevance_label=relevance_label,
        raw_experiencer_label=experiencer_label,
        raw_content_function=content_function,
        rules=rules,
    )


__all__ = [
    "ClassificationPostprocessResult",
    "RuleApplication",
    "apply_rec_postprocessing",
    "apply_relevance_postprocessing",
]
