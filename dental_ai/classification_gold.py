"""Load multilingual R/E/C classification gold data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dental_ai.schemas import ContentFunctionLabel, Country, ExperiencerLabel, Language, SourcePost


SHEET_META = {
    "中文": (Country.CHI, Language.ZH),
    "日文": (Country.JPN, Language.JA),
    "韩文": (Country.KOR, Language.KO),
}


@dataclass(frozen=True)
class ClassificationGoldRecord:
    """One human-adjudicated classification example."""

    post_id: str
    canonical_id: str
    country: Country
    language: Language
    original_title: str
    original_text: str
    analysis_text_en: str
    experiencer_label: ExperiencerLabel
    content_function: ContentFunctionLabel
    evidence_excerpt: str
    rationale: str
    source_sheet: str
    text_fingerprint: str

    @property
    def combined_source_text(self) -> str:
        return "\n".join(part for part in [self.original_title, self.original_text] if part).strip()

    def as_source_post(self) -> SourcePost:
        return SourcePost(
            post_id=self.post_id,
            country=self.country,
            language=self.language,
            original_title=self.original_title,
            original_text=self.original_text,
            text_fingerprint=self.text_fingerprint,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "post_id": self.post_id,
            "canonical_id": self.canonical_id,
            "country": self.country.value,
            "language": self.language.value,
            "original_title": self.original_title,
            "original_text": self.original_text,
            "analysis_text_en": self.analysis_text_en,
            "experiencer_label": self.experiencer_label.value,
            "content_function": self.content_function.value,
            "evidence_excerpt": self.evidence_excerpt,
            "rationale": self.rationale,
            "source_sheet": self.source_sheet,
            "text_fingerprint": self.text_fingerprint,
        }


def canonical_post_id(post_id: object) -> str:
    """Normalize equivalent IDs across raw and gold files.

    Examples:
    - CHI_6370... and XHS_6370... refer to the same Xiaohongshu note.
    - JPN_2972 and 2972 refer to the same raw Japanese row.
    """

    value = str(post_id).strip()
    if "_" in value:
        prefix, suffix = value.split("_", 1)
        if prefix in {"CHI", "JPN", "KOR", "XHS"}:
            return suffix
    return value


def source_text_fingerprint(title: str = "", text: str = "") -> str:
    """Hash normalized title/body text for split-overlap checks."""

    combined = "\n".join(part.strip() for part in [title, text] if part and part.strip()).strip()
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def load_classification_gold_xlsx(path: str | Path) -> list[ClassificationGoldRecord]:
    """Load the workbook containing the multilingual classification codebook/examples."""

    import pandas as pd

    path = Path(path)
    records: list[ClassificationGoldRecord] = []
    seen: set[tuple[Country, str]] = set()

    for sheet, (country, language) in SHEET_META.items():
        frame = pd.read_excel(path, sheet_name=sheet)
        for index, row in frame.iterrows():
            try:
                post_id = _clean_cell(row["帖子ID"])
                original_title = _clean_cell(row.get("原文标题", ""))
                original_text = _clean_cell(row.get("原文正文", ""))
                if not original_title and not original_text:
                    raise ValueError("missing source text")
                canonical_id = canonical_post_id(post_id)
                key = (country, canonical_id)
                if key in seen:
                    raise ValueError(f"duplicate canonical ID within classification gold: {key!r}")
                seen.add(key)
                records.append(
                    ClassificationGoldRecord(
                        post_id=post_id,
                        canonical_id=canonical_id,
                        country=country,
                        language=language,
                        original_title=original_title,
                        original_text=original_text,
                        analysis_text_en=_clean_cell(row.get("英文译文", "")),
                        experiencer_label=ExperiencerLabel(_clean_cell(row["经历主体"])),
                        content_function=ContentFunctionLabel(_clean_cell(row["内容功能"])),
                        evidence_excerpt=_clean_cell(row.get("证据摘录", "")),
                        rationale=_clean_cell(row.get("判定理由", "")),
                        source_sheet=sheet,
                        text_fingerprint=source_text_fingerprint(original_title, original_text),
                    )
                )
            except Exception as exc:
                raise ValueError(f"Invalid classification gold row {sheet}:{index + 2}: {exc}") from exc

    return records


def load_classification_gold_jsonl(path: str | Path) -> list[ClassificationGoldRecord]:
    """Load normalized classification gold JSONL exported by the split builder."""

    records: list[ClassificationGoldRecord] = []
    with Path(path).open(encoding="utf-8") as file:
        for index, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                records.append(
                    ClassificationGoldRecord(
                        post_id=str(item["post_id"]),
                        canonical_id=str(item["canonical_id"]),
                        country=Country(item["country"]),
                        language=Language(item["language"]),
                        original_title=str(item.get("original_title", "")),
                        original_text=str(item.get("original_text", "")),
                        analysis_text_en=str(item.get("analysis_text_en", "")),
                        experiencer_label=ExperiencerLabel(item["experiencer_label"]),
                        content_function=ContentFunctionLabel(item["content_function"]),
                        evidence_excerpt=str(item.get("evidence_excerpt", "")),
                        rationale=str(item.get("rationale", "")),
                        source_sheet=str(item.get("source_sheet", "")),
                        text_fingerprint=str(item.get("text_fingerprint", "")),
                    )
                )
            except Exception as exc:
                raise ValueError(f"Invalid classification gold JSONL row {path}:{index}: {exc}") from exc
    return records


def classification_gold_summary(records: Iterable[ClassificationGoldRecord]) -> dict[str, object]:
    """Return basic counts for loaded classification gold."""

    from collections import Counter

    rows = list(records)
    return {
        "posts": len(rows),
        "countries": dict(Counter(row.country.value for row in rows)),
        "languages": dict(Counter(row.language.value for row in rows)),
        "experiencer_labels": dict(Counter(row.experiencer_label.value for row in rows)),
        "content_functions": dict(Counter(row.content_function.value for row in rows)),
        "sheets": dict(Counter(row.source_sheet for row in rows)),
    }


def _clean_cell(value: object) -> str:
    if value is None:
        return ""
    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


__all__ = [
    "ClassificationGoldRecord",
    "canonical_post_id",
    "classification_gold_summary",
    "load_classification_gold_jsonl",
    "load_classification_gold_xlsx",
    "source_text_fingerprint",
]
