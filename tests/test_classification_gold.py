import pandas as pd

from dental_ai.classification_gold import (
    canonical_post_id,
    classification_gold_summary,
    load_classification_gold_xlsx,
)
from dental_ai.local_models import _classification_fewshot_payload
from dental_ai.schemas import Country, Language, SourcePost


def test_canonical_post_id_normalizes_raw_and_gold_prefixes():
    assert canonical_post_id("CHI_6370caad") == "6370caad"
    assert canonical_post_id("XHS_6370caad") == "6370caad"
    assert canonical_post_id("JPN_2972") == "2972"
    assert canonical_post_id("2972") == "2972"


def test_load_classification_gold_xlsx_and_summary(tmp_path):
    path = tmp_path / "classification.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(
            [
                {
                    "帖子ID": "XHS_a",
                    "原文标题": "牙痛",
                    "原文正文": "我牙疼去看医生",
                    "经历主体": "E1",
                    "内容功能": "C1",
                    "证据摘录": "我牙疼去看医生",
                    "判定理由": "作者本人经历",
                }
            ]
        ).to_excel(writer, sheet_name="中文", index=False)
        pd.DataFrame(
            [
                {
                    "帖子ID": "1",
                    "原文标题": "",
                    "原文正文": "歯が痛い。どうしたらいい?",
                    "英文译文": "My tooth hurts. What should I do?",
                    "经历主体": "E1",
                    "内容功能": "C2",
                    "证据摘录": "歯が痛い",
                    "判定理由": "真实求助",
                }
            ]
        ).to_excel(writer, sheet_name="日文", index=False)
        pd.DataFrame(
            [
                {
                    "帖子ID": "2",
                    "原文标题": "",
                    "原文正文": "치과 예약 안내",
                    "英文译文": "Dental appointment notice",
                    "经历主体": "E3",
                    "内容功能": "C4",
                    "证据摘录": "예약 안내",
                    "判定理由": "机构引流",
                }
            ]
        ).to_excel(writer, sheet_name="韩文", index=False)

    records = load_classification_gold_xlsx(path)
    summary = classification_gold_summary(records)

    assert len(records) == 3
    assert summary["countries"] == {"CHI": 1, "JPN": 1, "KOR": 1}
    assert summary["content_functions"] == {"C1": 1, "C2": 1, "C4": 1}


def test_classification_fewshot_payload_prefers_same_language_examples(tmp_path):
    records = load_classification_gold_xlsx(_make_workbook(tmp_path))
    post = SourcePost(
        post_id="JPN_999",
        country=Country.JPN,
        language=Language.JA,
        original_text="歯が痛い",
    )

    payload = _classification_fewshot_payload(post, records, k=2)

    assert payload
    assert payload[0]["language"] == "ja"


def _make_workbook(tmp_path):
    path = tmp_path / "classification.xlsx"
    with pd.ExcelWriter(path) as writer:
        for sheet, row in {
            "中文": {
                "帖子ID": "XHS_a",
                "原文正文": "牙痛急救办法",
                "经历主体": "E3",
                "内容功能": "C3",
            },
            "日文": {
                "帖子ID": "1",
                "原文正文": "歯が痛くて歯医者に行った",
                "经历主体": "E1",
                "内容功能": "C1",
            },
            "韩文": {
                "帖子ID": "2",
                "原文正文": "치과 예약 안내",
                "经历主体": "E3",
                "内容功能": "C4",
            },
        }.items():
            payload = {
                "原文标题": "",
                "英文译文": "",
                "证据摘录": row["原文正文"],
                "判定理由": "test",
            }
            payload.update(row)
            pd.DataFrame([payload]).to_excel(writer, sheet_name=sheet, index=False)
    return path
