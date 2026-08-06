from dental_ai.local_models import (
    _extract_label_payload,
    apply_classification_safeguards,
    apply_commercial_safeguard,
    apply_weak_commercial_demote_safeguard,
    has_strong_commercial_evidence,
)
from dental_ai.schemas import ContentFunctionLabel, Country, ExperiencerLabel, Language, SourcePost


def test_extract_label_payload_recovers_labels_from_truncated_json():
    text = (
        '{\n'
        '  "experiencer_label": "E3",\n'
        '  "content_function": "C3",\n'
        '  "experiencer_evidence": "",\n'
        '  "content_function_evidence": "long evidence that never closes'
    )

    assert _extract_label_payload(text, ["experiencer_label", "content_function"]) == {
        "experiencer_label": "E3",
        "content_function": "C3",
    }


def test_commercial_safeguard_promotes_product_advertorial_to_c4():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean=(
            "喝了一包芬必得精氨酸布洛芬颗粒, 真的就快速缓解了我的牙痛。"
            "不愧是西药大品牌! 全家都很信任它! #芬必得 #牙痛止痛药"
        ),
    )

    assert apply_commercial_safeguard(post, ContentFunctionLabel.C3) == ContentFunctionLabel.C4


def test_commercial_safeguard_preserves_noncommercial_c3():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean="牙痛急救办法: 可以短时间冰敷, 但如果持续疼痛需要及时就医。",
    )

    assert apply_commercial_safeguard(post, ContentFunctionLabel.C3) == ContentFunctionLabel.C3


def test_commercial_safeguard_requires_promotion_not_care_logistics():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean=(
            "作者本人的是阻生齿, 拔牙两次。第一次手术费975, 第二次手术费1300左右。"
            "挂号一定要趁早, 可以手机预约挂号。拔完牙24h内可以冰敷。"
            "价格不同地区会有所差异, 仅供参考。"
        ),
    )

    assert not has_strong_commercial_evidence(post)
    assert apply_commercial_safeguard(post, ContentFunctionLabel.C1) == ContentFunctionLabel.C1
    assert apply_commercial_safeguard(post, ContentFunctionLabel.C3) == ContentFunctionLabel.C3


def test_weak_commercial_demote_safeguard_turns_personal_cost_post_back_to_c1():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean=(
            "拔牙×2, 第一次手术费975, 第二次手术费1300左右。"
            "作者本人的是阻生齿。首先关于挂号的小tips: 可以在手机上提前预约挂号。"
            "拔牙后的注意事项: 拔完牙24h内可以冰敷一下。"
        ),
    )

    assert apply_weak_commercial_demote_safeguard(post, ContentFunctionLabel.C4) == ContentFunctionLabel.C1
    assert apply_classification_safeguards(post, ExperiencerLabel.E1, ContentFunctionLabel.C4) == (
        ExperiencerLabel.E1,
        ContentFunctionLabel.C1,
    )


def test_classification_safeguards_preserve_first_person_advertorial_as_e1_c4():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean="喝了一包芬必得小绿盒, 真的缓解了我的牙痛。不愧是大品牌! #芬必得 #牙痛止痛药",
    )

    assert apply_classification_safeguards(post, ExperiencerLabel.E1, ContentFunctionLabel.C3) == (
        ExperiencerLabel.E1,
        ContentFunctionLabel.C4,
    )


def test_classification_safeguards_preserve_clinic_account_promotion_as_c4():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean="智齿冠周炎症状和治疗一篇说清楚。#上海看牙 #上海颂皓口腔 @上海颂皓口腔",
    )

    assert apply_classification_safeguards(post, ExperiencerLabel.E3, ContentFunctionLabel.C3) == (
        ExperiencerLabel.E3,
        ContentFunctionLabel.C4,
    )


def test_classification_safeguards_demote_general_knowledge_to_e3():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean="什么是智齿冠周炎? 常见病因、症状、护理和治疗方法一篇说清楚。",
    )

    assert apply_classification_safeguards(post, ExperiencerLabel.E1, ContentFunctionLabel.C3) == (
        ExperiencerLabel.E3,
        ContentFunctionLabel.C3,
    )


def test_classification_safeguards_promote_genuine_question_to_c2():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        original_title="牙疼求助",
        text_clean="我牙疼三天了, 晚上睡不着, 要不要去急诊? 有没有人知道怎么办?",
    )

    assert apply_classification_safeguards(post, ExperiencerLabel.E1, ContentFunctionLabel.C1) == (
        ExperiencerLabel.E1,
        ContentFunctionLabel.C2,
    )


def test_classification_safeguards_do_not_promote_rhetorical_education_title_to_c2():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        original_title="牙痛怎么办? 8个急救办法",
        text_clean="牙痛怎么办? 8个急救办法: 冰敷、盐水漱口、及时就医。",
    )

    assert apply_classification_safeguards(post, ExperiencerLabel.E3, ContentFunctionLabel.C3) == (
        ExperiencerLabel.E3,
        ContentFunctionLabel.C3,
    )
