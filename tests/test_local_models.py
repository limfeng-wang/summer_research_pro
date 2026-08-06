from dental_ai.local_models import apply_classification_safeguards, apply_commercial_safeguard
from dental_ai.schemas import ContentFunctionLabel, Country, ExperiencerLabel, Language, SourcePost


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
