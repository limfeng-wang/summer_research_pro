from dental_ai.local_models import apply_commercial_safeguard
from dental_ai.schemas import ContentFunctionLabel, Country, Language, SourcePost


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
