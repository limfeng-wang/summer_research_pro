from dental_ai.rag import GoldRAGRetriever, retrieval_trace_rows
from dental_ai.schemas import (
    AssertionStatus,
    CSMDomain,
    ConceptStatus,
    ContentFunctionLabel,
    Country,
    ExperiencerLabel,
    ExtractionResult,
    JudgeVerdict,
    Language,
    NarrativeUnit,
    RelevanceLabel,
    SourcePost,
    SupportType,
)


def make_gold(post_id, text, *, country=Country.CHI, language=Language.ZH, experiencer=ExperiencerLabel.E1):
    return ExtractionResult(
        post_id=post_id,
        country=country,
        language=language,
        original_text=text,
        relevance_label=RelevanceLabel.R1,
        experiencer_label=experiencer,
        content_function=ContentFunctionLabel.C1,
        units=[
            NarrativeUnit(
                domain=CSMDomain.SYMPTOM_DESCRIPTION,
                evidence_span_original=text[:2],
                surface_text_working=text[:2],
                normalized_concept_en="Tooth pain",
                concept_status=ConceptStatus.NEW_CANDIDATE,
                support_type=SupportType.EXPLICIT,
                assertion=AssertionStatus.PRESENT,
                confidence=1.0,
                judge_verdict=JudgeVerdict.ACCEPT,
            )
        ],
    )


def test_gold_rag_retriever_lexical_fallback_returns_similar_gold():
    retriever = GoldRAGRetriever(
        [
            make_gold("g1", "牙疼 半夜 睡不着 布洛芬"),
            make_gold("g2", "智齿 拔牙 术后 肿痛"),
            make_gold("g3", "엄마 치통", country=Country.KOR, language=Language.KO, experiencer=ExperiencerLabel.E2),
        ],
        use_embeddings=False,
    )
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        original_text="半夜牙疼睡不着",
    )

    retrieved = retriever.retrieve_with_scores(post, k=2)

    assert [item.result.post_id for item in retrieved][0] == "g1"
    assert retrieved[0].rank == 1
    assert retrieval_trace_rows(post, retrieved)[0]["gold_post_id"] == "g1"


def test_gold_rag_retriever_skips_reranker_by_default(tmp_path):
    retriever = GoldRAGRetriever(
        [
            make_gold("g1", "牙疼 半夜 睡不着 布洛芬"),
            make_gold("g2", "智齿 拔牙 术后 肿痛", experiencer=ExperiencerLabel.E2),
        ],
        reranker_model_path=tmp_path,
        use_embeddings=False,
    )

    assert retriever.use_reranker is False
