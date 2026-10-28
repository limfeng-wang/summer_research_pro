from dental_ai.pipeline import HierarchicalAnnotator, PipelineConfig
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


class StubRelevance:
    def __init__(self, label):
        self.label = label

    def classify_relevance(self, post):
        return self.label


class StubR1:
    def __init__(self, experiencer, content_function):
        self.experiencer = experiencer
        self.content_function = content_function

    def classify_r1(self, post):
        return self.experiencer, self.content_function


class StubRetriever:
    def __init__(self):
        self.calls = 0

    def retrieve(self, post, *, k):
        self.calls += 1
        return []


class StubExtractor:
    def __init__(self):
        self.calls = 0

    def extract_csm(self, post, seed_examples):
        self.calls += 1
        return ExtractionResult(
            post_id=post.post_id,
            country=post.country,
            language=post.language,
            original_text=post.original_text,
            text_clean=post.text_clean,
            units=[
                NarrativeUnit(
                    domain=CSMDomain.SYMPTOM_DESCRIPTION,
                    evidence_span_original="牙疼",
                    surface_text_working="牙疼",
                    normalized_concept_en="Toothache",
                    concept_status=ConceptStatus.NEW_CANDIDATE,
                    support_type=SupportType.EXPLICIT,
                    assertion=AssertionStatus.PRESENT,
                    confidence=0.9,
                    judge_verdict=JudgeVerdict.NEEDS_HUMAN_REVIEW,
                )
            ],
        )


class StubJudge:
    def __init__(self):
        self.calls = 0

    def judge(self, post, result):
        self.calls += 1
        units = [unit.model_copy(update={"judge_verdict": JudgeVerdict.ACCEPT}) for unit in result.units]
        return result.model_copy(update={"units": units})


def make_post():
    return SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        original_text="牙疼得睡不着",
        text_clean="牙疼得睡不着",
    )


def test_r0_stops_without_downstream_labels_or_units():
    extractor = StubExtractor()
    retriever = StubRetriever()
    judge = StubJudge()
    annotator = HierarchicalAnnotator(
        relevance_classifier=StubRelevance(RelevanceLabel.R0),
        r1_classifier=StubR1(ExperiencerLabel.E1, ContentFunctionLabel.C1),
        csm_extractor=extractor,
        retriever=retriever,
        judge=judge,
    )

    output = annotator.annotate_post(make_post())

    assert output.result.relevance_label == RelevanceLabel.R0
    assert output.result.experiencer_label is None
    assert output.result.content_function is None
    assert output.result.units == []
    assert output.trace.stages == ["relevance"]
    assert output.trace.validation.ok
    assert extractor.calls == 0
    assert retriever.calls == 0
    assert judge.calls == 0


def test_r1_c3_stops_before_csm_extraction():
    extractor = StubExtractor()
    retriever = StubRetriever()
    annotator = HierarchicalAnnotator(
        relevance_classifier=StubRelevance(RelevanceLabel.R1),
        r1_classifier=StubR1(ExperiencerLabel.E3, ContentFunctionLabel.C3),
        csm_extractor=extractor,
        retriever=retriever,
    )

    output = annotator.annotate_post(make_post())

    assert output.result.relevance_label == RelevanceLabel.R1
    assert output.result.experiencer_label == ExperiencerLabel.E3
    assert output.result.content_function == ContentFunctionLabel.C3
    assert output.result.units == []
    assert output.trace.stages == ["relevance", "r1_classification"]
    assert output.trace.validation.ok
    assert extractor.calls == 0
    assert retriever.calls == 0


def test_primary_candidate_extracts_and_judges():
    extractor = StubExtractor()
    retriever = StubRetriever()
    judge = StubJudge()
    annotator = HierarchicalAnnotator(
        relevance_classifier=StubRelevance(RelevanceLabel.R1),
        r1_classifier=StubR1(ExperiencerLabel.E1, ContentFunctionLabel.C1),
        csm_extractor=extractor,
        retriever=retriever,
        judge=judge,
    )

    output = annotator.annotate_post(make_post())

    assert output.result.is_primary_csm_candidate
    assert len(output.result.units) == 1
    assert output.result.units[0].unit_id == "p1_u001"
    assert output.result.units[0].judge_verdict == JudgeVerdict.ACCEPT
    assert output.trace.stages == ["relevance", "r1_classification", "rag_retrieval", "csm_extraction", "judge"]
    assert output.trace.validation.ok
    assert extractor.calls == 1
    assert retriever.calls == 1
    assert judge.calls == 1


def test_proxy_extraction_can_be_disabled():
    extractor = StubExtractor()
    retriever = StubRetriever()
    annotator = HierarchicalAnnotator(
        relevance_classifier=StubRelevance(RelevanceLabel.R1),
        r1_classifier=StubR1(ExperiencerLabel.E2, ContentFunctionLabel.C2),
        csm_extractor=extractor,
        retriever=retriever,
        config=PipelineConfig(extract_proxy_csm=False),
    )

    output = annotator.annotate_post(make_post())

    assert output.result.is_proxy_csm_candidate
    assert output.result.units == []
    assert output.trace.validation.ok
    assert extractor.calls == 0
    assert retriever.calls == 0
