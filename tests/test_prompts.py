from dental_ai.prompts import R1_CLASSIFICATION_PROMPT


def test_r1_prompt_contains_core_priority_and_boundary_rules():
    assert "C4 > C2 > C3 > C1 > C5" in R1_CLASSIFICATION_PROMPT
    assert "advertorials disguised as personal experience" in R1_CLASSIFICATION_PROMPT
    assert "commercial target" in R1_CLASSIFICATION_PROMPT
    assert "cost-sharing, hospital registration" in R1_CLASSIFICATION_PROMPT
    assert "cost/process sharing alone is generic procedure information" in R1_CLASSIFICATION_PROMPT
    assert "Procedure-heavy posts are C3" in R1_CLASSIFICATION_PROMPT
    assert "Broad health hashtags alone are NOT C4" in R1_CLASSIFICATION_PROMPT
    assert "一张图看懂" in R1_CLASSIFICATION_PROMPT
    assert "Rhetorical question" in R1_CLASSIFICATION_PROMPT
    assert "An advertorial written in first person can still be E1" in R1_CLASSIFICATION_PROMPT
    assert "Return strict JSON only" in R1_CLASSIFICATION_PROMPT
