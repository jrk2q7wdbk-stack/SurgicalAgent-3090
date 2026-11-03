from agent.confidence import (
    build_grounded_visual_answer,
    classify_confidence,
    enrich_triplet,
)
from tools.qa_generator import generate_qa_pairs_from_prediction


def make_triplet(confidence):
    return {
        "id": 58,
        "instrument": "hook",
        "verb": "dissect",
        "target": "cystic_duct",
        "confidence": confidence,
    }


def test_three_confidence_levels_have_stable_boundaries():
    assert classify_confidence(0.30) == "low"
    assert classify_confidence(0.4999) == "low"
    assert classify_confidence(0.50) == "medium"
    assert classify_confidence(0.7499) == "medium"
    assert classify_confidence(0.75) == "high"
    assert classify_confidence(0.95) == "high"


def test_low_and_high_answers_do_not_use_the_same_tone():
    low = build_grounded_visual_answer("当前三元组是什么？", make_triplet(0.30))
    high = build_grounded_visual_answer("当前三元组是什么？", make_triplet(0.95))

    assert "不能作出确定判断" in low
    assert "仅供复核" in low
    assert "高置信度识别到" in high
    assert "hook - dissect - cystic_duct" in low
    assert "hook - dissect - cystic_duct" in high


def test_medium_answer_is_conservative():
    answer = build_grounded_visual_answer(
        "当前医生正在做什么？", make_triplet(0.60)
    )
    assert "可能" in answer
    assert "人工复核" in answer


def test_enriched_triplet_exposes_machine_readable_policy():
    result = enrich_triplet(make_triplet(0.30))
    assert result["status"] == "low_confidence"
    assert result["answer_policy"] == "refuse_definite_judgment"


def test_qa_generation_is_grounded_and_low_confidence_is_unanswerable():
    pairs = generate_qa_pairs_from_prediction(
        {"best_triplet": make_triplet(0.30)},
        source_id="frame_001",
        timestamp=2.0,
    )
    assert len(pairs) == 4
    assert all(item["answerable"] is False for item in pairs)
    assert all("hook" in item["answer"] for item in pairs)
    assert all(item["evidence"]["confidence_level"] == "low" for item in pairs)
