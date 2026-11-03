import json
from pathlib import Path

import pytest

from agent.tool_registry import validate_tool_arguments
from data.abnormal_samples import validate_abnormal_record
from data.cholect50 import get_video_split
from evaluation.backends import BackendResult
from evaluation.metrics import answer_facts_match, parameters_match, set_metric, text_structured_consistency
from evaluation.runner import evaluate
from evaluation.protocol import compact_structured_answer, parse_final_output
from training.tool_call_data import build_assistant_labels, compact_tool_result, export_standardized_conversations, to_transformers_messages, validate_conversation
from evaluation.protocol import SYSTEM_PROMPT


def test_timestamp_and_tool_parameter_boundaries():
    assert validate_tool_arguments("query_video_time", {"video_id": "VID01"})["error"]["code"] == "MISSING_ARGUMENT"
    # JSON schema type validation is distinct from semantic time validation.
    assert validate_tool_arguments("query_video_time", {"video_id": "VID01", "time": "10"})["error"]["code"] == "INVALID_ARGUMENT_TYPE"
    assert validate_tool_arguments("query_video_time", {"video_id": "VID01", "time": -1})["error"]["code"] == "TIME_NEGATIVE"
    assert validate_tool_arguments("query_video_range", {"video_id": "VID01", "start_time": 30, "end_time": 10})["error"]["code"] == "INVALID_TIME_RANGE"
    assert parameters_match({"time": 10.0}, {"time": 10.005}, time_tolerance=.01)[0]
    assert not parameters_match({"time": 10.0}, {"time": 10.02}, time_tolerance=.01)[0]


def test_triplet_pairing_is_preserved():
    reference = {"status": "answerable", "triplets": [
        {"instrument": "grasper", "verb": "retract", "target": "liver"},
        {"instrument": "hook", "verb": "dissect", "target": "gallbladder"},
    ]}
    wrong_pairing = {"status": "answerable", "triplets": [
        {"instrument": "grasper", "verb": "dissect", "target": "liver"},
        {"instrument": "hook", "verb": "retract", "target": "gallbladder"},
    ]}
    assert not answer_facts_match(reference, wrong_pairing)


def test_multi_triplet_sets_and_empty_set_policy():
    values = set_metric([({("a", "b", "c"), ("d", "e", "f")}, {("a", "b", "c"), ("d", "e", "f")}), (set(), set())])
    assert values["micro"]["f1"] == 1.0
    assert values["macro"]["f1"] == 1.0


def test_negation_conflicts_with_structured_facts():
    consistent, reason = text_structured_consistency("没有操作，grasper 未出现。", {"status": "answerable", "entities": ["grasper"]})
    assert consistent is False
    assert "negation" in reason


def test_refusal_clarification_and_tool_error_are_distinct():
    base = {"abnormal_type": "missing_video", "expected_behavior": "clarify", "handling_layer": "model", "correct_tool_call": None, "tool_result": None, "source_state": "missing_context", "final_answer": "请提供视频"}
    assert validate_abnormal_record(base) == []
    bad = dict(base, correct_tool_call={"name": "query_video_time", "arguments": {}})
    assert "clarification_must_not_call_tool" in validate_abnormal_record(bad)
    failed = dict(base, abnormal_type="video_read_failure", expected_behavior="report_tool_error", handling_layer="tool", correct_tool_call={"name": "analyze_video", "arguments": {"video_path": "x"}})
    assert "tool_error_sample_missing_error" in validate_abnormal_record(failed)


def test_frozen_video_splits_do_not_leak():
    split = get_video_split("rdv_original")
    assert not (set(split["train"]) & set(split["validation"]))
    assert not (set(split["train"]) & set(split["test"]))
    assert not (set(split["validation"]) & set(split["test"]))


def test_tool_failure_counts_as_task_failure():
    sample = {"sample_id": "x", "split": "validation", "question_type": "triplet", "user_question": "q", "correct_tool_call": {"name": "query_video_time", "arguments": {"video_id": "VID01", "time": 1.0}}, "structured_answer": {"status": "answerable", "triplets": []}, "answerable": True}
    class FailingBackend:
        backend_id = "failure"
        is_mock = True
        def run(self, _):
            return BackendResult(None, None, "", None, "test", error_type="tool_error")
    results, summary, review = evaluate([sample], FailingBackend())
    assert results[0]["task_success"] is False
    assert summary["normal_task_success_rate"]["value"] == 0
    assert review


def test_allowed_alternative_tool_scheme_is_accepted():
    sample = {"sample_id": "alt", "split": "validation", "question_type": "time_range_summary", "user_question": "q", "correct_tool_call": {"name": "query_video_range", "arguments": {"video_id": "VID01", "start_time": 1.0, "end_time": 2.0}}, "allowed_tool_calls": [{"name": "query_video_range", "arguments": {"video_id": "VID01", "start_time": 1.0, "end_time": 2.0}}, {"name": "summarize_video_events", "arguments": {"video_id": "VID01", "start_time": 1.0, "end_time": 2.0}}], "structured_answer": {"status": "answerable", "triplets": []}, "answerable": True}
    class AlternativeBackend:
        backend_id = "alternative"
        is_mock = True
        def run(self, _):
            return BackendResult({"name": "summarize_video_events", "arguments": {"video_id": "VID01", "start_time": 1.0, "end_time": 2.0}}, {}, "无已标注三元组", {"status": "answerable", "triplets": []}, "test")
    results, _, _ = evaluate([sample], AlternativeBackend())
    assert results[0]["joint_tool_parameter_correct"] is True


def test_expected_tool_failure_can_be_successful_abnormal_task():
    sample = {"sample_id": "tool-error", "split": "validation", "question_type": "abnormal_handling", "user_question": "分析损坏视频", "correct_tool_call": {"name": "analyze_video", "arguments": {"video_path": "missing.mp4"}}, "tool_result": {"ok": False, "status": "error", "error": {"code": "VIDEO_READ_FAILED"}}, "structured_answer": {"status": "report_tool_error", "abnormal_type": "video_read_failure"}, "final_answer": "视频读取失败，未获得视觉证据。", "answerable": False, "expected_behavior": "report_tool_error"}
    from evaluation.backends import ReferenceProtocolBackend
    results, summary, _ = evaluate([sample], ReferenceProtocolBackend())
    assert results[0]["abnormal_task_success"] is True
    assert summary["abnormal_task_success_rate"]["value"] == 1.0


def test_unparsed_answer_remains_in_denominator_and_pending_review():
    sample = {"sample_id": "unparsed", "split": "validation", "question_type": "triplet", "user_question": "q", "correct_tool_call": None, "structured_answer": {"status": "answerable", "triplets": []}, "answerable": True}
    class TextOnly:
        backend_id = "text-only"
        is_mock = False
        def run(self, _):
            return BackendResult(None, None, "无法自动解析的自然语言", None, "test")
    _, summary, review = evaluate([sample], TextOnly())
    assert summary["sample_count"] == 1
    assert summary["answer_scoring"]["automatic_coverage"] == 0
    assert summary["answer_scoring"]["final_complete_answer_accuracy"] is None
    assert len(review) == 1


def test_multi_round_sequence_requires_all_calls_in_order():
    first = {"name": "query_video_time", "arguments": {"video_id": "VID01", "time": 1.0}}
    second = {"name": "query_surgical_knowledge", "arguments": {"instrument": "grasper"}}
    sample = {"sample_id": "multi", "split": "validation", "question_type": "triplet", "user_question": "q", "correct_tool_call": first, "allowed_tool_call_sequences": [[first, second]], "structured_answer": {"status": "answerable", "triplets": []}, "answerable": True}
    class MultiBackend:
        backend_id = "multi"
        is_mock = True
        def run(self, _):
            return BackendResult(first, {}, "已完成", {"status": "answerable", "triplets": []}, "test", tool_calls=[first, second])
    results, _, _ = evaluate([sample], MultiBackend())
    assert results[0]["tool_selection_correct"] is True
    assert results[0]["parameter_correct"] is True


class DummyTokenizer:
    def apply_chat_template(self, messages, tools=None, tokenize=True, add_generation_prompt=False):
        # Prefix-stable representation with role and content/tool-call data.
        result = []
        for message in messages:
            encoded = json.dumps(message, sort_keys=True).encode("utf-8")
            result.extend(encoded)
            result.append(10)
        return result


def _messages():
    return [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "query_video_time", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "{\"ok\": true, \"status\": \"completed\"}"},
        {"role": "assistant", "content": "answer"},
    ]


def test_loss_mask_supervises_only_assistant_messages():
    encoded = build_assistant_labels(DummyTokenizer(), _messages(), [], max_length=10000)
    assert encoded["ok"]
    assert encoded["assistant_token_count"] > 0
    assert encoded["assistant_token_count"] < encoded["token_count"]
    # Tool result occurs in a masked span and is context, not a target.
    tool_bytes = list(json.dumps(_messages()[3], sort_keys=True).encode("utf-8")) + [10]
    start = encoded["input_ids"].index(tool_bytes[0], len(encoded["input_ids"]) // 3)
    assert -100 in encoded["labels"]


def test_overlength_is_filtered_without_truncation():
    encoded = build_assistant_labels(DummyTokenizer(), _messages(), [], max_length=5)
    assert not encoded["ok"]
    assert encoded["status"] == "overlength_filtered"
    assert "input_ids" not in encoded


def test_orphan_tool_result_rejected():
    errors = validate_conversation([{"role": "system", "content": "s"}, {"role": "tool", "tool_call_id": "missing", "content": "x"}, {"role": "assistant", "content": "a"}])
    assert "orphan_tool_result" in errors


def test_openai_tool_arguments_convert_to_transformers_object():
    converted = to_transformers_messages(_messages())
    assert converted[2]["tool_calls"][0]["function"]["arguments"] == {}
    assert isinstance(_messages()[2]["tool_calls"][0]["function"]["arguments"], str)


def test_compact_tool_result_preserves_all_triplets_but_drops_raw_samples():
    result = {"ok": True, "status": "human_annotation_available", "observed_triplets": [{"instrument": "a", "verb": "b", "target": "c"}, {"instrument": "d", "verb": "e", "target": "f"}], "events": [{"instrument": "a", "verb": "b", "target": "c", "start_time": 1, "end_time": 2}], "samples": [{"large": "raw"}], "evidence": {"raw_instances": [1, 2]}, "source": {"type": "human_annotation", "dataset": "CholecT50", "annotation_file": "private/path"}}
    compact = compact_tool_result(result)
    assert compact["schema_version"] == "surgical-agent-model-evidence/v2"
    assert len(compact["triplets"]) == 2
    assert len(compact["event_intervals"]) == 1
    assert "samples" not in compact and "evidence" not in compact
    assert "annotation_file" not in compact["source"]


def test_compact_event_keeps_answer_facts_and_only_relevant_original_bounds():
    result = {
        "ok": True,
        "status": "human_annotation_available",
        "observed_triplets": [{"id": 7, "instrument": "grasper", "verb": "grasp", "target": "gallbladder", "confidence": 1.0}],
        "events": [
            {"triplet_id": 7, "instrument": "grasper", "verb": "grasp", "target": "gallbladder", "start_time": 10.0, "end_time": 20.0, "left_truncated": True, "right_truncated": False, "original_start_time": 5.0, "original_end_time": 20.0, "sample_count": 11},
        ],
    }
    compact = compact_tool_result(result, "continuous_event")
    assert compact["triplet_columns"] == ["id", "instrument", "verb", "target", "confidence"]
    assert compact["triplets"] == [[7, "grasper", "grasp", "gallbladder", 1.0]]
    assert compact["event_intervals"] == [[7, 10.0, 20.0, True, False, 5.0, None]]


def test_question_specific_evidence_omits_unneeded_event_details():
    result = {"ok": True, "observed_triplets": [{"id": 7, "instrument": "grasper", "verb": "grasp", "target": "gallbladder", "confidence": 1.0}], "events": [{"triplet_id": 7, "instrument": "grasper", "verb": "grasp", "target": "gallbladder", "start_time": 1.0, "end_time": 2.0}]}
    summary = compact_tool_result(result, "time_range_summary")
    assert summary["triplets"] == [[7, "grasper", "grasp", "gallbladder", 1.0]]
    assert "event_intervals" not in summary
    instrument = compact_tool_result(result, "instrument")
    assert instrument["entity_columns"] == ["instrument", "confidence"]
    assert instrument["entities"] == [["grasper", 1.0]]
    assert "triplets" not in instrument


def test_compact_event_answer_round_trips_for_evaluation():
    canonical = {"status": "answerable", "events": [{"triplet_id": 7, "instrument": "grasper", "verb": "grasp", "target": "gallbladder", "start_time": 1.0, "end_time": 2.0, "left_truncated": True, "right_truncated": False, "original_start_time": 0.0}]}
    compact = compact_structured_answer(canonical)
    visible, normalized, error = parse_final_output("answer\nSTRUCTURED_JSON: " + json.dumps(compact))
    assert visible == "answer" and error is None
    assert normalized["events"][0]["instrument"] == "grasper"
    assert answer_facts_match(canonical, normalized)


def test_standardized_export_matches_runtime_prompt_and_structured_protocol(tmp_path):
    source = tmp_path / "source.jsonl"
    target = tmp_path / "target.jsonl"
    record = {"sample_id": "s1", "split": "train", "question_type": "instrument", "user_question": "q", "correct_tool_call": {"name": "query_video_time", "arguments": {"video_id": "VID01", "time": 1.0}}, "tool_result": {"ok": True, "status": "human_annotation_available", "observations": [{"id": 7, "instrument": "grasper", "verb": "grasp", "target": "gallbladder", "confidence": 1.0}]}, "final_answer": "grasper", "structured_answer": {"status": "answerable", "entities": ["grasper"]}, "answerable": True}
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    report = export_standardized_conversations([source], target)
    row = json.loads(target.read_text(encoding="utf-8"))
    assert report["valid"] and report["runtime_prompt_aligned"]
    assert row["messages"][0]["content"] == SYSTEM_PROMPT
    assert "STRUCTURED_JSON:" in row["messages"][-1]["content"]
    tool = json.loads(row["messages"][3]["content"])
    assert tool["schema_version"] == "surgical-agent-model-evidence/v2"
