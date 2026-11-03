"""Deterministic metrics for tool calls, structured facts and abnormal behavior."""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any, Iterable


DEFAULT_SYNONYMS = {
    "grasp": "grasp", "grasping": "grasp", "retracting": "retract",
    "gall bladder": "gallbladder", "cystic-duct": "cystic duct",
}


def normalize_text(value: Any, synonyms: dict[str, str] | None = None) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    for source, target in (synonyms or DEFAULT_SYNONYMS).items():
        text = text.replace(source, target)
    return text


def normalize_triplet(item: dict) -> tuple[str, str, str]:
    return tuple(normalize_text(item.get(key)) for key in ("instrument", "verb", "target"))


def fact_sets(structured: dict | None) -> tuple[set[str], set[tuple[str, str, str]]]:
    value = structured or {}
    entities = {normalize_text(item) for item in value.get("entities", []) if item is not None}
    triplets = {normalize_triplet(item) for item in value.get("triplets", []) if isinstance(item, dict)}
    triplets.update(normalize_triplet(item) for item in value.get("events", []) if isinstance(item, dict))
    if triplets:
        entities.update(part for triplet in triplets for part in triplet)
    return entities, triplets


def _prf(reference: set, predicted: set) -> tuple[int, int, int, float, float, float]:
    tp = len(reference & predicted)
    fp = len(predicted - reference)
    fn = len(reference - predicted)
    # Empty/empty is a correct set prediction. Empty/nonempty and reverse score 0.
    if not reference and not predicted:
        precision = recall = f1 = 1.0
    else:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, fn, precision, recall, f1


def set_metric(rows: Iterable[tuple[set, set]]) -> dict:
    values = [_prf(reference, predicted) for reference, predicted in rows]
    if not values:
        return {"count": 0, "micro": {"precision": None, "recall": None, "f1": None}, "macro": {"precision": None, "recall": None, "f1": None}}
    tp, fp, fn = (sum(row[index] for row in values) for index in range(3))
    micro_p = tp / (tp + fp) if tp + fp else 1.0
    micro_r = tp / (tp + fn) if tp + fn else 1.0
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    return {
        "count": len(values),
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f},
        "macro": {"precision": sum(x[3] for x in values) / len(values), "recall": sum(x[4] for x in values) / len(values), "f1": sum(x[5] for x in values) / len(values)},
        "empty_set_rule": "empty reference and empty prediction score 1; one-sided empty scores 0",
    }


def parameters_match(reference: dict, predicted: dict, time_tolerance: float = 0.01) -> tuple[bool, list[str]]:
    errors = []
    for key, expected in reference.items():
        if key not in predicted:
            errors.append(f"missing:{key}")
            continue
        actual = predicted[key]
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            if not isinstance(actual, (int, float)) or abs(float(expected) - float(actual)) > time_tolerance:
                errors.append(f"wrong:{key}")
        elif normalize_text(expected) != normalize_text(actual):
            errors.append(f"wrong:{key}")
    return not errors, errors


def answer_facts_match(reference: dict, predicted: dict | None) -> bool:
    if predicted is None:
        return False
    reference_status = reference.get("status")
    if reference_status and predicted.get("status") != reference_status:
        return False
    ref_entities, ref_triplets = fact_sets(reference)
    pred_entities, pred_triplets = fact_sets(predicted)
    if "entities" in reference and ref_entities != pred_entities:
        return False
    if "triplets" in reference and ref_triplets != pred_triplets:
        return False
    if "events" in reference:
        def events(value):
            return {
                (normalize_triplet(item), item.get("start_time"), item.get("end_time"), bool(item.get("left_truncated")), bool(item.get("right_truncated")))
                for item in value.get("events", []) if isinstance(item, dict)
            }
        if events(reference) != events(predicted):
            return False
    if reference.get("abnormal_type") != predicted.get("abnormal_type") and "abnormal_type" in reference:
        return False
    return True


def text_structured_consistency(text: str, structured: dict | None) -> tuple[bool | None, str | None]:
    if structured is None:
        return None, "structured_answer_missing"
    normalized = normalize_text(text)
    entities, triplets = fact_sets(structured)
    facts = entities | {part for triplet in triplets for part in triplet}
    missing = sorted(fact for fact in facts if fact and fact not in normalized)
    if missing:
        return False, "structured_facts_missing_from_text:" + ",".join(missing)
    status = structured.get("status")
    negation = any(token in normalized for token in ("没有", "不存在", "no operation", "none"))
    if status == "answerable" and facts and negation:
        return False, "negation_conflicts_with_answerable_facts"
    return True, None


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile_value
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


def summarize(results: list[dict]) -> dict:
    required = [row for row in results if row.get("reference_tool_call") is not None]
    tool_correct = [row for row in required if row.get("tool_selection_correct")]
    param_correct = [row for row in tool_correct if row.get("parameter_correct")]
    entity_rows, triplet_rows = [], []
    parameter_errors = Counter()
    for row in results:
        if row.get("complete_answer_auto_scored"):
            ref_entities, ref_triplets = fact_sets(row.get("reference_structured_answer"))
            pred_entities, pred_triplets = fact_sets(row.get("actual_structured_answer"))
            entity_rows.append((ref_entities, pred_entities))
            triplet_rows.append((ref_triplets, pred_triplets))
        parameter_errors.update(row.get("parameter_errors") or [])
    abnormal = [row for row in results if row.get("expected_behavior")]
    by_behavior = {}
    for behavior in ("refuse", "clarify", "report_tool_error"):
        group = [row for row in abnormal if row["expected_behavior"] == behavior]
        by_behavior[behavior] = {"numerator": sum(bool(row.get("abnormal_behavior_correct")) for row in group), "denominator": len(group), "accuracy": (sum(bool(row.get("abnormal_behavior_correct")) for row in group) / len(group) if group else None)}
    audited_answerability = [row for row in results if row.get("answerability_label_status") == "human_confirmed"]
    answerable = [row for row in audited_answerability if row.get("reference_answerable")]
    unanswerable = [row for row in audited_answerability if not row.get("reference_answerable")]
    refused_answerable = sum(row.get("actual_behavior") == "refuse" for row in answerable)
    refused_unanswerable = sum(row.get("actual_behavior") == "refuse" for row in unanswerable)
    completed = [float(row["latency_ms"]["complete"]) for row in results if row.get("latency_ms", {}).get("complete") is not None]
    ttft = [float(row["latency_ms"]["first_visible"]) for row in results if row.get("latency_ms", {}).get("first_visible") is not None]
    normal = [row for row in results if not row.get("expected_behavior")]
    abnormal_tasks = [row for row in results if row.get("expected_behavior")]
    auto_scored = [row for row in results if row.get("complete_answer_auto_scored")]
    pending_review = [row for row in results if not row.get("complete_answer_auto_scored") or row.get("text_structured_consistent") is False]
    return {
        "sample_count": len(results),
        "tool_selection_accuracy": {"numerator": len(tool_correct), "denominator": len(required), "value": len(tool_correct) / len(required) if required else None},
        "parameter_accuracy_given_correct_tool": {"numerator": len(param_correct), "denominator": len(tool_correct), "value": len(param_correct) / len(tool_correct) if tool_correct else None, "error_distribution": dict(parameter_errors)},
        "joint_tool_parameter_accuracy": {"numerator": sum(bool(row.get("joint_tool_parameter_correct")) for row in required), "denominator": len(required), "value": sum(bool(row.get("joint_tool_parameter_correct")) for row in required) / len(required) if required else None},
        "answer_scoring": {
            "automatic_coverage": len(auto_scored) / len(results) if results else None,
            "automatically_scored": len(auto_scored),
            "total": len(results),
            "pending_manual_review": len(pending_review),
            "provisional_complete_answer_accuracy": sum(bool(row.get("complete_answer_correct")) for row in auto_scored) / len(auto_scored) if auto_scored else None,
            "provisional_numerator": sum(bool(row.get("complete_answer_correct")) for row in auto_scored),
            "provisional_denominator": len(auto_scored),
            "final_complete_answer_accuracy": None if pending_review else (sum(bool(row.get("complete_answer_correct")) for row in results) / len(results) if results else None),
            "final_status": "pending_manual_review" if pending_review else "complete",
        },
        "entity_set": set_metric(entity_rows),
        "triplet_set": set_metric(triplet_rows),
        "abnormal_handling": by_behavior,
        "refusal": {
            "human_confirmed_denominator": len(audited_answerability),
            "pending_labels_excluded": len(results) - len(audited_answerability),
            "correct_refusal_rate": refused_unanswerable / len(unanswerable) if unanswerable else None,
            "false_refusal_rate": refused_answerable / len(answerable) if answerable else None,
            "answer_coverage": sum(row.get("actual_behavior") == "answer" for row in audited_answerability) / len(audited_answerability) if audited_answerability else None,
        },
        "normal_task_success_rate": {"numerator": sum(bool(row.get("normal_task_success")) for row in normal), "denominator": len(normal), "value": sum(bool(row.get("normal_task_success")) for row in normal) / len(normal) if normal else None},
        "abnormal_task_success_rate": {"numerator": sum(bool(row.get("abnormal_task_success")) for row in abnormal_tasks), "denominator": len(abnormal_tasks), "value": sum(bool(row.get("abnormal_task_success")) for row in abnormal_tasks) / len(abnormal_tasks) if abnormal_tasks else None, "expected_tool_failure_can_succeed": True},
        "overall_task_success_rate": {"numerator": sum(bool(row.get("task_success")) for row in results), "denominator": len(results), "value": sum(bool(row.get("task_success")) for row in results) / len(results) if results else None, "rule": "normal and abnormal tasks use their predefined success conditions"},
        "latency_ms": {"complete": {"count": len(completed), "median": percentile(completed, .5), "p95": percentile(completed, .95)}, "first_visible": {"count": len(ttft), "median": percentile(ttft, .5), "p95": percentile(ttft, .95), "note": "Non-streaming backends report unavailable."}},
    }
