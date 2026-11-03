"""Common evaluation runner and trace writer."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Iterable

from evaluation.metrics import answer_facts_match, fact_sets, parameters_match, summarize, text_structured_consistency


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stratified_subset(rows: list[dict], limit: int | None) -> list[dict]:
    if limit is None or limit >= len(rows):
        return rows
    groups = {}
    for row in rows:
        key = row.get("abnormal_type") or row.get("question_type") or "unknown"
        groups.setdefault(key, []).append(row)
    selected = []
    keys = sorted(groups)
    index = 0
    while len(selected) < limit and keys:
        key = keys[index % len(keys)]
        if groups[key]:
            selected.append(groups[key].pop(0))
        if not groups[key]:
            keys.remove(key)
            index = 0
        else:
            index += 1
    return selected


def evaluate(samples: Iterable[dict], backend, time_tolerance: float = 0.01, answerability_reviews: dict[str, str] | None = None) -> tuple[list[dict], dict, list[dict]]:
    results = []
    manual_review = []
    answerability_reviews = answerability_reviews or {}
    for sample in samples:
        started = time.perf_counter()
        outcome = backend.run(sample)
        complete_ms = outcome.complete_ms if outcome.complete_ms is not None else (time.perf_counter() - started) * 1000
        expected_call = sample.get("correct_tool_call")
        actual_call = outcome.tool_call
        allowed_calls = sample.get("allowed_tool_calls") or ([expected_call] if expected_call else [])
        actual_calls = outcome.tool_calls or ([actual_call] if actual_call else [])
        expected_sequences = sample.get("allowed_tool_call_sequences") or ([[call] for call in allowed_calls] if allowed_calls else [[]])
        tool_ok = expected_call is None and actual_call is None
        param_ok = expected_call is None and actual_call is None
        parameter_errors = []
        if expected_call is not None:
            name_matching = [sequence for sequence in expected_sequences if len(sequence) == len(actual_calls) and all(reference.get("name") == actual.get("name") for reference, actual in zip(sequence, actual_calls))]
            tool_ok = bool(name_matching)
            if tool_ok:
                attempts = []
                for sequence in name_matching:
                    errors = []
                    for index, (reference, actual) in enumerate(zip(sequence, actual_calls)):
                        matched, call_errors = parameters_match(reference.get("arguments") or {}, actual.get("arguments") or {}, time_tolerance)
                        errors.extend(f"call{index}:{item}" for item in call_errors)
                    attempts.append((not errors, errors))
                param_ok = any(item[0] for item in attempts)
                parameter_errors = [] if param_ok else min((item[1] for item in attempts), key=len)
            elif not actual_calls:
                parameter_errors = ["tool_not_called"]
            else:
                parameter_errors = ["wrong_tool_sequence"]
        facts_ok = answer_facts_match(sample.get("structured_answer") or {}, outcome.structured_answer)
        consistent, consistency_error = text_structured_consistency(outcome.final_answer, outcome.structured_answer)
        auto_scored = outcome.structured_answer is not None and consistent is not None
        expected_behavior = sample.get("expected_behavior")
        abnormal_correct = expected_behavior is None or expected_behavior == outcome.behavior
        terminal_error = outcome.error_type in {"timeout", "parse_error", "tool_error", "max_rounds"} or bool(outcome.error_type)
        joint_ok = tool_ok and param_ok
        normal_task = expected_behavior is None
        normal_success = normal_task and not terminal_error and joint_ok and auto_scored and facts_ok and consistent is not False
        # A tool returning an expected failure is task evidence, not a model/runtime
        # failure. report_tool_error succeeds when the model reports it correctly.
        abnormal_success = (not normal_task and not terminal_error and joint_ok and auto_scored and abnormal_correct and consistent is not False)
        task_success = normal_success or abnormal_success
        label_status = answerability_reviews.get(sample.get("sample_id"), "pending")
        backend_config = getattr(backend, "config", {})
        row = {
            "sample_id": sample.get("sample_id"), "split": sample.get("split"), "question_type": sample.get("question_type"),
            "question": sample.get("user_question"), "reference_time": sample.get("time_range"),
            "reference_tool_call": expected_call, "allowed_reference_tool_calls": allowed_calls, "allowed_tool_call_sequences": expected_sequences, "actual_tool_call": actual_call, "actual_tool_calls": actual_calls,
            "tool_result": outcome.tool_result, "reference_structured_answer": sample.get("structured_answer"),
            "actual_structured_answer": outcome.structured_answer, "final_answer": outcome.final_answer,
            "raw_response": outcome.raw_response, "route_source": outcome.route_source,
            "backend_id": backend.backend_id, "model_backend": backend_config.get("model_backend", "mock" if getattr(backend, "is_mock", False) else "unknown"), "evidence_backend": backend_config.get("evidence_backend"), "mock": bool(getattr(backend, "is_mock", False)),
            "tool_selection_correct": tool_ok, "parameter_correct": param_ok, "parameter_errors": parameter_errors,
            "joint_tool_parameter_correct": joint_ok, "complete_answer_auto_scored": auto_scored, "complete_answer_correct": auto_scored and facts_ok and consistent is not False,
            "text_structured_consistent": consistent, "consistency_error": consistency_error,
            "expected_behavior": expected_behavior, "actual_behavior": outcome.behavior, "abnormal_behavior_correct": abnormal_correct,
            "reference_answerable": sample.get("answerable"), "answerability_label_status": label_status,
            "output_parse_error": outcome.output_parse_error, "error_type": outcome.error_type, "normal_task_success": normal_success if normal_task else None, "abnormal_task_success": abnormal_success if not normal_task else None, "task_success": task_success,
            "latency_ms": {"first_visible": outcome.first_visible_ms, "complete": complete_ms},
        }
        if not auto_scored or consistent is False:
            manual_review.append({"sample_id": sample.get("sample_id"), "reason": outcome.output_parse_error or consistency_error or "natural_language_parse_or_structure_missing", "raw_answer": outcome.final_answer})
        results.append(row)
    summary = summarize(results)
    is_mock = bool(getattr(backend, "is_mock", False))
    unavailable_errors = {"api_key_missing", "local_model_not_loaded"}
    externally_blocked = bool(results) and all(
        row.get("error_type") in unavailable_errors
        or str(row.get("error_type") or "").startswith("dependency_missing:")
        or str(row.get("error_type") or "").startswith("local_model_not_loaded:")
        for row in results
    )
    summary.update({
        "backend_id": backend.backend_id,
        "backend_config": getattr(backend, "config", {}),
        "mock": is_mock,
        "run_status": "external_dependency_unavailable" if externally_blocked else "completed",
        "time_tolerance_seconds": time_tolerance,
        "model_performance_eligible": not is_mock and not externally_blocked,
    })
    if externally_blocked:
        from collections import Counter
        summary["performance_metrics_status"] = "not_computed_external_dependency_unavailable"
        summary["blocked_reason_counts"] = dict(Counter(row.get("error_type") for row in results))
        for key in ("tool_selection_accuracy", "parameter_accuracy_given_correct_tool", "joint_tool_parameter_accuracy", "normal_task_success_rate", "abnormal_task_success_rate", "overall_task_success_rate"):
            if key in summary:
                summary[key]["value"] = None
                summary[key]["scoring_status"] = "not_computed"
        summary["answer_scoring"]["provisional_complete_answer_accuracy"] = None
        summary["answer_scoring"]["final_complete_answer_accuracy"] = None
        summary["answer_scoring"]["final_status"] = "not_computed"
        for set_name in ("entity_set", "triplet_set"):
            for averaging in ("micro", "macro"):
                for metric in ("precision", "recall", "f1"):
                    summary[set_name][averaging][metric] = None
    return results, summary, manual_review


def write_evaluation(output_dir: str | Path, results: list[dict], summary: dict, manual_review: list[dict]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_sample.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (output / "manual_review.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in manual_review:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
