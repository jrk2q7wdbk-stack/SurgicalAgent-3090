"""Prepare tool-call conversations and assistant-only loss labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
from copy import deepcopy

from agent.tool_registry import model_tool_schemas
from agent.evidence_view import model_evidence_view
from evaluation.protocol import SYSTEM_PROMPT as RUNTIME_SYSTEM_PROMPT, compact_structured_answer


def qwen_tools() -> list[dict]:
    return [{"type": "function", "function": {key: value for key, value in schema.items() if key != "type"}} for schema in model_tool_schemas()]


_TRIPLET_KEYS = ("id", "triplet_id", "instrument", "verb", "target")
_EVENT_KEYS = (
    "triplet_id", "instrument", "verb", "target", "start_time", "end_time",
    "left_truncated", "right_truncated", "original_start_time", "original_end_time",
)


def _compact_triplets(value: object) -> object:
    if not isinstance(value, list):
        return value
    return [
        {key: item[key] for key in _TRIPLET_KEYS if key in item}
        if isinstance(item, dict) else item
        for item in value
    ]


def _compact_events(value: object) -> object:
    if not isinstance(value, list):
        return value
    compacted = []
    for item in value:
        if not isinstance(item, dict):
            compacted.append(item)
            continue
        event = {key: item[key] for key in _EVENT_KEYS if key in item}
        # Original bounds only add evidence when clipping actually occurred.
        if not event.get("left_truncated"):
            event.pop("original_start_time", None)
        if not event.get("right_truncated"):
            event.pop("original_end_time", None)
        compacted.append(event)
    return compacted


def compact_tool_result(result: dict, question_type: str | None = None) -> dict:
    """Keep model-relevant evidence while dropping duplicated raw diagnostics.

    The standardized record remains untouched and is the traceability source.
    """
    return model_evidence_view(result, question_type)


def validate_conversation(messages: list[dict]) -> list[str]:
    errors = []
    if not messages or messages[0].get("role") != "system":
        errors.append("system_message_missing")
    if not any(message.get("role") == "assistant" for message in messages):
        errors.append("assistant_target_missing")
    pending_calls = set()
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                call_id = call.get("id")
                if call_id:
                    pending_calls.add(call_id)
        elif message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending_calls:
                errors.append("orphan_tool_result")
            else:
                pending_calls.remove(call_id)
    if pending_calls:
        errors.append("tool_call_without_result")
    return errors


def to_transformers_messages(messages: list[dict], question_type: str | None = None) -> list[dict]:
    """Convert OpenAI JSON-string arguments to Transformers tool-call objects."""
    converted = deepcopy(messages)
    for message in converted:
        if message.get("role") == "tool":
            content = message.get("content")
            if isinstance(content, str):
                parsed_result = json.loads(content)
                message["content"] = json.dumps(compact_tool_result(parsed_result, question_type), ensure_ascii=False, sort_keys=True)
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                parsed = json.loads(arguments)
                if not isinstance(parsed, dict):
                    raise ValueError("tool call arguments must decode to an object")
                function["arguments"] = parsed
    return converted


def build_assistant_labels(tokenizer, messages: list[dict], tools: list[dict] | None = None, max_length: int = 1024) -> dict:
    """Render full turns and supervise deltas introduced by assistant turns only.

    Whole samples over max_length are rejected. No partial truncation is used,
    preventing a call/result pair from being separated.
    """
    errors = validate_conversation(messages)
    if errors:
        return {"ok": False, "status": "invalid_conversation", "errors": errors}
    rendered: list[int] = []
    labels: list[int] = []
    role_token_counts: dict[str, int] = {}
    role_supervised_counts: dict[str, int] = {}
    for index, message in enumerate(messages):
        prefix = tokenizer.apply_chat_template(messages[:index + 1], tools=tools, tokenize=True, add_generation_prompt=False)
        prefix_ids = list(prefix["input_ids"] if isinstance(prefix, dict) else prefix)
        if prefix_ids[:len(rendered)] != rendered:
            return {"ok": False, "status": "unstable_chat_template_prefix", "message_index": index}
        delta = prefix_ids[len(rendered):]
        role = str(message.get("role"))
        role_token_counts[role] = role_token_counts.get(role, 0) + len(delta)
        supervised = len(delta) if role == "assistant" else 0
        role_supervised_counts[role] = role_supervised_counts.get(role, 0) + supervised
        rendered.extend(delta)
        labels.extend(delta if message.get("role") == "assistant" else [-100] * len(delta))
    if len(rendered) > max_length:
        return {"ok": False, "status": "overlength_filtered", "token_count": len(rendered), "max_length": max_length}
    if not any(value != -100 for value in labels):
        return {"ok": False, "status": "no_assistant_tokens"}
    return {"ok": True, "input_ids": rendered, "attention_mask": [1] * len(rendered), "labels": labels, "token_count": len(rendered), "assistant_token_count": sum(value != -100 for value in labels), "role_token_counts": role_token_counts, "role_supervised_counts": role_supervised_counts}


def export_model_conversations(inputs: Iterable[str | Path], output: str | Path) -> dict:
    rows = []
    errors = []
    for input_path in inputs:
        path = Path(input_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                row = json.loads(raw)
                messages = row.get("messages") or []
                validation = validate_conversation(messages)
                if validation:
                    errors.append({"file": str(path), "line": line_number, "errors": validation})
                    continue
                try:
                    metadata = row.get("metadata") or {}
                    question_type = metadata.get("question_type") or metadata.get("abnormal_type")
                    converted = to_transformers_messages(messages, question_type)
                except (json.JSONDecodeError, ValueError) as exc:
                    errors.append({"file": str(path), "line": line_number, "errors": [f"tool_arguments_conversion_failed:{exc}"]})
                    continue
                rows.append({"messages": converted, "tools": qwen_tools(), "metadata": row.get("metadata") or {}})
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"valid": not errors, "exported": len(rows), "rejected": len(errors), "errors": errors[:100], "output": str(target)}


def export_standardized_conversations(inputs: Iterable[str | Path], output: str | Path) -> dict:
    """Build runtime-aligned training conversations from standardized truth."""
    rows, errors = [], []
    for input_path in inputs:
        path = Path(input_path)
        with path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                record = json.loads(raw)
                sample_id = record.get("sample_id")
                question_type = record.get("question_type") or record.get("abnormal_type")
                structured = record.get("structured_answer")
                if not sample_id or not isinstance(structured, dict):
                    errors.append({"file": str(path), "line": line_number, "errors": ["missing_sample_id_or_structured_answer"]})
                    continue
                messages = [
                    {"role": "system", "content": RUNTIME_SYSTEM_PROMPT},
                    {"role": "user", "content": record.get("user_question", "")},
                ]
                call = record.get("correct_tool_call")
                if call:
                    call_id = f"call_{sample_id}"
                    messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": call_id, "type": "function", "function": {"name": call["name"], "arguments": deepcopy(call.get("arguments") or {})}}]})
                    messages.append({"role": "tool", "tool_call_id": call_id, "name": call["name"], "content": json.dumps(model_evidence_view(record.get("tool_result") or {}, question_type), ensure_ascii=False, sort_keys=True, separators=(",", ":"))})
                final_text = (record.get("final_answer") or "").rstrip()
                final_text += "\nSTRUCTURED_JSON: " + json.dumps(compact_structured_answer(structured), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                messages.append({"role": "assistant", "content": final_text})
                validation = validate_conversation(messages)
                if validation:
                    errors.append({"file": str(path), "line": line_number, "errors": validation})
                    continue
                metadata = {
                    "sample_id": sample_id, "split": record.get("split"), "question_type": question_type,
                    "video_id": record.get("video_id"), "time_range": record.get("time_range"),
                    "answerable": record.get("answerable"), "evidence_source": record.get("evidence_source"),
                }
                if record.get("abnormal_type"):
                    metadata["abnormal_type"] = record["abnormal_type"]
                rows.append({"messages": messages, "tools": qwen_tools(), "metadata": metadata})
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"valid": not errors, "exported": len(rows), "rejected": len(errors), "errors": errors[:100], "output": str(target), "source_format": "standardized_truth", "runtime_prompt_aligned": True, "structured_output_supervised": True}
