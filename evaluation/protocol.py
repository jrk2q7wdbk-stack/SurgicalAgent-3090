"""Shared output protocol for API, local base and future fine-tuned models."""

from __future__ import annotations

import json
from typing import Any


PROTOCOL_VERSION = "surgical-agent-output/v2"
STRUCTURED_MARKER = "STRUCTURED_JSON:"
SYSTEM_PROMPT = """你是手术视频工具调用助手。必须自主决定是否调用给定工具，不得编造工具没有返回的事实。
工具调用必须使用提供方的 function/tool_calls 结构，arguments 必须是符合 schema 的 JSON 对象。
获得工具结果后，最终只输出两部分：
1. 用户可见的简洁文字答案；
2. 单独一行 STRUCTURED_JSON: 后跟一个 JSON 对象。
JSON 至少包含 status。正常可回答使用 status=answerable，并按问题提供 entities、triplets 或 events；
需澄清使用 status=clarify；证据不足或超范围使用 status=refuse；工具失败使用 status=report_tool_error，且不得把失败写成没有操作。
triplets 中每项必须保持 instrument、verb、target 的配对。"""


SYSTEM_PROMPT += "\nFor event answers, compact columnar JSON with triplet_columns/triplets and event_columns/event_intervals is allowed; preserve every triplet pairing, interval and truncation flag."


def compact_structured_answer(value: dict) -> dict:
    """Use a lossless columnar event representation for model targets."""
    events = value.get("events")
    if not isinstance(events, list) or not events:
        return value
    triplet_columns = ["id", "instrument", "verb", "target"]
    event_columns = ["triplet_id", "start_time", "end_time", "left_truncated", "right_truncated", "original_start_time", "original_end_time"]
    catalog = {}
    intervals = []
    for event in events:
        if not isinstance(event, dict):
            continue
        triplet_id = event.get("triplet_id", event.get("id"))
        catalog[(triplet_id, event.get("instrument"), event.get("verb"), event.get("target"))] = [triplet_id, event.get("instrument"), event.get("verb"), event.get("target")]
        left, right = bool(event.get("left_truncated")), bool(event.get("right_truncated"))
        intervals.append([triplet_id, event.get("start_time"), event.get("end_time"), left, right, event.get("original_start_time") if left else None, event.get("original_end_time") if right else None])
    compact = {key: item for key, item in value.items() if key != "events"}
    compact.update({"triplet_columns": triplet_columns, "triplets": list(catalog.values()), "event_columns": event_columns, "event_intervals": intervals})
    return compact


def normalize_structured_answer(value: dict) -> dict:
    """Normalize compact model output to the canonical evaluator structure."""
    if not isinstance(value.get("event_intervals"), list):
        return value
    triplet_columns = value.get("triplet_columns") or []
    triplet_catalog = {}
    normalized_triplets = []
    for row in value.get("triplets") or []:
        if not isinstance(row, list) or len(row) != len(triplet_columns):
            continue
        item = dict(zip(triplet_columns, row))
        triplet_catalog[item.get("id")] = item
        normalized_triplets.append({key: item.get(key) for key in ("id", "instrument", "verb", "target")})
    event_columns = value.get("event_columns") or []
    events = []
    for row in value.get("event_intervals") or []:
        if not isinstance(row, list) or len(row) != len(event_columns):
            continue
        event = dict(zip(event_columns, row))
        identity = triplet_catalog.get(event.get("triplet_id"), {})
        event.update({key: identity.get(key) for key in ("instrument", "verb", "target")})
        events.append(event)
    normalized = {key: item for key, item in value.items() if key not in {"triplet_columns", "event_columns", "event_intervals"}}
    normalized["triplets"] = normalized_triplets
    normalized["events"] = events
    return normalized


def parse_final_output(content: str) -> tuple[str, dict | None, str | None]:
    """Return visible text, structured JSON and deterministic parse error."""
    if STRUCTURED_MARKER not in content:
        return content.strip(), None, "structured_marker_missing"
    visible, encoded = content.rsplit(STRUCTURED_MARKER, 1)
    try:
        structured = json.loads(encoded.strip())
    except json.JSONDecodeError as exc:
        return visible.strip(), None, f"structured_json_invalid:{exc.msg}"
    if not isinstance(structured, dict):
        return visible.strip(), None, "structured_json_not_object"
    if not isinstance(structured.get("status"), str):
        return visible.strip(), None, "structured_status_missing"
    return visible.strip(), normalize_structured_answer(structured), None


def behavior_from_output(structured: dict | None, visible_text: str, last_tool_result: dict | None = None) -> str:
    status = (structured or {}).get("status")
    if status in {"answerable", "answer"}:
        return "answer"
    if status in {"clarify", "refuse", "report_tool_error"}:
        return status
    if last_tool_result and not last_tool_result.get("ok", False):
        return "report_tool_error"
    if any(token in visible_text for token in ("请提供", "请明确", "需要您")):
        return "clarify"
    if any(token in visible_text for token in ("无法回答", "不能回答", "拒绝")):
        return "refuse"
    return "answer"


def tool_call_dict(name: str, arguments: Any) -> dict:
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be a JSON object")
    return {"name": name, "arguments": arguments}
