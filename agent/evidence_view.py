"""Compact, answer-preserving evidence passed from tools to language models.

Registered tools continue to return their complete traceable result. This view
is the shared model boundary used by runtime backends and training exports.
"""

from __future__ import annotations

from typing import Any

MODEL_EVIDENCE_SCHEMA = "surgical-agent-model-evidence/v2"

_SCALAR_KEYS = (
    "ok", "status", "video_id", "requested_time", "actual_frame_time",
    "frame_id", "start_time", "end_time", "warning", "error",
    "tool_api_version", "evaluation_backend",
)
_SOURCE_KEYS = (
    "type", "dataset", "dataset_version", "is_ground_truth", "model_name",
    "model_version", "checkpoint_hash", "recognition_config_hash",
)


def _source(result: dict) -> Any:
    value = result.get("source")
    if isinstance(value, dict):
        return {key: value[key] for key in _SOURCE_KEYS if key in value}
    return value


def _all_triplets(result: dict) -> list[dict]:
    values = result.get("observations") or result.get("observed_triplets") or result.get("predicted_triplets") or []
    unique = {}
    for item in values:
        if not isinstance(item, dict):
            continue
        triplet_id = item.get("id", item.get("triplet_id"))
        key = (triplet_id, item.get("instrument"), item.get("verb"), item.get("target"))
        confidence = item.get("confidence", item.get("score"))
        current = unique.get(key)
        if current is None or (confidence is not None and (current.get("confidence") is None or confidence > current["confidence"])):
            unique[key] = {"id": triplet_id, "instrument": item.get("instrument"), "verb": item.get("verb"), "target": item.get("target"), "confidence": confidence}
    for event in result.get("events") or []:
        if not isinstance(event, dict):
            continue
        triplet_id = event.get("triplet_id", event.get("id"))
        key = (triplet_id, event.get("instrument"), event.get("verb"), event.get("target"))
        if key not in unique:
            unique[key] = {"id": triplet_id, "instrument": event.get("instrument"), "verb": event.get("verb"), "target": event.get("target"), "confidence": event.get("confidence")}
    return list(unique.values())


def _entity_view(triplets: list[dict], field: str) -> dict:
    values = {}
    for item in triplets:
        value = item.get(field)
        if value is None:
            continue
        confidence = item.get("confidence")
        if value not in values or (confidence is not None and (values[value] is None or confidence > values[value])):
            values[value] = confidence
    include_confidence = any(value is not None for value in values.values())
    columns = [field] + (["confidence"] if include_confidence else [])
    rows = [[value] + ([values[value]] if include_confidence else []) for value in sorted(values)]
    return {"entity_columns": columns, "entities": rows}


def _triplet_view(triplets: list[dict]) -> dict:
    include_confidence = any(item.get("confidence") is not None for item in triplets)
    columns = ["id", "instrument", "verb", "target"] + (["confidence"] if include_confidence else [])
    rows = []
    for item in sorted(triplets, key=lambda value: (value.get("id") is None, value.get("id"), value.get("instrument") or "", value.get("verb") or "", value.get("target") or "")):
        row = [item.get("id"), item.get("instrument"), item.get("verb"), item.get("target")]
        if include_confidence:
            row.append(item.get("confidence"))
        rows.append(row)
    return {"triplet_columns": columns, "triplets": rows}


def _event_view(result: dict, triplets: list[dict]) -> dict:
    catalog = _triplet_view(triplets)
    columns = ["triplet_id", "start_time", "end_time", "left_truncated", "right_truncated", "original_start_time", "original_end_time"]
    rows = []
    for event in result.get("events") or []:
        if not isinstance(event, dict):
            continue
        left = bool(event.get("left_truncated"))
        right = bool(event.get("right_truncated"))
        rows.append([
            event.get("triplet_id", event.get("id")), event.get("start_time"), event.get("end_time"), left, right,
            event.get("original_start_time") if left else None,
            event.get("original_end_time") if right else None,
        ])
    return {**catalog, "event_columns": columns, "event_intervals": rows}


def model_evidence_view(result: dict, question_type: str | None = None) -> dict:
    """Return the evidence necessary for the current answer type.

    Full tool results remain available to logs and deterministic scoring. The
    model receives this versioned, columnar view to avoid repeated JSON keys.
    """
    if not isinstance(result, dict):
        raise ValueError("tool result must be an object")
    view = {"schema_version": MODEL_EVIDENCE_SCHEMA, "question_type": question_type or "general"}
    view.update({key: result[key] for key in _SCALAR_KEYS if key in result})
    source = _source(result)
    if source is not None:
        view["source"] = source
    triplets = _all_triplets(result)
    if question_type == "instrument":
        view.update(_entity_view(triplets, "instrument"))
    elif question_type == "action":
        view.update(_entity_view(triplets, "verb"))
    elif question_type == "target":
        view.update(_entity_view(triplets, "target"))
    elif question_type == "time_range_summary":
        view.update(_triplet_view(triplets))
    elif question_type == "continuous_event":
        view.update(_event_view(result, triplets))
    else:
        view.update(_triplet_view(triplets))
        if result.get("events"):
            view.update(_event_view(result, triplets))
    return view
