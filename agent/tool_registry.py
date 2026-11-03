"""Validated, versioned tool contracts shared by API and local backends."""

from __future__ import annotations

from typing import Any, Callable, Dict
from copy import deepcopy

TOOL_API_VERSION = "surgical-agent-tools/v1"


def _function(name: str, description: str, properties: dict, required=None) -> dict:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required or [],
            "additionalProperties": False,
        },
    }


UNIFIED_TOOL_SCHEMAS = [
    _function(
        "analyze_video",
        "使用RDV分析原视频的指定绝对时间范围并持久化预测结果。",
        {
            "video_path": {"type": "string"},
            "start_time": {"type": "number"},
            "end_time": {"type": ["number", "null"]},
            "sample_interval": {"type": "number"},
            "max_frames": {"type": ["integer", "null"]},
            "force": {"type": "boolean"},
        },
        ["video_path"],
    ),
    _function(
        "query_video_time",
        "查询视频指定时刻；缺少采样点时可读取该时刻对应帧补充RDV识别。",
        {
            "video_id": {"type": "string"},
            "time": {"type": "number"},
            "available_until": {"type": ["number", "null"]},
            "supplement_if_missing": {"type": "boolean"},
        },
        ["video_id", "time"],
    ),
    _function(
        "query_video_range",
        "查询原视频绝对时间范围内已有的RDV采样预测。",
        {
            "video_id": {"type": "string"},
            "start_time": {"type": "number"},
            "end_time": {"type": "number"},
            "available_until": {"type": ["number", "null"]},
        },
        ["video_id", "start_time", "end_time"],
    ),
    _function(
        "summarize_video_events",
        "汇总指定范围内的预测操作及采样证据；结果不是真实手术阶段。",
        {
            "video_id": {"type": "string"},
            "start_time": {"type": ["number", "null"]},
            "end_time": {"type": ["number", "null"]},
            "available_until": {"type": ["number", "null"]},
        },
        ["video_id"],
    ),
    _function(
        "query_surgical_knowledge",
        "查询已有知识文件中的器械、动作、靶点或完整操作知识。",
        {
            "query": {"type": ["string", "null"]},
            "instrument": {"type": ["string", "null"]},
            "action": {"type": ["string", "null"]},
            "target": {"type": ["string", "null"]},
        },
        [],
    ),
]

_MODEL_TOOL_DESCRIPTIONS = {
    "analyze_video": "Run RDV on a video segment and cache its evidence.",
    "query_video_time": "Get video evidence at one absolute time.",
    "query_video_range": "Get video evidence in an absolute time range.",
    "summarize_video_events": "Summarize operation evidence in a video time range.",
    "query_surgical_knowledge": "Query traceable knowledge; do not use it for current-video facts.",
}


def model_tool_schemas() -> list[dict]:
    """Return the shared concise schema exposed to every language backend."""
    schemas = deepcopy(UNIFIED_TOOL_SCHEMAS)
    for schema in schemas:
        schema["description"] = _MODEL_TOOL_DESCRIPTIONS[schema["name"]]
    return schemas


_SCHEMAS = {item["name"]: item for item in UNIFIED_TOOL_SCHEMAS}


def _error(code: str, message: str, **details: Any) -> dict:
    return {
        "ok": False,
        "status": "error",
        "tool_api_version": TOOL_API_VERSION,
        "error": {"code": code, "message": message, "details": details},
    }


def _matches_type(value: Any, expected: Any) -> bool:
    choices = expected if isinstance(expected, list) else [expected]
    for name in choices:
        if name == "null" and value is None:
            return True
        if name == "string" and isinstance(value, str):
            return True
        if name == "boolean" and isinstance(value, bool):
            return True
        if name == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if name == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
    return False


def validate_tool_arguments(name: str, arguments: Any) -> dict | None:
    schema = _SCHEMAS.get(name)
    if schema is None:
        return _error("UNKNOWN_TOOL", f"未知工具：{name}")
    if not isinstance(arguments, dict):
        return _error("INVALID_ARGUMENTS", "工具参数必须是JSON对象")
    parameters = schema["parameters"]
    properties = parameters["properties"]
    unknown = sorted(set(arguments) - set(properties))
    if unknown:
        return _error("UNKNOWN_ARGUMENT", "包含未定义参数", arguments=unknown)
    missing = [key for key in parameters["required"] if key not in arguments]
    if missing:
        return _error("MISSING_ARGUMENT", "缺少必要参数", arguments=missing)
    for key, value in arguments.items():
        if not _matches_type(value, properties[key]["type"]):
            return _error(
                "INVALID_ARGUMENT_TYPE", f"参数 {key} 类型错误",
                expected=properties[key]["type"], actual=type(value).__name__,
            )
    # Semantic validation is shared by API and local-model execution. Duration
    # checks remain in the video tool because they require loaded video metadata.
    for key in ("time", "start_time", "end_time", "available_until"):
        value = arguments.get(key)
        if value is not None and isinstance(value, (int, float)) and value < 0:
            return _error("TIME_NEGATIVE", f"参数 {key} 不能为负数", argument=key, value=value)
    start, end = arguments.get("start_time"), arguments.get("end_time")
    if start is not None and end is not None and start > end:
        return _error("INVALID_TIME_RANGE", "start_time 不能晚于 end_time", start_time=start, end_time=end)
    interval = arguments.get("sample_interval")
    if interval is not None and interval <= 0:
        return _error("INVALID_SAMPLE_INTERVAL", "sample_interval 必须大于0", value=interval)
    max_frames = arguments.get("max_frames")
    if max_frames is not None and max_frames <= 0:
        return _error("INVALID_MAX_FRAMES", "max_frames 必须大于0", value=max_frames)
    return None


def _knowledge(arguments: dict) -> dict:
    from tools.graph_knowledge_tool import query_surgical_graph
    from tools.knowledge_tool import (
        get_instrument_knowledge, get_target_knowledge,
        get_triplet_knowledge, get_verb_knowledge,
    )
    instrument = arguments.get("instrument")
    action = arguments.get("action")
    target = arguments.get("target")
    query = (arguments.get("query") or "").strip().lower()
    if instrument and action and target:
        result = get_triplet_knowledge(instrument, action, target)
    elif instrument:
        result = get_instrument_knowledge(instrument)
    elif action:
        result = get_verb_knowledge(action)
    elif target:
        result = get_target_knowledge(target)
    elif query:
        # Query only traceable class relations; no free-form external facts.
        result = query_surgical_graph("summary") if query in (
            "summary", "概况", "知识库概况"
        ) else None
        if result is None:
            return _error(
                "KNOWLEDGE_QUERY_NEEDS_ENTITY",
                "自由查询词无法可靠映射，请提供器械、动作或靶点名称",
                query=query,
            )
    else:
        return _error(
            "MISSING_ARGUMENT", "至少提供 query、instrument、action 或 target"
        )
    return {
        "ok": True,
        "status": "completed",
        "tool_api_version": TOOL_API_VERSION,
        "source": "configured_surgical_knowledge_files",
        "result": result,
    }


def execute_registered_tool(name: str, arguments: dict) -> dict:
    validation_error = validate_tool_arguments(name, arguments)
    if validation_error:
        return validation_error
    from tools.video_tool import (
        analyze_surgical_video, query_video_range,
        query_video_time, summarize_video_events,
    )
    handlers: Dict[str, Callable[..., dict]] = {
        "analyze_video": lambda **kw: analyze_surgical_video(
            video_path=kw["video_path"],
            start_time=kw.get("start_time", 0.0),
            end_time=kw.get("end_time"),
            sample_interval_seconds=kw.get("sample_interval"),
            max_frames=kw.get("max_frames"),
            force=kw.get("force", False),
        ),
        "query_video_time": query_video_time,
        "query_video_range": query_video_range,
        "summarize_video_events": summarize_video_events,
        "query_surgical_knowledge": lambda **kw: _knowledge(kw),
    }
    try:
        result = handlers[name](**arguments)
    except Exception as exc:
        return _error("TOOL_EXECUTION_FAILED", f"{type(exc).__name__}: {exc}")
    if not isinstance(result, dict):
        return _error("INVALID_TOOL_RESULT", "工具没有返回JSON对象")
    result.setdefault("tool_api_version", TOOL_API_VERSION)
    return result


# Legacy registry retained for imports in older scripts.
TOOLS = {item["name"]: item for item in UNIFIED_TOOL_SCHEMAS}

