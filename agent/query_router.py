"""Rule-based routing for surgical temporal-memory questions."""

import re


_CHINESE_NUMBERS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十五": 15,
    "二十": 20,
    "三十": 30,
    "六十": 60,
}


def _number(value):
    if value is None:
        return None
    if value in _CHINESE_NUMBERS:
        return float(_CHINESE_NUMBERS[value])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_time(temporal_memory):
    latest = temporal_memory.get_latest_event()
    if not latest:
        return None
    return float(latest.get("end_time", latest.get("start_time", 0.0)))


def parse_temporal_expression(question, reference_time=None):
    """Parse time scope without requiring an already populated memory.

    ``reference_time`` is mandatory for current/recent expressions in the new
    simulated-real-time interface.  The legacy router below may explicitly
    choose the latest analyzed sample as its reference for compatibility.
    """
    text = (question or "").strip().lower()
    if not text:
        return None

    range_match = re.search(
        r"(?:第)?\s*(\d+(?:\.\d+)?)\s*秒?\s*(?:到|至|[-~—])\s*"
        r"(?:第)?\s*(\d+(?:\.\d+)?)\s*秒",
        text,
    )
    if range_match:
        return {
            "intent": "time_range",
            "start_time": float(range_match.group(1)),
            "end_time": float(range_match.group(2)),
        }

    recent_match = re.search(
        r"(?:最近|过去)\s*(\d+(?:\.\d+)?|一|二|两|三|四|五|六|七|八|九|十|十五|二十|三十|六十)\s*秒",
        text,
    )
    if recent_match:
        duration = _number(recent_match.group(1))
        if reference_time is None:
            return {
                "intent": "recent_duration",
                "duration": duration,
                "needs_reference_time": True,
            }
        return {
            "intent": "recent_duration",
            "duration": duration,
            "reference_time": float(reference_time),
            "start_time": max(0.0, float(reference_time) - duration),
            "end_time": float(reference_time),
        }

    point_match = re.search(r"(?:第|在)?\s*(\d+(?:\.\d+)?)\s*秒(?:时|处)?", text)
    if point_match:
        value = float(point_match.group(1))
        return {"intent": "time_point", "start_time": value, "end_time": value}

    if any(word in text for word in ("当前", "现在", "此刻")):
        if reference_time is None:
            return {"intent": "current", "needs_reference_time": True}
        value = float(reference_time)
        return {
            "intent": "current", "reference_time": value,
            "start_time": value, "end_time": value,
        }

    if any(word in text for word in ("分析这个视频", "分析这段视频", "视频中发生")):
        return {"intent": "full_video"}
    return None


def route_temporal_question(question, temporal_memory):
    """Return a targeted timeline query, or ``None`` for non-temporal text."""

    text = (question or "").strip().lower()
    if not text or not temporal_memory.get_events():
        return None

    if any(word in text for word in ("最长", "持续时间最长")):
        return {"intent": "longest", "query_type": "longest_action"}

    if any(word in text for word in ("多少个操作阶段", "多少个阶段", "统计")):
        return {"intent": "statistics", "query_type": "statistics"}

    if any(word in text for word in ("前后", "变化", "转换", "顺序")):
        return {"intent": "transitions", "query_type": "transitions"}

    parsed = parse_temporal_expression(question)
    if parsed and parsed.get("intent") == "time_range":
        return {
            "intent": "time_range",
            "query_type": "time_range",
            "start_time": parsed["start_time"],
            "end_time": parsed["end_time"],
        }

    if parsed and parsed.get("intent") == "recent_duration":
        duration = parsed.get("duration")
        end_time = _latest_time(temporal_memory)
        if duration is not None and end_time is not None:
            return {
                "intent": "recent_duration",
                "query_type": "time_range",
                "start_time": max(0.0, end_time - duration),
                "end_time": end_time,
            }

    if parsed and parsed.get("intent") in ("time_point", "current"):
        timestamp = (
            parsed.get("start_time")
            if parsed.get("start_time") is not None
            else _latest_time(temporal_memory)
        )
        return {
            "intent": "time_point",
            "query_type": "time_range",
            "start_time": timestamp,
            "end_time": timestamp,
        }

    if any(word in text for word in ("上一个事件", "前一个事件", "上一事件")):
        return {
            "intent": "previous_event",
            "query_type": "recent",
            "count": 2,
            "selection": "previous",
        }

    if any(word in text for word in ("当前事件", "当前操作", "现在", "此刻")):
        return {
            "intent": "current_event",
            "query_type": "recent",
            "count": 1,
            "selection": "latest",
        }

    if any(word in text for word in ("刚才", "最近发生", "最近的操作", "之前发生")):
        return {"intent": "recent", "query_type": "recent", "count": 5}

    return None
