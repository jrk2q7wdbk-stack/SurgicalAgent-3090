"""Deterministic offline video QA grounded only in RDV tool results."""

from __future__ import annotations

import time
from typing import Any

from agent.audit import write_audit_record
from agent.query_router import parse_temporal_expression
from tools.video_tool import (
    analyze_surgical_video,
    get_current_video_id,
    query_recent_video,
    query_video_range,
    query_video_time,
    summarize_video_events,
)


def _tool_error_answer(result: dict) -> str:
    error = result.get("error") or {}
    return f"无法完成视频查询：{error.get('message', '未知工具错误')}"


def _triplet_text(item: dict) -> str:
    return " - ".join(str(item.get(key) or "unknown") for key in (
        "instrument", "verb", "target"
    ))


def _render_point(result: dict) -> str:
    if not result.get("ok"):
        return _tool_error_answer(result)
    requested = result["requested_time"]
    actual = result["actual_frame_time"]
    header = (
        f"【RDV预测；非人工真值】请求时刻 {requested:.3f} 秒；"
        f"实际读取帧时刻 {actual:.3f} 秒（帧 {result['frame_id']}）。"
    )
    predictions = result.get("predictions") or []
    if predictions:
        lines = [header, "该帧达到保留阈值的预测为："]
        for item in predictions:
            lines.append(
                f"- {_triplet_text(item)}；分数 {float(item.get('confidence', 0.0)):.3f}"
            )
        return "\n".join(lines)
    candidate = result.get("best_candidate")
    if candidate:
        return (
            header + "\n模型没有识别到达到保留阈值的三元组；"
            f"最高候选为 {_triplet_text(candidate)}，"
            f"分数 {float(candidate.get('confidence', 0.0)):.3f}。"
            "这表示模型证据不足，不能断言该时刻没有操作。"
        )
    return (
        header + "\n模型未返回三元组候选。"
        "这不等于确定没有操作。"
    )


def _render_summary(result: dict) -> str:
    if not result.get("ok"):
        return _tool_error_answer(result)
    lines = [
        "【RDV预测汇总；非人工真值】",
        f"原视频绝对时间范围：{result['start_time']:.3f}～{result['end_time']:.3f} 秒。",
    ]
    triplets = result.get("predicted_triplets") or []
    if not triplets:
        lines.append(
            "范围内采样点没有保留到达到阈值的三元组；"
            "不能据此断言范围内没有操作。"
        )
    else:
        lines.append("范围内采样点出现过的预测操作：")
        for item in triplets:
            times = ", ".join(f"{float(value):.3f}" for value in item["evidence_times"])
            lines.append(
                f"- {_triplet_text(item)}；最高分 {item['max_score']:.3f}；"
                f"证据帧时刻 [{times}] 秒"
            )
    lines.append("采样间隙不作推断；预测事件组不是真实手术阶段。")
    return "\n".join(lines)


def answer_video_question(
    question: str,
    video_path: str = None,
    video_id: str = None,
    sample_interval: float = None,
    reference_time: float = None,
    simulated_realtime: bool = False,
    force: bool = False,
    write_audit: bool = True,
) -> dict:
    """Answer one offline/replay question with structured evidence.

    For simulated real-time queries, ``reference_time`` defines both "now"
    and the maximum accessible timestamp.  Future samples are never returned.
    """
    started = time.perf_counter()
    scope = parse_temporal_expression(question, reference_time=reference_time)
    trace = {
        "question": question,
        "video_path": video_path,
        "video_id": video_id,
        "route": scope,
        "response_path": "program_template",
        "tool_calls": [],
    }
    structured: dict[str, Any]
    answer: str
    try:
        if scope and scope.get("needs_reference_time"):
            structured = {
                "ok": False,
                "status": "needs_reference_time",
                "error": {
                    "code": "REFERENCE_TIME_REQUIRED",
                    "message": "“当前/过去N秒”需要程序明确提供 reference_time",
                },
            }
            answer = _tool_error_answer(structured)
            return {"answer": answer, "structured": structured, "route": scope,
                    "tool_calls": trace["tool_calls"]}

        if simulated_realtime and reference_time is None:
            structured = {
                "ok": False,
                "status": "needs_reference_time",
                "error": {
                    "code": "REFERENCE_TIME_REQUIRED",
                    "message": "模拟实时模式必须提供 reference_time",
                },
            }
            answer = _tool_error_answer(structured)
            return {"answer": answer, "structured": structured, "route": scope,
                    "tool_calls": trace["tool_calls"]}

        active_video_id = video_id or get_current_video_id()
        if video_path:
            start = 0.0
            end = None
            if scope and scope.get("start_time") is not None:
                start = scope["start_time"]
                end = scope["end_time"]
            if simulated_realtime:
                end = reference_time if end is None else min(end, reference_time)
            arguments = {
                "video_path": video_path,
                "start_time": start,
                "end_time": end,
                "sample_interval_seconds": sample_interval,
                "force": force,
            }
            analysis = analyze_surgical_video(**arguments)
            trace["tool_calls"].append({
                "tool": "analyze_video", "arguments": arguments,
                "result": analysis,
            })
            if not analysis.get("ok"):
                answer = _tool_error_answer(analysis)
                return {"answer": answer, "structured": analysis, "route": scope,
                        "tool_calls": trace["tool_calls"]}
            active_video_id = analysis["video"]["video_id"]

        if not active_video_id:
            structured = {
                "ok": False, "status": "error",
                "error": {"code": "VIDEO_REQUIRED",
                          "message": "请提供 video_path 或已分析的 video_id"},
            }
            answer = _tool_error_answer(structured)
            return {"answer": answer, "structured": structured, "route": scope,
                    "tool_calls": trace["tool_calls"]}

        available_until = reference_time if simulated_realtime else None
        intent = (scope or {}).get("intent", "full_video")
        if intent in ("time_point", "current"):
            arguments = {
                "video_id": active_video_id,
                "time": scope["start_time"],
                "available_until": available_until,
            }
            structured = query_video_time(**arguments)
            trace["tool_calls"].append({
                "tool": "query_video_time", "arguments": arguments,
                "result": structured,
            })
            answer = _render_point(structured)
        elif intent == "recent_duration":
            arguments = {
                "video_id": active_video_id,
                "reference_time": reference_time,
                "duration": scope["duration"],
            }
            ranged = query_recent_video(**arguments)
            trace["tool_calls"].append({
                "tool": "query_video_range", "arguments": arguments,
                "result": ranged,
            })
            structured = (
                summarize_video_events(
                    active_video_id, ranged.get("start_time"), ranged.get("end_time"),
                    available_until=reference_time,
                ) if ranged.get("ok") else ranged
            )
            answer = _render_summary(structured)
        else:
            start = scope.get("start_time") if scope else None
            end = scope.get("end_time") if scope else None
            arguments = {
                "video_id": active_video_id,
                "start_time": start,
                "end_time": end,
                "available_until": available_until,
            }
            structured = summarize_video_events(**arguments)
            trace["tool_calls"].append({
                "tool": "summarize_video_events", "arguments": arguments,
                "result": structured,
            })
            answer = _render_summary(structured)
        return {"answer": answer, "structured": structured, "route": scope,
                "video_id": active_video_id, "tool_calls": trace["tool_calls"]}
    finally:
        trace["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
        if "answer" in locals():
            trace["answer"] = answer
        if "structured" in locals():
            trace["status"] = structured.get("status")
        if write_audit:
            write_audit_record(trace)

