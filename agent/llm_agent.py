import os
import json
import time

from openai import (
    OpenAI,
    RateLimitError,
    APIConnectionError,
    APIError
)

from agent.grounded_reporting import (
    graph_only, image_scope, needs_explanation, render_event, render_video,
)
from agent.memory import ConversationMemory
from agent.prompts import SYSTEM_PROMPT
from agent.confidence import (
    build_grounded_visual_answer,
    enrich_triplet,
    is_perception_only_question
)
from agent.query_router import route_temporal_question, parse_temporal_expression
from agent.video_qa import answer_video_question
from agent.config import get_config
from agent.audit import write_audit_record
from agent.tool_registry import UNIFIED_TOOL_SCHEMAS, execute_registered_tool, model_tool_schemas
from agent.evidence_view import model_evidence_view

from tools.vision_tool import analyze_surgical_image
from tools.knowledge_tool import get_triplet_knowledge
from tools.graph_knowledge_tool import query_surgical_graph

from tools.video_tool import (
    analyze_surgical_video,
    get_temporal_memory,
    query_surgical_timeline
)


# ============================================================
# Tool Definitions
# ============================================================

TOOLS = [

    # ========================================================
    # 1. Image Tool
    # ========================================================

    {
        "type": "function",
        "name": "analyze_surgical_image",
        "description": (
            "分析单张腹腔镜手术图像，"
            "识别器械、动作、目标组织和手术三元组。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "待分析图像的本地路径。"
                }
            },
            "required": [
                "image_path"
            ],
            "additionalProperties": False
        }
    },

    # ========================================================
    # 2. Knowledge Tool
    # ========================================================

    {
        "type": "function",
        "name": "get_triplet_knowledge",
        "description": (
            "查询手术三元组相关知识，"
            "包括器械、动作和目标组织的基本含义。"
        ),
        "parameters": {
            "type": "object",
            "properties": {

                "instrument": {
                    "type": "string",
                    "description": "器械名称。"
                },

                "verb": {
                    "type": "string",
                    "description": "手术动作名称。"
                },

                "target": {
                    "type": "string",
                    "description": "目标组织名称。"
                }
            },
            "required": [
                "instrument",
                "verb",
                "target"
            ],
            "additionalProperties": False
        }
    },

    # ========================================================
    # 3. Video Tool
    # ========================================================

    {
        "type": "function",
        "name": "analyze_surgical_video",
        "description": (
            "分析腹腔镜手术视频，"
            "按时间间隔采样视频帧，"
            "调用视觉模型进行三元组识别，"
            "并建立手术时间事件序列。"
        ),
        "parameters": {
            "type": "object",
            "properties": {

                "video_path": {
                    "type": "string",
                    "description": "待分析视频的本地路径。"
                },

                "sample_interval_seconds": {
                    "type": "number",
                    "description": (
                        "视频采样时间间隔，单位秒。"
                        "默认2秒。"
                    )
                },

                "max_frames": {
                    "type": [
                        "integer",
                        "null"
                    ],
                    "description": (
                        "最多分析多少个采样帧。"
                        "null表示分析完整视频。"
                    )
                }
            },
            "required": [
                "video_path"
            ],
            "additionalProperties": False
        }
    },

    # ========================================================
    # 4. Recent Temporal Events
    # ========================================================

    {
        "type": "function",
        "name": "get_recent_surgical_events",
        "description": (
            "读取最近一次视频分析后保存在 Temporal Memory "
            "中的最近手术事件。"
            "适合回答刚才、最近几个操作等简单时序问题。"
            "该工具不会重新运行视觉模型。"
        ),
        "parameters": {
            "type": "object",
            "properties": {

                "count": {
                    "type": "integer",
                    "description": (
                        "返回最近多少个时间事件，默认10个。"
                    )
                }
            },
            "required": [],
            "additionalProperties": False
        }
    },

    # ========================================================
    # 5. Temporal Timeline Query
    # ========================================================

    {
        "type": "function",
        "name": "query_surgical_timeline",
        "description": (
            "查询已经分析完成的手术视频时间序列。"
            "用于查询某个时间段发生了什么、"
            "视频事件统计、"
            "持续时间最长的操作、"
            "最近事件以及前后操作变化。"
            "如果视频尚未分析，应先调用 analyze_surgical_video。"
        ),
        "parameters": {
            "type": "object",
            "properties": {

                "query_type": {
                    "type": "string",
                    "enum": [
                        "recent",
                        "time_range",
                        "statistics",
                        "longest_action",
                        "transitions"
                    ],
                    "description": "时间序列查询类型。"
                },

                "start_time": {
                    "type": "number",
                    "description": (
                        "time_range 查询的开始时间，单位秒。"
                    )
                },

                "end_time": {
                    "type": "number",
                    "description": (
                        "time_range 查询的结束时间，单位秒。"
                    )
                },

                "count": {
                    "type": "integer",
                    "description": (
                        "recent 查询返回的最近事件数量。"
                    )
                }
            },
            "required": [
                "query_type"
            ],
            "additionalProperties": False
        }
    }

    ,

    # ========================================================
    # 6. Surgical Knowledge Graph
    # ========================================================

    {
        "type": "function",
        "name": "query_surgical_graph",
        "description": (
            "查询 CholecT50 Surgical Knowledge Graph。"
            "可根据 IVT class ID、器械、动作、目标组织"
            "或它们的组合查询数据集中的有效手术三元组关系；"
            "也可以查询知识库中器械支持的动作。"
            "该工具返回的是知识图谱关系，"
            "不是当前视频中的视觉观察结果。"
        ),
        "parameters": {
            "type": "object",
            "properties": {

                "query_type": {
                    "type": "string",
                    "enum": [
                        "triplet",
                        "instrument",
                        "verb",
                        "target",
                        "relation",
                        "supported_actions",
                        "summary"
                    ],
                    "description": (
                        "图谱查询类型。"
                        "triplet=按IVT class查询；"
                        "instrument=按器械查询；"
                        "verb=按动作查询；"
                        "target=按目标查询；"
                        "relation=组合条件查询；"
                        "supported_actions=查询器械知识库支持动作；"
                        "summary=查询图谱基本信息。"
                    )
                },

                "class_id": {
                    "type": [
                        "integer",
                        "null"
                    ],
                    "description": (
                        "IVT class ID，范围通常为0-99。"
                        "query_type=triplet时使用。"
                    )
                },

                "instrument": {
                    "type": [
                        "string",
                        "null"
                    ],
                    "description": "器械名称。"
                },

                "verb": {
                    "type": [
                        "string",
                        "null"
                    ],
                    "description": "动作名称。"
                },

                "target": {
                    "type": [
                        "string",
                        "null"
                    ],
                    "description": "目标组织名称。"
                },

                "include_null": {
                    "type": "boolean",
                    "description": (
                        "是否包含 null triplet。"
                        "默认 false。"
                    )
                }
            },
            "required": [
                "query_type"
            ],
            "additionalProperties": False
        }
    }

]

# New model-generated calls use only validated v1 video/knowledge contracts.
# Legacy functions remain callable by Python code during migration.
TOOLS = [
    item for item in TOOLS
    if item.get("name") in {"analyze_surgical_image", "query_surgical_graph"}
] + model_tool_schemas()


# ============================================================
# Surgical LLM Agent
# ============================================================

class SurgicalLLMAgent:

    def __init__(
        self,
        model=None,
        max_history_turns=10,
        max_tool_rounds=None
    ):

        config = get_config()
        llm_config = config.section("llm")

        # ====================================================
        # DeepSeek API Key
        # ====================================================

        api_key = config.llm_api_key()

        # ====================================================
        # DeepSeek Client
        # ====================================================

        self.client = (
            OpenAI(
                api_key=api_key,
                base_url=llm_config.get(
                    "base_url", "https://api.deepseek.com"
                )
            )
            if api_key
            else None
        )

        # ====================================================
        # DeepSeek Model
        # ====================================================

        self.model = (
            model
            or llm_config.get("model", "deepseek-v4-flash")
        )

        # ====================================================
        # Conversation Memory
        # ====================================================

        self.memory = ConversationMemory(
            max_history_turns=max_history_turns
        )

        self.max_tool_rounds = int(
            max_tool_rounds
            if max_tool_rounds is not None
            else llm_config.get("max_tool_rounds", 6)
        )
        self.llm_provider = llm_config.get("provider", "deepseek")
        self._active_trace = None
        self._last_response_path = None

        # ====================================================
        # Current Video State
        # ====================================================

        self.current_video_path = None
        self.current_video_id = None

        self.last_video_result = None

        print(
            f"[Agent] LLM provider: {self.llm_provider}"
        )

        print(
            f"[Agent] LLM model: "
            f"{self.model}"
        )
        if self.client is None:
            print(
                "[Agent] LLM API key is not configured; "
                "offline deterministic routes remain available."
            )

    # ========================================================
    # Execute Tool
    # ========================================================

    def _execute_tool(
        self,
        tool_name,
        arguments
    ):
        started = time.perf_counter()
        trace_item = {
            "tool": tool_name,
            "arguments": arguments,
        }
        try:
            result = self._execute_tool_impl(tool_name, arguments)
            trace_item["result"] = result
            trace_item["status"] = (
                "error" if isinstance(result, dict) and "error" in result
                else "ok"
            )
            return result
        except Exception as exc:
            trace_item["status"] = "exception"
            trace_item["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            trace_item["elapsed_ms"] = round(
                (time.perf_counter() - started) * 1000, 3
            )
            if self._active_trace is not None:
                self._active_trace.setdefault("tool_calls", []).append(trace_item)

    def _execute_tool_impl(
        self,
        tool_name,
        arguments
    ):
        # ====================================================
        # Tool Name Normalization
        # ====================================================

        tool_name_aliases = {
            "query_surgical_time line":
                "query_surgical_timeline",
            "query surgical timeline":
                "query_surgical_timeline",
            "query surgical graph":
                "query_surgical_graph"
        }

        tool_name = tool_name_aliases.get(
            tool_name,
            tool_name
        )

        if any(schema["name"] == tool_name for schema in UNIFIED_TOOL_SCHEMAS):
            result = execute_registered_tool(tool_name, arguments)
            if tool_name == "analyze_video" and result.get("ok"):
                self.current_video_path = result.get("video", {}).get("path")
                self.current_video_id = result.get("video", {}).get("video_id")
                self.last_video_result = result
            return result

        print(
            f"[Agent] Calling tool: "
            f"{tool_name}"
        )

        # ====================================================
        # 1. Image Analysis
        # ====================================================

        if tool_name == (
            "analyze_surgical_image"
        ):

            image_path = arguments.get(
                "image_path"
            )

            if not image_path:
                return {
                    "error": (
                        "analyze_surgical_image "
                        "缺少 image_path。"
                    )
                }

            result = (
                analyze_surgical_image(
                    image_path
                )
            )

            self.memory.update_vision(
                image_path=image_path,
                vision_result=result
            )

            return result

        # ====================================================
        # 2. Knowledge Query
        # ====================================================

        if tool_name == (
            "get_triplet_knowledge"
        ):

            instrument = arguments.get(
                "instrument"
            )

            verb = arguments.get(
                "verb"
            )

            target = arguments.get(
                "target"
            )

            result = (
                get_triplet_knowledge(
                    instrument=instrument,
                    verb=verb,
                    target=target
                )
            )

            self.memory.update_knowledge(
                result
            )

            return result

        # ====================================================
        # 3. Video Analysis
        # ====================================================

        if tool_name == (
            "analyze_surgical_video"
        ):

            video_path = arguments.get(
                "video_path"
            )

            if not video_path:
                return {
                    "error": (
                        "analyze_surgical_video "
                        "缺少 video_path。"
                    )
                }

            sample_interval_seconds = (
                arguments.get(
                    "sample_interval_seconds",
                    get_config().section("video").get(
                        "sample_interval", 2.0
                    )
                )
            )

            max_frames = arguments.get(
                "max_frames",
                None
            )

            result = (
                analyze_surgical_video(
                    video_path=video_path,
                    sample_interval_seconds=(
                        sample_interval_seconds
                    ),
                    max_frames=max_frames
                )
            )

            self.current_video_path = (
                os.path.abspath(
                    video_path
                )
            )

            self.last_video_result = (
                result
            )
            if result.get("ok"):
                self.current_video_id = result.get("video", {}).get("video_id")

            return result

        # ====================================================
        # 4. Recent Temporal Events
        # ====================================================

        if tool_name == (
            "get_recent_surgical_events"
        ):

            count = arguments.get(
                "count",
                10
            )

            temporal_memory = (
                get_temporal_memory()
            )

            events = (
                temporal_memory
                .get_recent_events(
                    count=count
                )
            )

            context_text = (
                temporal_memory
                .build_context_text(
                    count=count
                )
            )

            return {
                "video_path":
                    self.current_video_path,

                "events":
                    events,

                "context":
                    context_text
            }

        # ====================================================
        # 5. Timeline Query
        # ====================================================

        if tool_name == (
            "query_surgical_timeline"
        ):

            query_type = arguments.get(
                "query_type"
            )

            start_time = arguments.get(
                "start_time",
                None
            )

            end_time = arguments.get(
                "end_time",
                None
            )

            count = arguments.get(
                "count",
                10
            )

            if not query_type:
                return {
                    "error": (
                        "query_surgical_timeline "
                        "缺少 query_type。"
                    )
                }

            result = (
                query_surgical_timeline(
                    query_type=query_type,
                    start_time=start_time,
                    end_time=end_time,
                    count=count
                )
            )

            return result

        # ====================================================
        # 6. Surgical Knowledge Graph
        # ====================================================

        if tool_name == (
            "query_surgical_graph"
        ):

            query_type = arguments.get(
                "query_type"
            )

            class_id = arguments.get(
                "class_id",
                None
            )

            instrument = arguments.get(
                "instrument",
                None
            )

            verb = arguments.get(
                "verb",
                None
            )

            target = arguments.get(
                "target",
                None
            )

            include_null = arguments.get(
                "include_null",
                False
            )

            if not query_type:
                return {
                    "error": (
                        "query_surgical_graph "
                        "缺少 query_type。"
                    )
                }

            result = query_surgical_graph(
                query_type=query_type,
                class_id=class_id,
                instrument=instrument,
                verb=verb,
                target=target,
                include_null=include_null
            )

            return result

        # ====================================================
        # Unknown Tool
        # ====================================================

        return {
            "error": (f"Unknown tool: {tool_name}")
        }

    # ========================================================
    # Build Memory Context
    # ========================================================

    def _build_memory_context(self, include_image=True):
        """
        构建提供给 LLM 的记忆上下文。

        不自动注入完整视频时间线。

        对于具体时间查询，
        应让 LLM 调用 query_surgical_timeline。
        """

        context_parts = []

        # ====================================================
        # 1. Conversation / Image Memory
        # ====================================================

        memory_context = (
            self.memory
            .build_context_text() if include_image else ""
        )

        if memory_context:
            context_parts.append(
                memory_context
            )

        # ====================================================
        # 2. Temporal Summary
        # ====================================================

        temporal_memory = (
            get_temporal_memory()
        )

        temporal_events = (
            temporal_memory
            .get_events()
        )

        if temporal_events:

            try:
                statistics = (
                    temporal_memory
                    .get_statistics()
                )

                video_summary = (
                    "已有一段手术视频完成时序分析。\n"
                    f"当前共记录 "
                    f"{statistics['total_events']} "
                    f"个时序事件，"
                    f"其中 "
                    f"{statistics['action_events']} "
                    f"个明确操作阶段，"
                    f"{statistics['no_action_events']} "
                    f"个未检测到明确操作的阶段。\n"
                    "当前上下文没有提供完整时间线。"
                    "如需查询具体时间点、时间范围、"
                    "最长操作、统计信息或前后变化，"
                    "请调用 query_surgical_timeline。"
                )

            except Exception:

                video_summary = (
                    "已有一段手术视频完成时序分析。\n"
                    "当前上下文没有提供完整时间线。"
                    "如需查询具体时序信息，"
                    "请调用 query_surgical_timeline。"
                )

            context_parts.append(
                video_summary
            )

        # ====================================================
        # 3. Current Video
        # ====================================================

        if self.current_video_path:

            context_parts.append(
                "当前已分析视频：\n"
                f"{self.current_video_path}"
            )

        if self.current_video_id:
            context_parts.append(
                "当前视频标识：\n"
                f"{self.current_video_id}"
            )

        # ====================================================
        # 4. Empty Context
        # ====================================================

        if not context_parts:
            return ""

        # ====================================================
        # 5. Merge
        # ====================================================

        return "\n\n".join(
            context_parts
        )

    # ========================================================
    # Build Current User Message
    # ========================================================

    def _get_routed_temporal_evidence(
        self,
        question,
        image_path=None,
        video_path=None
    ):
        """Route temporal questions before the LLM sees the prompt.

        This guarantees that a point/range/recent question reads only the
        relevant Temporal Memory slice rather than receiving the whole video
        history and relying on the LLM to choose a tool correctly.
        """

        if image_path or video_path:
            return None

        temporal_memory = get_temporal_memory()
        route = route_temporal_question(
            question=question,
            temporal_memory=temporal_memory
        )
        if not route:
            return None

        if self._active_trace is not None:
            self._active_trace["route"] = route

        result = query_surgical_timeline(
            query_type=route["query_type"],
            start_time=route.get("start_time"),
            end_time=route.get("end_time"),
            count=route.get("count", 10)
        )

        events = result.get("events") or []
        selection = route.get("selection")
        if selection == "previous":
            result["selected_event"] = (
                events[-2] if len(events) >= 2 else None
            )
        elif selection == "latest":
            result["selected_event"] = (
                events[-1] if events else None
            )

        return {
            "source_type": "video",
            "video_path": self.current_video_path,
            "route": route,
            "result": result,
            "evidence_rule": (
                "只根据本结果回答对应的时间问题。"
                "type=uncertain 表示证据不足，必须拒绝确定判断；"
                "不得用知识库关系补全视频观察。"
            )
        }

    def _build_user_message(
        self,
        question,
        image_path=None,
        video_path=None
    ):

        sections = []

        sections.append(
            f"用户问题：\n{question}"
        )

        routed_temporal_evidence = (
            self._get_routed_temporal_evidence(
                question=question,
                image_path=image_path,
                video_path=video_path
            )
        )

        if routed_temporal_evidence:
            sections.append(
                "按问题自动检索到的时序证据：\n"
                + json.dumps(
                    routed_temporal_evidence,
                    ensure_ascii=False,
                    default=str
                )
            )

        # ====================================================
        # Image
        # ====================================================

        if image_path:

            absolute_image_path = (
                os.path.abspath(
                    image_path
                )
            )

            sections.append(
                "用户本轮提供了图像：\n"
                f"{absolute_image_path}\n"
                "如果问题涉及当前图像中的视觉内容，"
                "请调用 analyze_surgical_image。"
            )

        # ====================================================
        # Video
        # ====================================================

        if video_path:

            absolute_video_path = (
                os.path.abspath(
                    video_path
                )
            )

            sections.append(
                "用户本轮提供了视频：\n"
                f"{absolute_video_path}\n"
                "如果用户要求分析这段视频，"
                "请调用 analyze_video。"
            )

        # ====================================================
        # Existing Memory Context
        # ====================================================

        memory_context = (
            self._build_memory_context(
                include_image=not routed_temporal_evidence and not video_path
                and not graph_only(question)
            )
        )

        if graph_only(question):
            memory_context = ""

        if memory_context:

            sections.append(
                "当前记忆上下文：\n"
                + memory_context
            )

        else:

            sections.append(
                "当前没有可用的历史记忆上下文。"
            )

        # ====================================================
        # No New Visual Input
        # ====================================================

        if (
            not image_path
            and not video_path
        ):

            sections.append(
                "本轮没有提供新的图像或视频。"
                "可以使用已有 Conversation Memory "
                "和 Temporal Memory 回答指代性问题。"
                "如果需要具体视频时间信息，"
                "请调用 query_surgical_timeline。"
                "不得假装观察到新的视觉内容。"
            )

        return "\n\n".join(
            sections
        )

    # ========================================================
    # Main Answer Function
    # ========================================================

    def answer(
        self, question, image_path=None, video_path=None,
        reference_time=None, simulated_realtime=False,
    ):
        """Answer and always record the actual program/LLM route used."""
        started = time.perf_counter()
        self._last_response_path = "unknown"
        self._active_trace = {
            "question": question,
            "image_path": os.path.abspath(image_path) if image_path else None,
            "video_path": os.path.abspath(video_path) if video_path else None,
            "reference_time": reference_time,
            "simulated_realtime": simulated_realtime,
            "tool_calls": [],
        }
        try:
            answer = self._answer_impl(
                question, image_path, video_path,
                reference_time=reference_time,
                simulated_realtime=simulated_realtime,
            )
            self._active_trace["status"] = "ok"
            self._active_trace["answer"] = answer
            return answer
        except Exception as exc:
            self._active_trace["status"] = "exception"
            self._active_trace["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._active_trace["response_path"] = self._last_response_path
            self._active_trace["llm_provider"] = self.llm_provider
            self._active_trace["llm_model"] = self.model
            self._active_trace["elapsed_ms"] = round(
                (time.perf_counter() - started) * 1000, 3
            )
            write_audit_record(self._active_trace)
            self._active_trace = None

    def _answer_impl(
        self,
        question,
        image_path=None,
        video_path=None,
        reference_time=None,
        simulated_realtime=False,
    ):

        temporal_scope = parse_temporal_expression(
            question, reference_time=reference_time
        )
        if (
            not image_path
            and not needs_explanation(question)
            and (video_path or (self.current_video_id and temporal_scope))
        ):
            qa_result = answer_video_question(
                question=question,
                video_path=video_path,
                video_id=None if video_path else self.current_video_id,
                reference_time=reference_time,
                simulated_realtime=simulated_realtime,
                write_audit=False,
            )
            self._last_response_path = "program_template"
            self.current_video_id = qa_result.get("video_id", self.current_video_id)
            if video_path:
                self.current_video_path = os.path.abspath(video_path)
            self.last_video_result = qa_result.get("structured")
            answer = qa_result["answer"]
            self.memory.add_user_message(question)
            self.memory.add_assistant_message(answer)
            if self._active_trace is not None:
                self._active_trace["route"] = qa_result.get("route")
                self._active_trace["structured_answer"] = qa_result.get("structured")
                self._active_trace["tool_calls"] = qa_result.get("tool_calls", [])
            return answer

        # Core perception questions do not need free-form generation.  Run the
        # RDV model directly and render only the returned labels, which gives a
        # hard anti-hallucination and confidence-language guarantee.
        if (not video_path and is_perception_only_question(question)
                and not self._get_routed_temporal_evidence(question, image_path, video_path)
                and not (self.current_video_path and not image_scope(question, image_path))):
            if image_path:
                vision_result = analyze_surgical_image(image_path)
                self.memory.update_vision(
                    image_path=image_path,
                    vision_result=vision_result
                )

            best_triplet = self.memory.get_best_triplet()
            if best_triplet:
                direct_answer = build_grounded_visual_answer(
                    question=question,
                    triplet=enrich_triplet(best_triplet)
                )
                direct_answer = (
                    f"【静态图像识别结果】来源：{self.memory.current_image_path}\n"
                    + direct_answer
                )
                self.memory.add_user_message(question)
                self.memory.add_assistant_message(direct_answer)
                self._last_response_path = "program_template"
                return direct_answer

        # Deterministic answers for pure temporal questions and video summaries.
        if not image_path and not needs_explanation(question):
            temporal = get_temporal_memory()
            events = temporal.get_events()
            direct_answer = None
            if video_path:
                result = self._execute_tool("analyze_surgical_video", {
                    "video_path": video_path,
                    "sample_interval_seconds": get_config().section(
                        "video"
                    ).get("sample_interval", 2.0),
                })
                if "error" not in result:
                    direct_answer = render_video(
                        result.get("event_sequence", result.get("events", [])),
                        self.current_video_path,
                    )
            elif events and any(w in question for w in ("最后一个采样点", "最后采样点")):
                event = events[-1]
                points = event.get("sample_points") or []
                direct_answer = (
                    f"【视频识别结果】来源：{self.current_video_path}\n"
                    + render_event(event, max(points, key=lambda p: p["timestamp"]) if points else None)
                )
            elif events:
                evidence = self._get_routed_temporal_evidence(question)
                if evidence and evidence["route"].get("intent") == "time_point":
                    matches = evidence["result"].get("events") or []
                    timestamp = evidence["route"]["start_time"]
                    exact = []
                    for event in matches:
                        for point in event.get("sample_points") or []:
                            if abs(float(point["timestamp"]) - timestamp) < 1e-6:
                                exact.append(render_event(event, point))
                    direct_answer = (
                        f"【视频识别结果】来源：{self.current_video_path}\n"
                        + ("\n\n".join(exact) if exact else
                           f"{timestamp}秒没有精确采样点，不能用邻近帧替代该时刻的观察。")
                    )
            if direct_answer is not None:
                self.memory.add_user_message(question)
                self.memory.add_assistant_message(direct_answer)
                self._last_response_path = "program_template"
                return direct_answer

        # ====================================================
        # Build Current User Message
        # ====================================================

        current_user_message = (
            self._build_user_message(
                question=question,
                image_path=image_path,
                video_path=video_path
            )
        )

        # ====================================================
        # Existing Conversation History
        # ====================================================

        conversation = (
            self.memory
            .get_relevant_history(
                question=question,
                max_messages=2
            )
        )

        # 防止原始 memory list 被直接修改
        conversation = list(
            conversation
        )

        if graph_only(question) or self._get_routed_temporal_evidence(
            question, image_path, video_path
        ):
            conversation = []

        conversation.append({
            "role": "user",
            "content":
                current_user_message
        })

        if self.client is None:
            self._last_response_path = "llm_unavailable"
            unavailable = (
                "当前问题需要文字大模型生成回答，但未配置 "
                f"{get_config().section('llm').get('api_key_env', 'API key')}。"
                "离线 RDV 和程序模板路径仍可使用。"
            )
            self.memory.add_user_message(question)
            self.memory.add_assistant_message(unavailable)
            return unavailable

        self._last_response_path = "llm_api_pending"

        # ====================================================
        # Tool Calling Loop
        # ====================================================

        for round_index in range(
            self.max_tool_rounds
        ):

            print(
                f"[Agent] LLM round "
                f"{round_index + 1}"
            )

            try:

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}]
                    + conversation,
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                key: value for key, value in tool.items()
                                if key != "type"
                            },
                        }
                        for tool in TOOLS
                    ],
                )

            # =================================================
            # Rate Limit
            # =================================================

            except RateLimitError as exc:

                self._last_response_path = "llm_error"

                print(
                    "[Agent] DeepSeek API "
                    "rate limit reached."
                )

                print(
                    f"[Agent] Error: {exc}"
                )

                return (
                    "当前 DeepSeek API "
                    "请求达到限额。"
                    "视觉模型、视频分析和"
                    "时间记忆仍然可以正常工作，"
                    "但暂时无法继续调用语言模型。"
                    "请稍后重试。"
                )

            # =================================================
            # Connection
            # =================================================

            except APIConnectionError as exc:

                self._last_response_path = "llm_error"

                print(
                    "[Agent] DeepSeek API "
                    "connection failed."
                )

                print(
                    f"[Agent] Error: {exc}"
                )

                return (
                    "当前无法连接 DeepSeek API，"
                    "请检查网络连接、API 地址"
                    "以及 DeepSeek 服务状态。"
                )

            # =================================================
            # Other API Error
            # =================================================

            except APIError as exc:

                self._last_response_path = "llm_error"

                print(
                    "[Agent] DeepSeek API error."
                )

                print(
                    f"[Agent] Error: {exc}"
                )

                return (
                    "DeepSeek API 当前出现异常，"
                    "请稍后重试。"
                )

            # =================================================
            # Unexpected Error
            # =================================================

            except Exception as exc:

                self._last_response_path = "llm_error"

                print(
                    "[Agent] Unexpected error."
                )

                print(
                    f"[Agent] Error: {exc}"
                )

                return (
                    "LLM 调用过程中出现异常，"
                    "请查看终端日志进行排查。"
                )

            # =================================================
            # Add Model Output
            # =================================================

            message = response.choices[0].message
            tool_calls = message.tool_calls or []
            assistant_message = {
                "role": "assistant",
                "content": message.content,
            }
            if tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ]
            # Preserve provider reasoning fields when returned during tool calls.
            reasoning_content = getattr(message, "reasoning_content", None)
            if reasoning_content is not None:
                assistant_message["reasoning_content"] = reasoning_content
            conversation.append(assistant_message)
            tool_called = bool(tool_calls)

            for call in tool_calls:
                try:
                    arguments = json.loads(call.function.arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("工具参数必须是 JSON 对象")
                    tool_result = self._execute_tool(
                        call.function.name, arguments
                    )
                except Exception as exc:
                    print(f"[Agent] Tool error: {exc}")
                    tool_result = {"error": str(exc)}

                conversation.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(
                        model_evidence_view(tool_result),
                        ensure_ascii=False, default=str, separators=(",", ":")
                    ),
                })

            # =================================================
            # No Tool Call -> Final Answer
            # =================================================

            if not tool_called:

                final_answer = (
                    (message.content or "").strip()
                )

                if not final_answer:

                    final_answer = (
                        "模型本轮没有生成有效文本回答。"
                    )

                # Perception-only answers are rendered from the exact RDV
                # fields.  The LLM cannot replace a label or flatten the
                # confidence distinction in these core answers.
                best_triplet = self.memory.get_best_triplet()
                if (
                    best_triplet
                    and is_perception_only_question(question)
                    and image_scope(question, image_path)
                ):
                    final_answer = build_grounded_visual_answer(
                        question=question,
                        triplet=enrich_triplet(best_triplet)
                    )

                # For any current-image follow-up, low-confidence visual
                # evidence is a hard refusal boundary, including knowledge
                # requests referring to "this operation".
                if best_triplet:
                    enriched = enrich_triplet(best_triplet)
                    refers_to_visual = bool(image_path) or image_scope(question, image_path) and any(
                        marker in question
                        for marker in (
                            "这个器械",
                            "这个操作",
                            "当前图像",
                            "刚才识别"
                        )
                    )
                    if (
                        refers_to_visual
                        and enriched["confidence_level"] == "low"
                    ):
                        final_answer = build_grounded_visual_answer(
                            question=question,
                            triplet=enriched
                        )

                # 保存原始用户问题
                self.memory.add_user_message(
                    question
                )

                self.memory.add_assistant_message(
                    final_answer
                )

                self._last_response_path = "llm_generated"
                return final_answer

        # ====================================================
        # Too Many Tool Rounds
        # ====================================================

        fallback_answer = (
            "本次任务调用工具次数过多，"
            "未能在限定轮次内完成回答。"
        )

        self.memory.add_user_message(
            question
        )

        self.memory.add_assistant_message(
            fallback_answer
        )

        self._last_response_path = "tool_round_limit"
        return fallback_answer

    # ========================================================
    # Reset Conversation Memory
    # ========================================================

    def clear_conversation(
        self
    ):

        self.memory.clear_conversation()

        print(
            "[Agent] Conversation "
            "memory cleared."
        )

    # ========================================================
    # Reset Image Context
    # ========================================================

    def clear_visual_context(
        self
    ):

        self.memory.clear_visual_context()

        print(
            "[Agent] Image visual "
            "context cleared."
        )

    # ========================================================
    # Reset Temporal Memory
    # ========================================================

    def clear_temporal_context(
        self
    ):

        temporal_memory = (
            get_temporal_memory()
        )

        temporal_memory.clear()

        self.current_video_path = None
        self.current_video_id = None

        self.last_video_result = None

        print(
            "[Agent] Temporal "
            "memory cleared."
        )

    # ========================================================
    # Clear All
    # ========================================================

    def clear_all(
        self
    ):

        self.memory.clear_all()

        temporal_memory = (
            get_temporal_memory()
        )

        temporal_memory.clear()

        self.current_video_path = None
        self.current_video_id = None

        self.last_video_result = None

        print(
            "[Agent] All memory cleared."
        )
