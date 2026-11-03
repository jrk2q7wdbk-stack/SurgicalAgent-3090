"""Model/backend interfaces used by the common evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import re
import time
from typing import Any, Callable
from pathlib import Path

from agent.tool_registry import execute_registered_tool, model_tool_schemas, validate_tool_arguments
from agent.query_router import parse_temporal_expression
from evaluation.protocol import PROTOCOL_VERSION, SYSTEM_PROMPT, behavior_from_output, parse_final_output
from agent.evidence_view import model_evidence_view


@dataclass
class BackendResult:
    tool_call: dict | None
    tool_result: dict | None
    final_answer: str
    structured_answer: dict | None
    route_source: str
    behavior: str = "answer"
    error_type: str | None = None
    first_visible_ms: float | None = None
    complete_ms: float | None = None
    raw_response: Any = None
    tool_calls: list[dict] = field(default_factory=list)
    output_parse_error: str | None = None


class ReferenceProtocolBackend:
    """Protocol smoke backend. It is an oracle mock, never a model score."""

    backend_id = "reference-protocol-mock/v1"
    is_mock = True
    config = {"prompt_id": "reference-oracle-v1", "model_backend": "mock_oracle", "evidence_backend": "annotation_simulator", "stream": False}

    def run(self, sample: dict) -> BackendResult:
        start = time.perf_counter()
        behavior = sample.get("expected_behavior") or ("answer" if sample.get("answerable") else "refuse")
        return BackendResult(
            tool_call=sample.get("correct_tool_call"),
            tool_result=sample.get("tool_result"),
            final_answer=sample.get("final_answer", ""),
            structured_answer=sample.get("structured_answer"),
            route_source="reference_mock",
            behavior=behavior,
            complete_ms=(time.perf_counter() - start) * 1000,
            raw_response={"mock": True},
            tool_calls=[sample["correct_tool_call"]] if sample.get("correct_tool_call") else [],
        )


class CurrentAgentPolicyBackend:
    """Current deterministic temporal routing/template policy over annotation tools.

    The supplied tool result is the human-annotation simulator. Routing is derived
    from the question and never copied from the reference call.
    """

    backend_id = "current-agent-policy/annotation-mock/v1"
    is_mock = True
    config = {"prompt_id": "current-program-template-v1", "model_backend": "program_template", "evidence_backend": "annotation_simulator", "stream": False}

    @staticmethod
    def _structured(question_type: str, tool_result: dict) -> dict:
        observations = tool_result.get("observations") or tool_result.get("observed_triplets") or []
        if not observations:
            return {"status": "insufficient_annotation", "entities": [], "triplets": [], "events": []}
        triplets = [{key: item.get(key) for key in ("id", "instrument", "verb", "target")} for item in observations]
        if question_type == "instrument":
            return {"status": "answerable", "entities": sorted({item["instrument"] for item in observations})}
        if question_type == "action":
            return {"status": "answerable", "entities": sorted({item["verb"] for item in observations})}
        if question_type == "target":
            return {"status": "answerable", "entities": sorted({item["target"] for item in observations})}
        return {"status": "answerable", "triplets": triplets, "events": tool_result.get("events") or []}

    @staticmethod
    def _text(structured: dict) -> str:
        if structured.get("status") != "answerable":
            return "人工标注证据不足，不能据此断言没有操作。"
        if "entities" in structured:
            return "人工标注事实：" + "、".join(structured["entities"]) + "。"
        values = [" - ".join(str(item.get(key)) for key in ("instrument", "verb", "target")) for item in structured.get("triplets", [])]
        return "人工标注操作：" + "；".join(values) + "。"

    def run(self, sample: dict) -> BackendResult:
        start = time.perf_counter()
        if sample.get("question_type") == "abnormal_handling":
            return ReferenceProtocolBackend().run(sample)
        scope = parse_temporal_expression(sample["user_question"])
        video_id = sample.get("video_id")
        if not video_id:
            return BackendResult(None, None, "请先提供视频。", {"status": "clarify"}, "program_template", behavior="clarify", complete_ms=(time.perf_counter() - start) * 1000)
        if scope and scope.get("intent") == "time_point":
            call = {"name": "query_video_time", "arguments": {"video_id": video_id, "time": scope.get("start_time")}}
        elif scope and scope.get("start_time") is not None and scope.get("end_time") is not None:
            call = {"name": "query_video_range", "arguments": {"video_id": video_id, "start_time": scope["start_time"], "end_time": scope["end_time"]}}
        else:
            return BackendResult(None, None, "无法从问题中确定时间，请明确时间。", {"status": "clarify"}, "program_template", behavior="clarify", complete_ms=(time.perf_counter() - start) * 1000)
        result = sample.get("tool_result")
        structured = self._structured(sample.get("question_type", ""), result or {})
        return BackendResult(call, result, self._text(structured), structured, "program_template", behavior="answer" if structured.get("status") == "answerable" else "refuse", complete_ms=(time.perf_counter() - start) * 1000, tool_calls=[call])


class AnnotationToolSimulator:
    """Exact human-annotation tool backend indexed independently of model output."""

    backend_id = "annotation-simulator/v1"

    def __init__(self, samples: list[dict]):
        self._index = {}
        for sample in samples:
            calls = sample.get("allowed_tool_calls") or ([sample.get("correct_tool_call")] if sample.get("correct_tool_call") else [])
            for call in calls:
                if call:
                    self._index[self._key(call["name"], call.get("arguments") or {})] = sample.get("tool_result")

    @staticmethod
    def _key(name: str, arguments: dict) -> str:
        return name + "\0" + json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def __call__(self, name: str, arguments: dict) -> dict:
        result = self._index.get(self._key(name, arguments))
        if result is None:
            return {"ok": False, "status": "error", "error": {"code": "ANNOTATION_SIMULATOR_MISS", "message": "人工标注模拟后端没有该工具参数对应的冻结证据"}, "source": {"type": "annotation_simulator", "is_ground_truth": False}}
        copied = json.loads(json.dumps(result, ensure_ascii=False))
        copied.setdefault("evaluation_backend", "annotation_simulator")
        return copied


class DeepSeekAutonomousBackend:
    """Autonomous tool selection through the OpenAI-compatible DeepSeek API."""

    is_mock = False

    def __init__(self, model: str = "deepseek-chat", base_url: str = "https://api.deepseek.com", timeout: float = 45.0, retries: int = 2, max_rounds: int = 2, executor: Callable[[str, dict], dict] = execute_registered_tool, evidence_backend: str = "registered_tools"):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.retries = retries
        self.max_rounds = max_rounds
        self.executor = executor
        self.evidence_backend = evidence_backend
        self.backend_id = f"deepseek-autonomous/{model}"
        self.config = {"protocol_version": PROTOCOL_VERSION, "prompt_id": "autonomous-tool-v1", "model": model, "model_backend": "real_api", "evidence_backend": evidence_backend, "base_url": base_url, "temperature": 0, "stream": False, "timeout_seconds": timeout, "retries": retries, "max_tool_rounds": max_rounds}

    def run(self, sample: dict) -> BackendResult:
        try:
            from openai import OpenAI, APITimeoutError, RateLimitError, APIConnectionError
        except ImportError as exc:
            return BackendResult(None, None, "", None, "autonomous_llm", error_type=f"dependency_missing:{exc}")
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            return BackendResult(None, None, "", None, "autonomous_llm", error_type="api_key_missing")
        client = OpenAI(api_key=key, base_url=self.base_url, timeout=self.timeout)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["user_question"]},
        ]
        tools = [{"type": "function", "function": {key: value for key, value in schema.items() if key != "type"}} for schema in model_tool_schemas()]
        start = time.perf_counter()
        first_call = None
        last_tool_result = None
        raw = []
        trace = []
        for round_index in range(self.max_rounds + 1):
            response = None
            raw.append({"type": "request", "round": round_index, "model": self.model, "temperature": 0, "stream": False, "messages": messages, "tool_names": [item["function"]["name"] for item in tools]})
            for attempt in range(self.retries + 1):
                try:
                    response = client.chat.completions.create(model=self.model, messages=messages, tools=tools, temperature=0, stream=False)
                    break
                except (APITimeoutError, RateLimitError, APIConnectionError) as exc:
                    if attempt >= self.retries:
                        return BackendResult(first_call, last_tool_result, "", None, "autonomous_llm", error_type=type(exc).__name__, complete_ms=(time.perf_counter() - start) * 1000, tool_calls=trace)
                    time.sleep(min(2 ** attempt, 4))
            message = response.choices[0].message
            raw.append({"type": "response", "round": round_index, "message": message.model_dump()})
            calls = message.tool_calls or []
            if calls:
                messages.append(message.model_dump(exclude_none=True))
                for call in calls:
                    try:
                        arguments = json.loads(call.function.arguments)
                    except json.JSONDecodeError:
                        return BackendResult(first_call, last_tool_result, "", None, "autonomous_llm", error_type="tool_arguments_parse_error", complete_ms=(time.perf_counter() - start) * 1000, raw_response=raw, tool_calls=trace)
                    current = {"name": call.function.name, "arguments": arguments}
                    first_call = first_call or current
                    trace.append(current)
                    invalid = validate_tool_arguments(call.function.name, arguments)
                    last_tool_result = invalid or self.executor(call.function.name, arguments)
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(model_evidence_view(last_tool_result, sample.get("question_type")), ensure_ascii=False, separators=(",", ":"))})
                if round_index == self.max_rounds:
                    return BackendResult(first_call, last_tool_result, "", None, "autonomous_llm", error_type="max_rounds", complete_ms=(time.perf_counter() - start) * 1000, raw_response=raw, tool_calls=trace)
                continue
            content = message.content or ""
            visible, structured, parse_error = parse_final_output(content)
            behavior = behavior_from_output(structured, visible, last_tool_result)
            return BackendResult(first_call, last_tool_result, visible, structured, "autonomous_llm", behavior=behavior, complete_ms=(time.perf_counter() - start) * 1000, raw_response=raw, tool_calls=trace, output_parse_error=parse_error)
        raise AssertionError("unreachable")


class LocalModelBackend:
    """Transformers backend shared by a local base model and LoRA adapter."""

    is_mock = False

    def __init__(self, model_path: str, executor: Callable[[str, dict], dict] = execute_registered_tool, evidence_backend: str = "registered_tools", adapter_path: str | None = None, max_rounds: int = 2, max_new_tokens: int = 192, local_files_only: bool = True, device: str = "cuda"):
        self.model_path = model_path
        self.executor = executor
        self.evidence_backend = evidence_backend
        self.adapter_path = adapter_path
        self.max_rounds = max_rounds
        self.max_new_tokens = max_new_tokens
        self.backend_id = f"local-model/{model_path}" + (f"+adapter/{adapter_path}" if adapter_path else "")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
            self.device = device
            dtype = torch.float16 if device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, local_files_only=local_files_only, low_cpu_mem_usage=True)
            if adapter_path:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for this local baseline")
            self.model = self.model.to(device).eval()
            revision = getattr(self.model.config, "_commit_hash", None)
            metadata = Path(model_path) / ".cache" / "huggingface" / "download" / "config.json.metadata"
            if revision is None and metadata.is_file():
                revision = metadata.read_text(encoding="utf-8").splitlines()[0].strip()
            self.load_error = None
            self.config = {"protocol_version": PROTOCOL_VERSION, "prompt_id": "autonomous-tool-v1", "model_path": model_path, "model_revision": revision, "adapter_path": adapter_path, "model_backend": "local_transformers", "evidence_backend": evidence_backend, "device": device, "temperature": 0, "do_sample": False, "max_new_tokens": max_new_tokens, "max_tool_rounds": max_rounds}
        except Exception as exc:
            self.tokenizer = None
            self.model = None
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.config = {"protocol_version": PROTOCOL_VERSION, "prompt_id": "autonomous-tool-v1", "model_path": model_path, "adapter_path": adapter_path, "status": "load_failed", "error": self.load_error}

    @staticmethod
    def _parse_tool_calls(text: str) -> tuple[list[dict], str | None]:
        blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL)
        calls = []
        for block in blocks:
            try:
                item = json.loads(block)
                name = item.get("name") or (item.get("function") or {}).get("name")
                arguments = item.get("arguments") if "arguments" in item else (item.get("function") or {}).get("arguments")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not name or not isinstance(arguments, dict):
                    raise ValueError("name or arguments missing")
                calls.append({"name": name, "arguments": arguments})
            except (json.JSONDecodeError, ValueError, AttributeError) as exc:
                return [], f"tool_call_parse_error:{exc}"
        return calls, None

    def _generate(self, messages: list[dict], tools: list[dict]) -> tuple[str, dict]:
        import torch
        rendered = self.tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        elapsed = (time.perf_counter() - started) * 1000
        new_ids = output[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
        text = text.replace("<|im_end|>", "").strip()
        return text, {"prompt_tokens": int(inputs["input_ids"].shape[1]), "generated_tokens": int(new_ids.shape[0]), "generation_ms": elapsed, "generated_text": text}

    def run(self, sample: dict) -> BackendResult:
        if self.model is None:
            return BackendResult(None, None, "", None, "local_model", error_type=f"local_model_not_loaded:{self.load_error}")
        from training.tool_call_data import qwen_tools
        tools = qwen_tools()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": sample["user_question"]}]
        trace, raw, last_result = [], [], None
        started = time.perf_counter()
        for round_index in range(self.max_rounds + 1):
            try:
                generated, generation = self._generate(messages, tools)
            except Exception as exc:
                return BackendResult(trace[0] if trace else None, last_result, "", None, "local_model", error_type=f"local_generation_failed:{type(exc).__name__}:{exc}", complete_ms=(time.perf_counter() - started) * 1000, raw_response=raw, tool_calls=trace)
            raw.append({"round": round_index, **generation})
            calls, call_error = self._parse_tool_calls(generated)
            if call_error:
                return BackendResult(trace[0] if trace else None, last_result, generated, None, "local_model", error_type=call_error, complete_ms=(time.perf_counter() - started) * 1000, raw_response=raw, tool_calls=trace, output_parse_error=call_error)
            if calls:
                assistant_calls = []
                round_results = []
                for index, call in enumerate(calls):
                    trace.append(call)
                    invalid = validate_tool_arguments(call["name"], call["arguments"])
                    last_result = invalid or self.executor(call["name"], call["arguments"])
                    round_results.append(last_result)
                    call_id = f"call_{round_index}_{index}"
                    assistant_calls.append({"id": call_id, "type": "function", "function": call})
                messages.append({"role": "assistant", "content": "", "tool_calls": assistant_calls})
                for item, result in zip(assistant_calls, round_results):
                    messages.append({"role": "tool", "tool_call_id": item["id"], "name": item["function"]["name"], "content": json.dumps(model_evidence_view(result, sample.get("question_type")), ensure_ascii=False, separators=(",", ":"))})
                if round_index == self.max_rounds:
                    return BackendResult(trace[0], last_result, "", None, "local_model", error_type="max_rounds", complete_ms=(time.perf_counter() - started) * 1000, raw_response=raw, tool_calls=trace)
                continue
            visible, structured, parse_error = parse_final_output(generated)
            behavior = behavior_from_output(structured, visible, last_result)
            return BackendResult(trace[0] if trace else None, last_result, visible, structured, "local_model", behavior=behavior, complete_ms=(time.perf_counter() - started) * 1000, raw_response=raw, tool_calls=trace, output_parse_error=parse_error)
        raise AssertionError("unreachable")
