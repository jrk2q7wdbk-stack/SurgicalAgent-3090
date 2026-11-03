"""Generate abnormal, clarification and tool-error samples without changing the base set."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable


ABNORMAL_SCHEMA = "surgical-tool-abnormal-qa/v1"


@dataclass(frozen=True)
class AbnormalConfig:
    train_per_type: int = 5
    validation_per_type: int = 2
    test_per_type: int = 1
    seed: int = 20261102


CASES = {
    "negative_time": ("refuse", "parameter_validation"),
    "time_beyond_video": ("refuse", "parameter_validation"),
    "inverted_time_range": ("refuse", "parameter_validation"),
    "missing_video": ("clarify", "model"),
    "missing_time": ("clarify", "model"),
    "ambiguous_time_reference": ("clarify", "model"),
    "video_read_failure": ("report_tool_error", "tool"),
    "rdv_inference_failure": ("report_tool_error", "tool"),
    "empty_or_insufficient_evidence": ("refuse", "model"),
    "unsupported_question": ("refuse", "model"),
}


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _id(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return "abn-" + hashlib.sha256(raw).hexdigest()[:20]


def _error(code: str, message: str, layer: str) -> dict:
    return {
        "ok": False,
        "status": "error",
        "error": {"code": code, "message": message, "layer": layer},
        "source": {"type": "error", "is_ground_truth": False},
    }


def _case_payload(case: str, video_id: str, duration: float, variant: int, evidence: dict | None) -> dict:
    suffix = "" if variant == 0 else f"（样例{variant + 1}）"
    if case == "negative_time":
        args = {"video_id": video_id, "time": -float(variant + 1)}
        return {"question": f"查看{video_id}负{variant + 1}秒的操作{suffix}", "tool": {"name": "query_video_time", "arguments": args}, "result": _error("TIME_NEGATIVE", "time must be non-negative", "parameter_validation"), "answer": "时间不能为负数，请提供不小于0秒的时间。", "state": "invalid_parameter"}
    if case == "time_beyond_video":
        value = duration + variant + 10.0
        args = {"video_id": video_id, "time": value}
        return {"question": f"查看{video_id}第{value:g}秒的操作{suffix}", "tool": {"name": "query_video_time", "arguments": args}, "result": _error("TIME_OUT_OF_RANGE", "time exceeds video duration", "parameter_validation"), "answer": f"查询时间超出视频范围；{video_id}可用标注结束于约{duration:g}秒。", "state": "invalid_parameter"}
    if case == "inverted_time_range":
        start, end = 30.0 + variant, 10.0
        args = {"video_id": video_id, "start_time": start, "end_time": end}
        return {"question": f"汇总{video_id}第{start:g}秒到{end:g}秒{suffix}", "tool": {"name": "query_video_range", "arguments": args}, "result": _error("INVALID_TIME_RANGE", "start_time is later than end_time", "parameter_validation"), "answer": "开始时间晚于结束时间，请重新给出有效范围。", "state": "invalid_parameter"}
    if case == "missing_video":
        return {"question": f"第{12 + variant}秒用了什么器械？{suffix}", "tool": None, "result": None, "answer": "请先选择或提供要查询的视频。", "state": "missing_context"}
    if case == "missing_time":
        return {"question": f"查询{video_id}指定时刻的操作{suffix}", "tool": None, "result": None, "answer": "请提供要查询的具体时间。", "state": "missing_parameter"}
    if case == "ambiguous_time_reference":
        return {"question": f"看一下刚才那个时候发生了什么{suffix}", "tool": None, "result": None, "answer": "“刚才那个时候”没有明确参考时刻，请提供具体秒数或程序给出的当前时刻。", "state": "ambiguous_reference"}
    if case == "video_read_failure":
        tool = {"name": "analyze_video", "arguments": {"video_path": f"missing_{variant}.mp4"}}
        return {"question": f"分析这个无法读取的视频{suffix}", "tool": tool, "result": _error("VIDEO_READ_FAILED", "video could not be opened", "tool"), "answer": "视频读取失败，未获得任何视觉证据；请检查文件路径和编码。", "state": "tool_error"}
    if case == "rdv_inference_failure":
        tool = {"name": "query_video_time", "arguments": {"video_id": video_id, "time": 12.0 + variant}}
        return {"question": f"查询{video_id}第{12 + variant}秒的操作{suffix}", "tool": tool, "result": _error("RDV_INFERENCE_FAILED", "RDV inference failed", "tool"), "answer": "RDV推理失败，当前没有可用于回答的预测证据。", "state": "tool_error"}
    if case == "empty_or_insufficient_evidence":
        if evidence:
            return {"question": evidence["user_question"], "tool": evidence["correct_tool_call"], "result": evidence["tool_result"], "answer": evidence["final_answer"], "state": "annotation_has_no_positive_triplet"}
        tool = {"name": "query_video_time", "arguments": {"video_id": video_id, "time": 0.0}}
        result = {"ok": True, "status": "annotation_unavailable", "observations": [], "source": {"type": "human_annotation", "is_ground_truth": True}, "warning": "Missing annotation is not evidence of absence."}
        return {"question": f"{video_id}第0秒有什么操作？{suffix}", "tool": tool, "result": result, "answer": "该时刻缺少足够标注证据，不能据此断言没有操作。", "state": "unannotated"}
    return {"question": f"根据{video_id}判断患者明年的恢复情况{suffix}", "tool": None, "result": None, "answer": "该问题超出当前视频操作查询和可追溯知识工具的支持范围，无法回答。", "state": "unsupported_scope"}


def _conversation(record: dict) -> dict:
    messages = [
        {"role": "system", "content": "只依据已验证工具结果回答；参数缺失时澄清，工具失败时如实报告，不把空记录说成确定不存在。"},
        {"role": "user", "content": record["user_question"]},
    ]
    call = record.get("correct_tool_call")
    if call:
        messages.extend([
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=False, sort_keys=True)}}]},
            {"role": "tool", "tool_call_id": "call_1", "name": call["name"], "content": json.dumps(record["tool_result"], ensure_ascii=False, sort_keys=True)},
        ])
    messages.append({"role": "assistant", "content": record["final_answer"]})
    return {"messages": messages, "metadata": {"sample_id": record["sample_id"], "split": record["split"], "abnormal_type": record["abnormal_type"]}}


def validate_abnormal_record(record: dict) -> list[str]:
    errors = []
    if record.get("abnormal_type") not in CASES:
        errors.append("unknown_abnormal_type")
    expected = CASES.get(record.get("abnormal_type"))
    if expected and (record.get("expected_behavior"), record.get("handling_layer")) != expected:
        errors.append("behavior_or_layer_mismatch")
    behavior = record.get("expected_behavior")
    call = record.get("correct_tool_call")
    if behavior == "clarify" and call is not None:
        errors.append("clarification_must_not_call_tool")
    if behavior == "report_tool_error" and not (record.get("tool_result") or {}).get("error"):
        errors.append("tool_error_sample_missing_error")
    if record.get("source_state") in {"unannotated", "annotation_has_no_positive_triplet", "rdv_empty"} and "没有操作" in record.get("final_answer", ""):
        errors.append("empty_evidence_incorrectly_asserts_absence")
    return errors


def generate_abnormal_data(base_dir: str | Path, output_dir: str | Path, config: AbnormalConfig = AbnormalConfig()) -> dict:
    base = Path(base_dir).resolve()
    output = Path(output_dir).resolve()
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    rng = random.Random(config.seed)
    distribution = {}
    issues = []
    for split, per_type in (("train", config.train_per_type), ("validation", config.validation_per_type), ("test_supplement", config.test_per_type)):
        source_split = "test" if split == "test_supplement" else split
        records = _read_jsonl(base / f"{source_split}.standardized.jsonl")
        videos = sorted({row["video_id"] for row in records})
        durations = {video: max(row["time_range"][1] for row in records if row["video_id"] == video) for video in videos}
        empty = [row for row in records if not row.get("answerable")]
        rows = []
        for case, (behavior, layer) in CASES.items():
            for index in range(per_type):
                video_id = videos[(index + list(CASES).index(case)) % len(videos)]
                evidence = empty[index % len(empty)] if empty and case == "empty_or_insufficient_evidence" else None
                split_offset = {"train": 0, "validation": 100, "test_supplement": 200}[split]
                payload = _case_payload(case, video_id, durations[video_id], index + split_offset, evidence)
                identity = {"split": split, "case": case, "index": index, "question": payload["question"]}
                record = {
                    "schema_version": ABNORMAL_SCHEMA,
                    "sample_id": _id(identity),
                    "split": split,
                    "question_type": "abnormal_handling",
                    "user_question": payload["question"],
                    "correct_tool_call": payload["tool"],
                    "tool_result": payload["result"],
                    "final_answer": payload["answer"],
                    "structured_answer": {"status": behavior, "abnormal_type": case, "source_state": payload["state"]},
                    "video_id": video_id if case not in {"missing_video", "ambiguous_time_reference"} else None,
                    "time_range": None,
                    "answerable": False,
                    "abnormal_type": case,
                    "expected_behavior": behavior,
                    "handling_layer": layer,
                    "source_state": payload["state"],
                    "generation_basis": "deterministic contract case; no normal evidence fabricated",
                    "frozen_supplement": split == "test_supplement",
                }
                errors = validate_abnormal_record(record)
                if errors:
                    issues.append({"sample_id": record["sample_id"], "errors": errors})
                rows.append(record)
        rng.shuffle(rows)
        _write_jsonl(output / f"{split}.abnormal.standardized.jsonl", rows)
        _write_jsonl(output / f"{split}.abnormal.conversations.jsonl", (_conversation(row) for row in rows))
        distribution[split] = {"count": len(rows), "per_type": per_type, "types": {case: per_type for case in CASES}}
    result = {
        "schema_version": ABNORMAL_SCHEMA,
        "valid": not issues,
        "issues": issues,
        "seed": config.seed,
        "configuration": {"train_per_type": config.train_per_type, "validation_per_type": config.validation_per_type, "test_per_type": config.test_per_type},
        "distribution": distribution,
        "base_manifest": str(base / "manifest.json"),
        "base_test_sha256": hashlib.sha256((base / "test.standardized.jsonl").read_bytes()).hexdigest(),
        "test_policy": "The original test set is unchanged. test_supplement is separate and frozen.",
        "tool_api_version": "surgical-agent-tools/v1",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def validate_abnormal_directory(input_dir: str | Path) -> dict:
    root = Path(input_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    issues = []
    seen_ids = set()
    questions_by_split = {}
    distribution = {}
    for split in ("train", "validation", "test_supplement"):
        records = _read_jsonl(root / f"{split}.abnormal.standardized.jsonl")
        conversations = _read_jsonl(root / f"{split}.abnormal.conversations.jsonl")
        questions_by_split[split] = {row["user_question"] for row in records}
        type_counts = {}
        for row in records:
            errors = validate_abnormal_record(row)
            if row.get("split") != split:
                errors.append("split_mismatch")
            if row.get("sample_id") in seen_ids:
                errors.append("duplicate_sample_id")
            seen_ids.add(row.get("sample_id"))
            if errors:
                issues.append({"sample_id": row.get("sample_id"), "errors": errors})
            type_counts[row["abnormal_type"]] = type_counts.get(row["abnormal_type"], 0) + 1
        conversation_ids = [row.get("metadata", {}).get("sample_id") for row in conversations]
        if [row["sample_id"] for row in records] != conversation_ids:
            issues.append({"split": split, "errors": ["standardized_conversation_mismatch"]})
        distribution[split] = {"count": len(records), "types": dict(sorted(type_counts.items()))}
    for left, right in (("train", "validation"), ("train", "test_supplement"), ("validation", "test_supplement")):
        overlap = questions_by_split[left] & questions_by_split[right]
        if overlap:
            issues.append({"splits": [left, right], "errors": ["exact_question_leakage"], "count": len(overlap)})
    base_test = Path(manifest["base_manifest"]).parent / "test.standardized.jsonl"
    current_hash = hashlib.sha256(base_test.read_bytes()).hexdigest()
    if current_hash != manifest.get("base_test_sha256"):
        issues.append({"errors": ["base_test_changed_since_generation"]})
    return {"schema_version": "surgical-abnormal-validation/v1", "valid": not issues, "issues": issues, "distribution": distribution, "sample_count": len(seen_ids), "base_test_unchanged": current_hash == manifest.get("base_test_sha256")}
