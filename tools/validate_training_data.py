"""Validate standardized and conversation exports before model training."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.qa_pipeline import validate_record  # noqa: E402


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                yield line_number, json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: JSON解析失败: {exc}") from exc


def validate_directory(input_dir: str | Path) -> dict:
    root = Path(input_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少manifest.json：{root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    issues = []
    sample_ids = set()
    video_splits = {}
    report_splits = {}
    for split in ("train", "validation", "test"):
        standard_path = root / f"{split}.standardized.jsonl"
        conversation_path = root / f"{split}.conversations.jsonl"
        type_counts = Counter()
        answerable = Counter()
        split_ids = []
        split_videos = set()
        for line, record in _read_jsonl(standard_path):
            errors = validate_record(record)
            if record.get("split") != split:
                errors.append("split_field_mismatch")
            sample_id = record.get("sample_id")
            if sample_id in sample_ids:
                errors.append("duplicate_sample_id_across_files")
            sample_ids.add(sample_id)
            split_ids.append(sample_id)
            split_videos.add(record.get("video_id"))
            type_counts[record.get("question_type")] += 1
            answerable[str(record.get("answerable")).lower()] += 1
            if errors:
                issues.append({
                    "file": str(standard_path), "line": line,
                    "sample_id": sample_id, "errors": errors,
                })
        conversation_ids = []
        for line, conversation in _read_jsonl(conversation_path):
            messages = conversation.get("messages") or []
            roles = [message.get("role") for message in messages]
            errors = []
            if roles != ["system", "user", "assistant", "tool", "assistant"]:
                errors.append("invalid_message_role_sequence")
            metadata = conversation.get("metadata") or {}
            conversation_ids.append(metadata.get("sample_id"))
            if len(messages) == 5:
                calls = messages[2].get("tool_calls") or []
                if len(calls) != 1:
                    errors.append("expected_one_tool_call")
                else:
                    try:
                        arguments = json.loads(calls[0]["function"]["arguments"])
                        if not isinstance(arguments, dict):
                            errors.append("tool_arguments_not_object")
                    except (KeyError, TypeError, json.JSONDecodeError):
                        errors.append("tool_arguments_invalid_json")
                    if messages[3].get("tool_call_id") != calls[0].get("id"):
                        errors.append("tool_call_id_mismatch")
                try:
                    tool_result = json.loads(messages[3].get("content", ""))
                    if tool_result.get("source", {}).get("type") != "human_annotation":
                        errors.append("conversation_non_human_source")
                except (AttributeError, json.JSONDecodeError):
                    errors.append("tool_result_invalid_json")
                if not messages[4].get("content"):
                    errors.append("empty_final_assistant_answer")
            if errors:
                issues.append({
                    "file": str(conversation_path), "line": line,
                    "sample_id": metadata.get("sample_id"), "errors": errors,
                })
        if split_ids != conversation_ids:
            issues.append({
                "split": split,
                "errors": ["standardized_conversation_order_or_count_mismatch"],
            })
        video_splits[split] = split_videos
        report_splits[split] = {
            "sample_count": len(split_ids),
            "video_count": len(split_videos),
            "question_type_counts": dict(sorted(type_counts.items())),
            "answerable_counts": dict(sorted(answerable.items())),
        }
    names = list(video_splits)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = sorted(video_splits[left] & video_splits[right])
            if overlap:
                issues.append({
                    "errors": ["video_split_leakage"],
                    "splits": [left, right], "videos": overlap,
                })
    expected = manifest.get("outputs", {})
    for split, summary in report_splits.items():
        if expected.get(split, {}).get("sample_count") != summary["sample_count"]:
            issues.append({
                "split": split, "errors": ["manifest_sample_count_mismatch"]
            })
    return {
        "schema_version": "surgical-training-data-validation/v1",
        "input_dir": str(root),
        "valid": not issues,
        "total_samples": len(sample_ids),
        "splits": report_splits,
        "checks": {
            "json_parse": True,
            "required_fields": True,
            "tool_time_consistency": True,
            "human_annotation_source": True,
            "conversation_structure": True,
            "video_split_disjoint": True,
            "sample_ids_unique": True,
        },
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = validate_directory(args.input_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        path = Path(args.output)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

