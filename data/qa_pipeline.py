"""Generate tool-calling QA data from CholecT50 human annotations."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

from data.cholect50 import (
    CholecT50Reader,
    FrameAnnotation,
    TripletInstance,
    VideoAnnotation,
    get_video_split,
)


SCHEMA_VERSION = "surgical-tool-qa/v1"
SYSTEM_MESSAGE = (
    "你是手术视频问答助手。必须先根据问题调用给定工具，再严格依据工具返回回答；"
    "不得用常识补充工具中没有的当前视频事实。"
)


POINT_QUESTIONS = {
    "instrument": "{video}第{time:g}秒的人工标注中出现了哪些器械？",
    "action": "{video}第{time:g}秒的人工标注中出现了哪些动作？",
    "target": "{video}第{time:g}秒的人工标注中涉及哪些靶点？",
    "triplet": "{video}第{time:g}秒的完整操作三元组是什么？",
    "time_point_operation": "查询{video}第{time:g}秒发生了哪些有标注依据的操作。",
}


def _unique_triplets(instances: Sequence[TripletInstance]) -> List[dict]:
    values = {}
    for item in instances:
        values.setdefault(item.triplet_id, {
            "id": item.triplet_id,
            "instrument": item.instrument,
            "verb": item.verb,
            "target": item.target,
            "confidence": 1.0,
            "instance_count": 0,
        })["instance_count"] += 1
    return [values[key] for key in sorted(values)]


def _point_tool_result(frame: FrameAnnotation) -> dict:
    observations = _unique_triplets(frame.triplets)
    return {
        "ok": True,
        "status": (
            "human_annotation_available"
            if observations else "annotation_has_no_positive_triplet"
        ),
        "video_id": frame.video_id,
        "requested_time": frame.timestamp,
        "actual_frame_time": frame.timestamp,
        "frame_id": frame.frame_id,
        "supplemented": False,
        # Same contract field as deployment. Source metadata prevents these
        # labels from being confused with RDV predictions.
        "predictions": deepcopy(observations),
        "observations": deepcopy(observations),
        "best_candidate": deepcopy(observations[0]) if observations else None,
        "source": {
            "type": "human_annotation",
            "dataset": "CholecT50",
            "dataset_version": "2.0",
            "is_ground_truth": True,
            "annotation_file": str(frame.annotation_path),
            "annotation_fps": frame.fps,
        },
        "evidence": {
            "video_id": frame.video_id,
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "frame_path": str(frame.frame_path),
            "raw_instances": [item.to_dict() for item in frame.triplets],
        },
        "warning": (
            "没有正三元组标签不等于能够断言画面中不存在操作。"
            if not observations else None
        ),
    }


def _entities(triplets: Sequence[dict], field: str) -> List[str]:
    return sorted({str(item[field]) for item in triplets if item.get(field)})


def _format_list(values: Sequence[str]) -> str:
    return "、".join(values)


def _point_answer(question_type: str, tool_result: dict) -> Tuple[str, dict, bool]:
    triplets = tool_result["observations"]
    if not triplets:
        return (
            "该帧人工标注中没有正三元组标签，因此证据不足，无法据此断言没有操作。",
            {"status": "insufficient_annotation", "entities": [], "triplets": []},
            False,
        )
    if question_type == "instrument":
        values = _entities(triplets, "instrument")
        answer = f"人工标注显示器械为：{_format_list(values)}。"
        structured = {"status": "answerable", "entities": values}
    elif question_type == "action":
        values = _entities(triplets, "verb")
        answer = f"人工标注显示动作为：{_format_list(values)}。"
        structured = {"status": "answerable", "entities": values}
    elif question_type == "target":
        values = _entities(triplets, "target")
        answer = f"人工标注显示靶点为：{_format_list(values)}。"
        structured = {"status": "answerable", "entities": values}
    else:
        values = [
            f"{item['instrument']} - {item['verb']} - {item['target']}"
            for item in triplets
        ]
        answer = f"人工标注中的操作三元组为：{_format_list(values)}。"
        structured = {
            "status": "answerable",
            "triplets": [
                {key: item[key] for key in ("id", "instrument", "verb", "target")}
                for item in triplets
            ],
        }
    return answer, structured, True


def _sample_id(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _standard_record(
    *, question: str, question_type: str, tool_name: str, arguments: dict,
    tool_result: dict, answer: str, structured_answer: dict,
    video_id: str, time_range: Sequence[float], split: str, answerable: bool,
) -> dict:
    identity = {
        "question_type": question_type,
        "video_id": video_id,
        "time_range": list(time_range),
        "tool_name": tool_name,
        "arguments": arguments,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": _sample_id(identity),
        "user_question": question,
        "correct_tool_call": {"name": tool_name, "arguments": arguments},
        "tool_result": tool_result,
        "final_answer": answer,
        "structured_answer": structured_answer,
        "video_id": video_id,
        "time_range": list(time_range),
        "evidence_source": {
            "type": "human_annotation",
            "dataset": "CholecT50",
            "dataset_version": "2.0",
            "is_ground_truth": True,
        },
        "question_type": question_type,
        "answerable": answerable,
        "split": split,
        "generation_method": "verified_template",
    }


def _point_records(frame: FrameAnnotation, split: str) -> Iterator[dict]:
    tool_result = _point_tool_result(frame)
    arguments = {"video_id": frame.video_id, "time": frame.timestamp}
    for question_type, template in POINT_QUESTIONS.items():
        answer, structured, answerable = _point_answer(question_type, tool_result)
        yield _standard_record(
            question=template.format(video=frame.video_id, time=frame.timestamp),
            question_type=question_type,
            tool_name="query_video_time",
            arguments=deepcopy(arguments),
            tool_result=deepcopy(tool_result),
            answer=answer,
            structured_answer=structured,
            video_id=frame.video_id,
            time_range=(frame.timestamp, frame.timestamp),
            split=split,
            answerable=answerable,
        )


def select_stratified_point_frames(
    video: VideoAnnotation,
    max_frames: int,
    max_per_signature: int = 2,
) -> List[FrameAnnotation]:
    """Prefer rare label combinations and avoid repeated adjacent labels."""
    buckets: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for frame_id, rows in ((int(key), value) for key, value in video.annotations.items()):
        signature = tuple(sorted({int(row[0]) for row in rows if int(row[0]) >= 0}))
        buckets[signature].append(frame_id)
    selected = []
    # Rare signatures first, then deterministic label order.
    for signature, frame_ids in sorted(buckets.items(), key=lambda item: (len(item[1]), item[0])):
        positions = [len(frame_ids) // 2]
        if max_per_signature > 1 and len(frame_ids) > 1:
            positions.append(0)
        for position in positions[:max_per_signature]:
            selected.append(frame_ids[position])
            if len(selected) >= max_frames:
                return [video.frame(frame_id) for frame_id in sorted(set(selected))]
    return [video.frame(frame_id) for frame_id in sorted(set(selected))]


def build_continuous_events(video: VideoAnnotation) -> List[dict]:
    """Merge a triplet only across consecutive annotated frames.

    No gap filling or phase inference is performed. Event boundaries are the
    first/last 1 FPS human-annotated frames carrying that triplet.
    """
    active: Dict[int, dict] = {}
    events: List[dict] = []
    for frame_id in range(video.num_frames):
        frame = video.frame(frame_id)
        present = {item.triplet_id: item for item in frame.triplets}
        for triplet_id in list(active):
            if triplet_id not in present:
                events.append(active.pop(triplet_id))
        for triplet_id, item in present.items():
            if triplet_id not in active:
                active[triplet_id] = {
                    "triplet_id": triplet_id,
                    "instrument": item.instrument,
                    "verb": item.verb,
                    "target": item.target,
                    "start_frame": frame_id,
                    "end_frame": frame_id,
                    "start_time": frame.timestamp,
                    "end_time": frame.timestamp,
                    "sample_count": 1,
                }
            else:
                event = active[triplet_id]
                event["end_frame"] = frame_id
                event["end_time"] = frame.timestamp
                event["sample_count"] += 1
    events.extend(active.values())
    return sorted(events, key=lambda item: (item["start_frame"], item["triplet_id"]))


def _range_tool_result(
    video: VideoAnnotation,
    events: Sequence[dict],
    start_time: float,
    end_time: float,
) -> dict:
    start_frame = max(0, int(round(start_time * video.fps)))
    end_frame = min(video.num_frames - 1, int(round(end_time * video.fps)))
    samples = []
    unique = {}
    for frame_id in range(start_frame, end_frame + 1):
        frame = video.frame(frame_id)
        observations = _unique_triplets(frame.triplets)
        samples.append({
            "requested_time": frame.timestamp,
            "actual_time": frame.timestamp,
            "frame_id": frame.frame_id,
            "observation_status": (
                "human_annotation_available"
                if observations else "annotation_has_no_positive_triplet"
            ),
            "predictions": deepcopy(observations),
            "observations": deepcopy(observations),
            "source": {"type": "human_annotation", "is_ground_truth": True},
        })
        for item in observations:
            unique[item["id"]] = deepcopy(item)
    clipped_events = []
    for event in events:
        if event["end_time"] < start_time or event["start_time"] > end_time:
            continue
        clipped = deepcopy(event)
        clipped["original_start_time"] = event["start_time"]
        clipped["original_end_time"] = event["end_time"]
        clipped["start_time"] = max(event["start_time"], start_time)
        clipped["end_time"] = min(event["end_time"], end_time)
        clipped["left_truncated"] = event["start_time"] < start_time
        clipped["right_truncated"] = event["end_time"] > end_time
        clipped["boundary_rule"] = "consecutive_positive_frames_at_1fps"
        clipped_events.append(clipped)
    return {
        "ok": True,
        "status": "human_annotation_available",
        "video_id": video.video_id,
        "start_time": start_time,
        "end_time": end_time,
        "samples": samples,
        "events": clipped_events,
        "observed_triplets": [unique[key] for key in sorted(unique)],
        "source": {
            "type": "human_annotation",
            "dataset": "CholecT50",
            "dataset_version": video.dataset_version,
            "annotation_file": str(video.annotation_path),
            "annotation_fps": video.fps,
            "is_ground_truth": True,
        },
        "event_rule": {
            "merge": "same triplet on consecutive annotated frames",
            "gap_filling": False,
            "window_clipping_is_marked": True,
        },
    }


def _range_answer(tool_result: dict, include_events: bool) -> Tuple[str, dict, bool]:
    triplets = tool_result["observed_triplets"]
    if not triplets:
        return (
            "该时间范围的人工标注中没有正三元组标签；不能自动推断范围内不存在操作。",
            {"status": "insufficient_annotation", "triplets": [], "events": []},
            False,
        )
    names = [
        f"{item['instrument']} - {item['verb']} - {item['target']}"
        for item in triplets
    ]
    if not include_events:
        return (
            f"人工标注显示该范围出现过：{_format_list(names)}。",
            {"status": "answerable", "triplets": triplets},
            True,
        )
    parts = []
    for event in tool_result["events"]:
        markers = []
        if event["left_truncated"]:
            markers.append("左边界截断")
        if event["right_truncated"]:
            markers.append("右边界截断")
        suffix = f"（{'、'.join(markers)}）" if markers else ""
        parts.append(
            f"{event['instrument']} - {event['verb']} - {event['target']} "
            f"[{event['start_time']:g}, {event['end_time']:g}]秒{suffix}"
        )
    return (
        "按连续1 FPS正标注合并的事件为：" + "；".join(parts) + "。",
        {"status": "answerable", "events": deepcopy(tool_result["events"])},
        True,
    )


def _range_records(
    video: VideoAnnotation,
    events: Sequence[dict],
    split: str,
    windows: Sequence[int],
    stride: int,
    max_ranges: int,
) -> Iterator[dict]:
    duration = (video.num_frames - 1) / video.fps
    starts = list(range(0, int(duration) + 1, stride))[:max_ranges]
    for index, start in enumerate(starts):
        window = windows[index % len(windows)]
        end = min(duration, float(start + window))
        if end <= start:
            continue
        tool_result = _range_tool_result(video, events, float(start), end)
        arguments = {
            "video_id": video.video_id,
            "start_time": float(start),
            "end_time": end,
        }
        for question_type, include_events, question in (
            (
                "time_range_summary", False,
                f"{video.video_id}第{start:g}秒到第{end:g}秒出现过哪些人工标注操作？",
            ),
            (
                "continuous_event", True,
                f"汇总{video.video_id}第{start:g}秒到第{end:g}秒的连续标注事件，并标记截断边界。",
            ),
        ):
            answer, structured, answerable = _range_answer(tool_result, include_events)
            yield _standard_record(
                question=question,
                question_type=question_type,
                tool_name="query_video_range",
                arguments=deepcopy(arguments),
                tool_result=deepcopy(tool_result),
                answer=answer,
                structured_answer=structured,
                video_id=video.video_id,
                time_range=(float(start), end),
                split=split,
                answerable=answerable,
            )


def validate_record(record: dict) -> List[str]:
    errors = []
    required = {
        "sample_id", "user_question", "correct_tool_call", "tool_result",
        "final_answer", "video_id", "time_range", "evidence_source",
        "question_type", "answerable", "split", "structured_answer",
    }
    missing = sorted(required - set(record))
    if missing:
        errors.append("missing_fields:" + ",".join(missing))
        return errors
    call = record["correct_tool_call"]
    result = record["tool_result"]
    if call.get("name") not in ("query_video_time", "query_video_range"):
        errors.append("unsupported_tool")
    if call.get("arguments", {}).get("video_id") != record["video_id"]:
        errors.append("video_id_mismatch")
    if result.get("video_id") != record["video_id"]:
        errors.append("tool_result_video_id_mismatch")
    if result.get("source", {}).get("type") != "human_annotation":
        errors.append("non_human_evidence")
    if record["evidence_source"].get("type") != "human_annotation":
        errors.append("source_marker_mismatch")
    start, end = record["time_range"]
    if start > end:
        errors.append("inverted_time_range")
    args = call["arguments"]
    if call["name"] == "query_video_time":
        if args.get("time") != start or start != end:
            errors.append("point_time_mismatch")
    else:
        if args.get("start_time") != start or args.get("end_time") != end:
            errors.append("range_time_mismatch")
    if record["answerable"] and record["structured_answer"].get("status") != "answerable":
        errors.append("answerability_mismatch")
    return errors


def to_conversation(record: dict) -> dict:
    call_id = "call_" + record["sample_id"]
    call = record["correct_tool_call"]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": record["user_question"]},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(
                            call["arguments"], ensure_ascii=False, sort_keys=True
                        ),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(record["tool_result"], ensure_ascii=False),
            },
            {"role": "assistant", "content": record["final_answer"]},
        ],
        "metadata": {
            key: record[key] for key in (
                "sample_id", "video_id", "time_range", "evidence_source",
                "question_type", "answerable", "split",
            )
        },
    }


def _write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            count += 1
    return count


@dataclass
class GenerationConfig:
    split_variant: str = "rdv_original"
    fold: int = 1
    max_point_frames_per_video: int = 40
    max_per_signature: int = 2
    range_windows: Tuple[int, ...] = (10, 30)
    range_stride: int = 60
    max_ranges_per_video: int = 8
    seed: int = 2026


def generate_dataset(
    dataset_root: str | Path,
    output_dir: str | Path,
    config: GenerationConfig = None,
) -> dict:
    config = config or GenerationConfig()
    random.seed(config.seed)
    reader = CholecT50Reader(dataset_root)
    validation = reader.validate(decode_all=True)
    if not validation["valid"]:
        raise ValueError(f"CholecT50验证失败：{validation['issues'][:5]}")
    splits = get_video_split(config.split_variant, config.fold)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rejected = []
    review = []
    selection_log = []
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_validation": validation,
        "configuration": {
            "split_variant": config.split_variant,
            "fold": config.fold,
            "max_point_frames_per_video": config.max_point_frames_per_video,
            "max_per_signature": config.max_per_signature,
            "range_windows": list(config.range_windows),
            "range_stride": config.range_stride,
            "max_ranges_per_video": config.max_ranges_per_video,
            "seed": config.seed,
        },
        "split_videos": splits,
        "outputs": {},
        "source_policy": (
            "Only CholecT50 Release 2.0 human annotations are used as answers. "
            "RDV predictions are not used as ground truth."
        ),
    }
    seen_ids = set()
    for split, video_ids in splits.items():
        accepted = []
        type_counts = defaultdict(int)
        answerable_counts = defaultdict(int)
        for video_id in video_ids:
            video = reader.read_video(video_id)
            point_frames = select_stratified_point_frames(
                video,
                max_frames=config.max_point_frames_per_video,
                max_per_signature=config.max_per_signature,
            )
            events = build_continuous_events(video)
            selection_log.append({
                "split": split,
                "video_id": video_id,
                "total_frames": video.num_frames,
                "selected_point_frame_ids": [item.frame_id for item in point_frames],
                "selected_point_frame_count": len(point_frames),
                "filtered_point_frame_count": video.num_frames - len(point_frames),
                "filter_reason": (
                    "stratified rare-signature sampling with per-video and "
                    "per-signature caps; continuous repeats are not all emitted"
                ),
                "continuous_event_count": len(events),
            })
            candidates = []
            for frame in point_frames:
                candidates.extend(_point_records(frame, split))
            candidates.extend(_range_records(
                video, events, split,
                windows=config.range_windows,
                stride=config.range_stride,
                max_ranges=config.max_ranges_per_video,
            ))
            for record in candidates:
                if record["sample_id"] in seen_ids:
                    rejected.append({
                        "record": record,
                        "reasons": ["duplicate_sample_id"],
                    })
                    continue
                errors = validate_record(record)
                if errors:
                    rejected.append({"record": record, "reasons": errors})
                    continue
                seen_ids.add(record["sample_id"])
                accepted.append(record)
                if not record["answerable"]:
                    review.append({
                        "sample_id": record["sample_id"],
                        "split": split,
                        "video_id": record["video_id"],
                        "time_range": record["time_range"],
                        "reason": "annotation_has_no_positive_triplet",
                        "recommended_action": (
                            "确认该样本应保留为证据不足/拒答样本；"
                            "不得改写为确定没有操作"
                        ),
                        "record": record,
                    })
                type_counts[record["question_type"]] += 1
                answerable_counts[str(record["answerable"]).lower()] += 1
        standard_path = output / f"{split}.standardized.jsonl"
        conversation_path = output / f"{split}.conversations.jsonl"
        _write_jsonl(standard_path, accepted)
        _write_jsonl(conversation_path, (to_conversation(item) for item in accepted))
        manifest["outputs"][split] = {
            "standardized": str(standard_path),
            "conversations": str(conversation_path),
            "sample_count": len(accepted),
            "question_type_counts": dict(sorted(type_counts.items())),
            "answerable_counts": dict(sorted(answerable_counts.items())),
        }
    rejected_path = output / "rejected.jsonl"
    _write_jsonl(rejected_path, rejected)
    manifest["rejected"] = {
        "path": str(rejected_path),
        "count": len(rejected),
    }
    selection_path = output / "selection_log.jsonl"
    _write_jsonl(selection_path, selection_log)
    manifest["selection_log"] = {
        "path": str(selection_path),
        "video_count": len(selection_log),
        "filtered_point_frame_count": sum(
            item["filtered_point_frame_count"] for item in selection_log
        ),
    }
    review_path = output / "review_queue.jsonl"
    _write_jsonl(review_path, review)
    manifest["review_queue"] = {
        "path": str(review_path),
        "count": len(review),
        "reason": "unanswerable samples are queued for human policy review",
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest

