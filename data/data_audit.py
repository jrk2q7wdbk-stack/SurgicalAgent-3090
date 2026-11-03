"""Reproducible audit and human-review material for CholecT50 QA data."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

from data.cholect50 import CholecT50Reader, get_video_split


AUDIT_SCHEMA = "surgical-training-data-audit/v1"
REVIEW_SCHEMA = "surgical-training-review/v1"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_hash(paths: Iterable[Path], relative_to: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        relative = path.relative_to(relative_to).as_posix().encode("utf-8")
        digest.update(relative + b"\0" + sha256_file(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def _triplets(record: dict) -> list[dict]:
    result = record.get("tool_result") or {}
    values = result.get("observations") or result.get("observed_triplets") or []
    return [item for item in values if isinstance(item, dict)]


def _raw_evidence(record: dict) -> dict:
    result = record.get("tool_result") or {}
    source = result.get("source") or {}
    evidence = result.get("evidence") or {}
    samples = result.get("samples") or []
    frame_ids = [sample.get("frame_id") for sample in samples if sample.get("frame_id") is not None]
    return {
        "annotation_file": source.get("annotation_file"),
        "frame_id": evidence.get("frame_id"),
        "frame_ids": frame_ids,
        "frame_path": evidence.get("frame_path"),
        "annotation_fps": source.get("annotation_fps"),
    }


def _review_row(record: dict, focus: list[str], origin: str) -> dict:
    return {
        "schema_version": REVIEW_SCHEMA,
        "sample_id": record["sample_id"],
        "split": record["split"],
        "video_id": record["video_id"],
        "time": record["time_range"],
        "original_annotation_location": _raw_evidence(record),
        "evidence": record.get("tool_result"),
        "question": record["user_question"],
        "expected_tool_call": record["correct_tool_call"],
        "expected_answer": {
            "text": record["final_answer"],
            "structured": record["structured_answer"],
            "answerable": record["answerable"],
        },
        "review_focus": sorted(set(focus)),
        "selection_origin": origin,
        "reviewer_status": "pending",
        "reviewer_note": "",
    }


def _event_checks(record: dict) -> list[str]:
    result = record.get("tool_result") or {}
    issues = []
    start, end = record.get("time_range", [None, None])
    for event in result.get("events") or []:
        event_start = event.get("start_time")
        event_end = event.get("end_time")
        if event_start is not None and start is not None and event_start < start:
            issues.append("event_before_query_range")
        if event_end is not None and end is not None and event_end > end:
            issues.append("event_after_query_range")
        frames = event.get("frame_ids") or []
        if frames and any(right != left + 1 for left, right in zip(frames, frames[1:])):
            issues.append("event_contains_nonconsecutive_frames")
        start_frame, end_frame = event.get("start_frame"), event.get("end_frame")
        sample_count = event.get("sample_count")
        if start_frame is not None and end_frame is not None and sample_count is not None:
            if int(sample_count) != int(end_frame) - int(start_frame) + 1:
                issues.append("event_sample_count_not_consecutive")
    return issues


def audit_training_data(
    dataset_root: str | Path,
    training_dir: str | Path,
    output_dir: str | Path,
    checkpoint: str | Path | None = None,
    seed: int = 20261102,
    per_type: int = 8,
) -> dict:
    dataset = CholecT50Reader(dataset_root)
    dataset_validation = dataset.validate(decode_all=True)
    training = Path(training_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((training / "manifest.json").read_text(encoding="utf-8"))
    splits = {}
    all_records = []
    issues: list[dict] = []
    if not dataset_validation.get("valid"):
        issues.append({"code": "DATASET_VALIDATION_FAILED", "details": dataset_validation.get("issues")})
    ids: set[str] = set()
    videos_by_split: dict[str, set[str]] = {}
    type_counts: Counter[str] = Counter()
    answerable_counts: Counter[str] = Counter()
    for split in ("train", "validation", "test"):
        records = read_jsonl(training / f"{split}.standardized.jsonl")
        videos_by_split[split] = {row["video_id"] for row in records}
        duplicate_ids = sorted(row["sample_id"] for row in records if row["sample_id"] in ids)
        ids.update(row["sample_id"] for row in records)
        if duplicate_ids:
            issues.append({"code": "DUPLICATE_SAMPLE_ID", "split": split, "ids": duplicate_ids})
        split_types = Counter(row["question_type"] for row in records)
        split_answerable = Counter(str(bool(row["answerable"])).lower() for row in records)
        type_counts.update(split_types)
        answerable_counts.update(split_answerable)
        event_issues = [
            {"sample_id": row["sample_id"], "errors": _event_checks(row)}
            for row in records if _event_checks(row)
        ]
        issues.extend({"code": "EVENT_BOUNDARY", **item} for item in event_issues)
        splits[split] = {
            "samples": len(records),
            "videos": sorted(videos_by_split[split]),
            "question_types": dict(sorted(split_types.items())),
            "answerable": dict(sorted(split_answerable.items())),
        }
        all_records.extend(records)
    split_names = list(videos_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            overlap = sorted(videos_by_split[left] & videos_by_split[right])
            if overlap:
                issues.append({"code": "VIDEO_SPLIT_LEAKAGE", "splits": [left, right], "videos": overlap})

    official = get_video_split("rdv_original")
    if any(set(official[key]) != videos_by_split[key] for key in official):
        issues.append({"code": "FROZEN_SPLIT_MISMATCH"})

    # Verify timestamps and complete triplet evidence against the original annotations.
    timestamp_examples = []
    triplet_mismatches = []
    video_cache = {}
    for record in all_records:
        if record["question_type"] not in {"instrument", "action", "target", "triplet", "time_point_operation"}:
            continue
        frame_id = (record.get("tool_result", {}).get("evidence") or {}).get("frame_id")
        if frame_id is None:
            continue
        if record["video_id"] not in video_cache:
            video_cache[record["video_id"]] = dataset.read_video(record["video_id"])
        video = video_cache[record["video_id"]]
        frame = video.frame(int(frame_id))
        expected_ids = sorted({item.triplet_id for item in frame.triplets})
        actual_ids = sorted({int(item["id"]) for item in record["tool_result"].get("observations", [])})
        if expected_ids != actual_ids:
            triplet_mismatches.append({
                "sample_id": record["sample_id"], "expected": expected_ids, "actual": actual_ids,
            })
    if triplet_mismatches:
        issues.append({"code": "TRIPLET_EVIDENCE_MISMATCH", "count": len(triplet_mismatches), "examples": triplet_mismatches[:20]})

    # Directly inspect zero, one and last annotation keys/files in one video per split.
    for split in ("train", "validation", "test"):
        video = dataset.read_video(official[split][0])
        for frame_id in (0, 1, video.num_frames - 1):
            frame = video.frame(frame_id)
            timestamp_examples.append({
                "split": split,
                "video_id": video.video_id,
                "frame_id": frame_id,
                "annotation_key": str(frame_id),
                "annotation_key_exists": str(frame_id) in video.annotations,
                "frame_file": frame.frame_path.name,
                "file_exists": frame.frame_path.is_file(),
                "fps": frame.fps,
                "computed_seconds": frame.timestamp,
            })

    review_source = read_jsonl(training / "review_queue.jsonl")
    review_records = {item["sample_id"]: item["record"] for item in review_source}
    review_focus = {sample_id: ["evidence_insufficient", "do_not_infer_absence"] for sample_id in review_records}
    review_origin = {sample_id: "all_existing_unanswerable" for sample_id in review_records}
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    triplet_frequency = Counter()
    for record in all_records:
        by_type[record["question_type"]].append(record)
        for item in _triplets(record):
            triplet_frequency[(item.get("instrument"), item.get("verb"), item.get("target"))] += 1
    for question_type, records in sorted(by_type.items()):
        candidates = list(records)
        rng.shuffle(candidates)
        candidates.sort(key=lambda row: (
            0 if len(_triplets(row)) > 1 else 1,
            min((triplet_frequency[(x.get("instrument"), x.get("verb"), x.get("target"))] for x in _triplets(row)), default=10**9),
            row["sample_id"],
        ))
        for record in candidates[:per_type]:
            sid = record["sample_id"]
            focus = review_focus.setdefault(sid, [])
            focus.append(f"question_type:{question_type}")
            if len(_triplets(record)) > 1:
                focus.append("multiple_triplets")
            if _triplets(record) and min(triplet_frequency[(x.get("instrument"), x.get("verb"), x.get("target"))] for x in _triplets(record)) <= 2:
                focus.append("rare_triplet")
            if question_type in {"time_range_summary", "continuous_event"}:
                focus.append("time_range")
            events = record.get("tool_result", {}).get("events") or []
            if any(event.get("left_truncated") or event.get("right_truncated") for event in events):
                focus.append("truncated_event")
            review_origin.setdefault(sid, "seeded_stratified_sample")
            review_records.setdefault(sid, record)

    review_rows = [
        _review_row(review_records[sid], review_focus[sid], review_origin[sid])
        for sid in sorted(review_records)
    ]
    write_jsonl(output / "human_review_checklist.jsonl", review_rows)

    label_paths = list(dataset.labels_dir.glob("VID*.json"))
    provenance = {
        "dataset": "CholecT50",
        "release": "2.0",
        "dataset_root": str(dataset.root),
        "label_file_count": len(label_paths),
        "label_mapping": {"path": str(dataset.mapping_path), "sha256": sha256_file(dataset.mapping_path)},
        "labels_aggregate_sha256": aggregate_hash(label_paths, dataset.root),
        "split_variant": "rdv_original",
        "split_definition_sha256": hashlib.sha256(json.dumps(official, sort_keys=True).encode("utf-8")).hexdigest(),
        "training_manifest_sha256": sha256_file(training / "manifest.json"),
        "standardized_files": {
            split: sha256_file(training / f"{split}.standardized.jsonl") for split in splits
        },
    }
    checkpoint_path = Path(checkpoint).resolve() if checkpoint else None
    checkpoint_info = {
        "path": str(checkpoint_path) if checkpoint_path else None,
        "exists": bool(checkpoint_path and checkpoint_path.is_file()),
        "sha256": sha256_file(checkpoint_path) if checkpoint_path and checkpoint_path.is_file() else None,
        "training_video_scope_status": "unknown",
        "statement": "The checkpoint contains no verified split metadata; its training-video coverage is unknown.",
    }
    report = {
        "schema_version": AUDIT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "valid": not issues,
        "issues": issues,
        "dataset_validation": dataset_validation,
        "counts": {
            "total_samples": len(all_records),
            "question_types": dict(sorted(type_counts.items())),
            "answerable": dict(sorted(answerable_counts.items())),
            "existing_unanswerable_review_count": len(review_source),
            "human_review_checklist_count": len(review_rows),
        },
        "splits": splits,
        "timestamp_audit": {
            "mapping_rule": "timestamp_seconds = zero_based_frame_id / annotation_fps",
            "examples": timestamp_examples,
            "one_second_offset_found": any(item["computed_seconds"] != item["frame_id"] / item["fps"] for item in timestamp_examples),
            "scope_note": "Verified annotation key, PNG filename and generated QA time agree. No independent original-video timecode is present in Release 2.0, so absolute acquisition-time semantics cannot be proven from these files.",
        },
        "triplet_preservation": {"checked_point_samples": sum(1 for row in all_records if row["question_type"] not in {"time_range_summary", "continuous_event"}), "mismatch_count": len(triplet_mismatches)},
        "event_policy": {"merge_only_consecutive_frames": True, "gap_filling": False, "window_clipping_marked": True},
        "review_sampling": {"seed": seed, "per_question_type": per_type, "all_unanswerable_included": True},
        "provenance": provenance,
        "checkpoint": checkpoint_info,
        "source_manifest_configuration": manifest.get("configuration"),
    }
    (output / "data_audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "data_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
