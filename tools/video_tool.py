"""Persistent, timestamp-preserving RDV video analysis tools.

All observations in this module are model predictions. Event groups summarize
sampled observations and are not ground-truth phases or exact boundaries.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, Iterable, List, Optional

import cv2

from agent.config import get_config
from agent.confidence import classify_confidence
from agent.temporal_memory import TemporalMemory
from tools.vision_tool import analyze_surgical_image


SCHEMA_VERSION = "surgical-video-analysis/v2"
_VISION_CONFIG = get_config().section("vision")
_temporal_memory = TemporalMemory(
    max_events=1000,
    confidence_threshold=float(_VISION_CONFIG.get("triplet_threshold", 0.3)),
)
_RESULTS: Dict[str, dict] = {}
_CURRENT_VIDEO_ID: Optional[str] = None
_FILE_HASH_CACHE: Dict[tuple, str] = {}


def _error(code: str, message: str, **details: Any) -> dict:
    return {
        "ok": False,
        "status": "error",
        "error": {"code": code, "message": message, "details": details},
    }


def _as_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是数字")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数字") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数字")
    return number


def _sha256(path: Path) -> str:
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key in _FILE_HASH_CACHE:
        return _FILE_HASH_CACHE[key]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    _FILE_HASH_CACHE[key] = value
    return value


def _video_metadata(video_path: str) -> dict:
    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到视频：{path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if fps <= 0 or total_frames <= 0:
        raise RuntimeError(
            f"视频元数据无效：fps={fps}, total_frames={total_frames}"
        )
    content_sha256 = _sha256(path)
    return {
        "video_id": f"sha256:{content_sha256}",
        "path": str(path),
        "name": path.name,
        "content_sha256": content_sha256,
        "size_bytes": path.stat().st_size,
        "fps": fps,
        "total_frames": total_frames,
        "duration": total_frames / fps,
        "last_frame_time": (total_frames - 1) / fps,
    }


def _model_identity() -> dict:
    config = get_config()
    vision = config.section("vision")
    checkpoint = config.path("model_checkpoint")
    return {
        "family": vision.get("model_family", "RDV"),
        "version": vision.get("model_version", checkpoint.stem),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "device": vision.get("device", "cpu"),
        "component_threshold": float(vision.get("component_threshold", 0.5)),
        "triplet_threshold": float(vision.get("triplet_threshold", 0.3)),
        "top_k_triplets": int(vision.get("top_k_triplets", 5)),
    }


def _cache_key(video: dict, model: dict, analysis: dict) -> str:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "video_sha256": video["content_sha256"],
        "model": model,
        "analysis": {
            "start_time": analysis["start_time"],
            "end_time": analysis["end_time"],
            "sample_interval": analysis["sample_interval"],
            "max_frames": analysis.get("max_frames"),
        },
    }
    encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_path(cache_key: str) -> Path:
    path = get_config().path("result_cache") / "videos"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{cache_key}.json"


def _write_result(result: dict) -> None:
    path = Path(result["persistence"]["cache_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def _load_path(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"不支持的缓存格式：{result.get('schema_version')}")
    return result


def _sample_status(vision_result: dict) -> str:
    retained = vision_result.get("triplets") or []
    if retained:
        confidence = float(retained[0].get("confidence", 0.0))
        return (
            "recognized_prediction"
            if classify_confidence(confidence) != "low"
            else "uncertain_prediction"
        )
    if vision_result.get("best_triplet"):
        return "below_threshold_candidate"
    return "model_no_recognition"


def _make_sample(
    vision_result: dict,
    requested_time: float,
    actual_time: float,
    frame_id: int,
    sampling_kind: str,
) -> dict:
    return {
        "requested_time": round(float(requested_time), 6),
        "actual_time": round(float(actual_time), 6),
        "frame_id": int(frame_id),
        "sampling_kind": sampling_kind,
        "timestamp_basis": "original_video_frame_index_divided_by_fps",
        "observation_status": _sample_status(vision_result),
        "predictions": deepcopy(vision_result.get("triplets") or []),
        "best_candidate": deepcopy(vision_result.get("best_triplet")),
        "components": {
            "instruments": deepcopy(vision_result.get("instruments") or []),
            "verbs": deepcopy(vision_result.get("verbs") or []),
            "targets": deepcopy(vision_result.get("targets") or []),
        },
        "source": {"type": "rdv_prediction", "is_ground_truth": False},
    }


def _predict_requested_time(
    cap: cv2.VideoCapture,
    metadata: dict,
    requested_time: float,
    sampling_kind: str,
    temp_dir: Path,
) -> dict:
    frame_id = min(
        int(round(requested_time * metadata["fps"])),
        metadata["total_frames"] - 1,
    )
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(
            f"无法读取请求时间 {requested_time:.6f}s 对应帧 {frame_id}"
        )
    actual_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    if actual_frame < 0:
        actual_frame = frame_id
    actual_time = actual_frame / metadata["fps"]
    frame_path = temp_dir / f"frame_{actual_frame:010d}.jpg"
    written, encoded = cv2.imencode(".jpg", frame)
    if not written:
        raise RuntimeError(f"无法编码视频帧：{actual_frame}")
    frame_path.write_bytes(encoded.tobytes())
    vision_result = analyze_surgical_image(str(frame_path))
    return _make_sample(
        vision_result, requested_time, actual_time, actual_frame, sampling_kind
    )


def _event_signature(sample: dict) -> tuple:
    best = sample.get("best_candidate") or {}
    return (
        sample.get("observation_status"), best.get("instrument"),
        best.get("verb"), best.get("target"),
    )


def _build_events(samples: Iterable[dict]) -> List[dict]:
    events: List[dict] = []
    for sample in sorted(samples, key=lambda item: item["actual_time"]):
        signature = _event_signature(sample)
        if not events or events[-1]["_signature"] != signature:
            best = sample.get("best_candidate") or {}
            events.append({
                "_signature": signature,
                "event_type": "predicted_observation_group",
                "observation_status": sample["observation_status"],
                "start_time": sample["actual_time"],
                "end_time": sample["actual_time"],
                "prediction": ({
                    "instrument": best.get("instrument"),
                    "verb": best.get("verb"),
                    "target": best.get("target"),
                } if best else None),
                "scores": [],
                "sample_evidence": [],
                "source": {
                    "type": "rdv_prediction",
                    "is_ground_truth": False,
                    "boundary_basis": "sampled_observations_only",
                },
                "warning": (
                    "这是预测采样点的汇总，不是真实手术阶段，"
                    "边界也不是逐帧确认的动作边界。"
                ),
            })
        event = events[-1]
        event["end_time"] = sample["actual_time"]
        best = sample.get("best_candidate") or {}
        if best.get("confidence") is not None:
            event["scores"].append(float(best["confidence"]))
        event["sample_evidence"].append({
            "requested_time": sample["requested_time"],
            "actual_time": sample["actual_time"],
            "frame_id": sample["frame_id"],
            "sampling_kind": sample["sampling_kind"],
            "observation_status": sample["observation_status"],
            "predictions": deepcopy(sample.get("predictions") or []),
        })
    for event in events:
        event["mean_score"] = (
            sum(event["scores"]) / len(event["scores"])
            if event["scores"] else None
        )
        event.pop("_signature", None)
    return events


def _sync_legacy_memory(result: dict) -> None:
    _temporal_memory.clear()
    for sample in result.get("samples", []):
        _temporal_memory.add_prediction(
            timestamp=sample["actual_time"],
            frame_id=sample["frame_id"],
            vision_result={
                "triplets": deepcopy(sample.get("predictions") or []),
                "best_triplet": deepcopy(sample.get("best_candidate")),
            },
        )
    _temporal_memory.finalize_pending()


def _register_result(result: dict) -> dict:
    global _CURRENT_VIDEO_ID
    video_id = result["video"]["video_id"]
    _RESULTS[video_id] = result
    _CURRENT_VIDEO_ID = video_id
    _sync_legacy_memory(result)
    return result


def _analysis_times(start: float, end: float, interval: float) -> List[float]:
    values = []
    index = 0
    while start + index * interval <= end + 1e-9:
        values.append(min(start + index * interval, end))
        index += 1
    if not values or abs(values[-1] - end) > 1e-6:
        values.append(end)
    return values


def analyze_surgical_video(
    video_path: str,
    sample_interval_seconds: float = None,
    max_frames: int = None,
    start_time: float = 0.0,
    end_time: float = None,
    force: bool = False,
) -> dict:
    """Analyze an absolute time range and persist all retained predictions."""
    try:
        metadata = _video_metadata(video_path)
        video_config = get_config().section("video")
        interval = _as_number(
            sample_interval_seconds
            if sample_interval_seconds is not None
            else video_config.get("sample_interval", 2.0),
            "sample_interval",
        )
        start = _as_number(start_time, "start_time")
        end = metadata["last_frame_time"] if end_time is None else _as_number(
            end_time, "end_time"
        )
        if interval <= 0:
            raise ValueError("sample_interval 必须大于 0")
        if start < 0 or end < 0:
            raise ValueError("start_time 和 end_time 不能为负数")
        if end < start:
            raise ValueError("end_time 不能早于 start_time")
        if start > metadata["last_frame_time"] + 1e-9:
            raise ValueError(
                f"start_time 超出视频范围，末帧时间为 {metadata['last_frame_time']:.6f}s"
            )
        if end > metadata["last_frame_time"] + 1e-9:
            raise ValueError(
                f"end_time 超出视频范围，末帧时间为 {metadata['last_frame_time']:.6f}s"
            )
        if max_frames is not None:
            if isinstance(max_frames, bool) or int(max_frames) <= 0:
                raise ValueError("max_frames 必须是正整数或 null")
            max_frames = int(max_frames)

        model = _model_identity()
        analysis = {
            "start_time": start,
            "end_time": end,
            "sample_interval": interval,
            "max_frames": max_frames,
            "timestamp_scope": "absolute_time_in_original_video",
        }
        key = _cache_key(metadata, model, analysis)
        path = _cache_path(key)
        if path.is_file() and not force:
            cached = _load_path(path)
            cached["cache"] = {"hit": True, "key": key}
            return deepcopy(_register_result(cached))

        times = _analysis_times(start, end, interval)
        if max_frames is not None:
            times = times[:max_frames]
        temp_root = get_config().path("result_cache") / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="frames_", dir=str(temp_root)))
        cap = cv2.VideoCapture(metadata["path"])
        samples = []
        try:
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频：{metadata['path']}")
            for timestamp in times:
                samples.append(_predict_requested_time(
                    cap, metadata, timestamp, "scheduled", temp_dir
                ))
        finally:
            cap.release()
            shutil.rmtree(temp_dir, ignore_errors=True)

        events = _build_events(samples)
        result = {
            "ok": True,
            "status": "completed",
            "schema_version": SCHEMA_VERSION,
            "video": metadata,
            "model": model,
            "analysis": analysis,
            "samples": samples,
            "events": events,
            "event_sequence": deepcopy(events),
            "source": {"type": "rdv_prediction", "is_ground_truth": False},
            "cache": {"hit": False, "key": key},
            "persistence": {"cache_path": str(path)},
            "warnings": [
                "模型未识别到不等于确定没有操作。",
                "事件是预测采样点汇总，不是真实手术阶段。",
            ],
        }
        _write_result(result)
        return deepcopy(_register_result(result))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return _error("VIDEO_ANALYSIS_INVALID", str(exc))


def load_video_result(identifier: str = None) -> dict:
    """Reload a persisted analysis by video id, cache key, or cache path."""
    identifier = identifier or _CURRENT_VIDEO_ID
    if not identifier:
        return _error("VIDEO_NOT_SELECTED", "没有当前视频分析结果")
    if identifier in _RESULTS:
        return deepcopy(_register_result(_RESULTS[identifier]))
    candidate = Path(str(identifier))
    paths: List[Path] = []
    if candidate.is_file():
        paths.append(candidate)
    cache_dir = get_config().path("result_cache") / "videos"
    direct = cache_dir / f"{identifier}.json"
    if direct.is_file():
        paths.append(direct)
    if cache_dir.is_dir():
        paths.extend(sorted(
            cache_dir.glob("*.json"),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        ))
    checked = set()
    for path in paths:
        if path in checked:
            continue
        checked.add(path)
        try:
            result = _load_path(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if identifier in (
            result.get("video", {}).get("video_id"),
            result.get("cache", {}).get("key"), str(path),
        ):
            return deepcopy(_register_result(result))
    return _error("VIDEO_RESULT_NOT_FOUND", f"找不到视频分析结果：{identifier}")


def _validate_access_time(result: dict, timestamp: float, available_until: Any) -> float:
    value = _as_number(timestamp, "time")
    last = float(result["video"]["last_frame_time"])
    if value < 0 or value > last + 1e-9:
        raise ValueError(f"查询时间超出范围 0～{last:.6f}s")
    if available_until is not None:
        boundary = _as_number(available_until, "available_until")
        if boundary < 0 or boundary > last + 1e-9:
            raise ValueError(f"available_until 超出范围 0～{last:.6f}s")
        if value > boundary + 1e-9:
            raise PermissionError(
                f"模拟实时模式只能访问 {boundary:.6f}s 及以前的信息"
            )
    return value


def _add_exact_sample(result: dict, requested_time: float) -> dict:
    metadata = result["video"]
    temp_root = get_config().path("result_cache") / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="exact_", dir=str(temp_root)))
    cap = cv2.VideoCapture(metadata["path"])
    try:
        sample = _predict_requested_time(
            cap, metadata, requested_time, "exact_query", temp_dir
        )
    finally:
        cap.release()
        shutil.rmtree(temp_dir, ignore_errors=True)
    result["samples"].append(sample)
    result["samples"].sort(key=lambda item: (item["actual_time"], item["requested_time"]))
    result["events"] = _build_events(result["samples"])
    result["event_sequence"] = deepcopy(result["events"])
    _write_result(result)
    _register_result(result)
    return sample


def query_video_time(
    video_id: str,
    time: float,
    available_until: float = None,
    supplement_if_missing: bool = True,
) -> dict:
    result = load_video_result(video_id)
    if not result.get("ok"):
        return result
    try:
        requested = _validate_access_time(result, time, available_until)
        matches = [
            item for item in result["samples"]
            if abs(float(item["requested_time"]) - requested) <= 1e-6
        ]
        supplemented = False
        if not matches and supplement_if_missing:
            matches = [_add_exact_sample(result, requested)]
            supplemented = True
        if not matches:
            return _error(
                "EXACT_SAMPLE_MISSING",
                "该时刻没有采样证据，且未启用补充识别",
                requested_time=requested,
            )
        sample = min(matches, key=lambda item: abs(item["actual_time"] - requested))
        return {
            "ok": True,
            "status": sample["observation_status"],
            "video_id": result["video"]["video_id"],
            "requested_time": requested,
            "actual_frame_time": sample["actual_time"],
            "frame_id": sample["frame_id"],
            "supplemented": supplemented,
            "predictions": deepcopy(sample["predictions"]),
            "best_candidate": deepcopy(sample["best_candidate"]),
            "source": deepcopy(sample["source"]),
            "evidence": deepcopy(sample),
            "warning": "actual_frame_time 是实际读取帧时间，不用邻近帧冒充精确时刻。",
        }
    except PermissionError as exc:
        return _error("REALTIME_FUTURE_ACCESS", str(exc))
    except (ValueError, RuntimeError) as exc:
        return _error("INVALID_TIME_QUERY", str(exc))


def query_video_range(
    video_id: str,
    start_time: float,
    end_time: float,
    available_until: float = None,
) -> dict:
    result = load_video_result(video_id)
    if not result.get("ok"):
        return result
    try:
        start = _validate_access_time(result, start_time, available_until)
        end = _validate_access_time(result, end_time, available_until)
        if end < start:
            raise ValueError("end_time 不能早于 start_time")
        analyzed_start = float(result["analysis"]["start_time"])
        analyzed_end = float(result["analysis"]["end_time"])
        if start < analyzed_start - 1e-9 or end > analyzed_end + 1e-9:
            return _error(
                "RANGE_NOT_ANALYZED",
                "请求范围超出当前持久化分析范围，请先分析该范围",
                requested_start=start,
                requested_end=end,
                analyzed_start=analyzed_start,
                analyzed_end=analyzed_end,
            )
        samples = [
            deepcopy(item) for item in result["samples"]
            if start - 1e-9 <= float(item["actual_time"]) <= end + 1e-9
            and (available_until is None
                 or float(item["actual_time"]) <= float(available_until) + 1e-9)
        ]
        return {
            "ok": True,
            "status": "completed",
            "video_id": result["video"]["video_id"],
            "start_time": start,
            "end_time": end,
            "samples": samples,
            "events": _build_events(samples),
            "source": deepcopy(result["source"]),
            "sampling_basis": deepcopy(result["analysis"]),
            "warning": "仅汇总范围内已有预测采样证据；采样间隙不作推断。",
        }
    except PermissionError as exc:
        return _error("REALTIME_FUTURE_ACCESS", str(exc))
    except ValueError as exc:
        return _error("INVALID_TIME_RANGE", str(exc))


def summarize_video_events(
    video_id: str,
    start_time: float = None,
    end_time: float = None,
    available_until: float = None,
) -> dict:
    result = load_video_result(video_id)
    if not result.get("ok"):
        return result
    if start_time is None:
        start_time = result["analysis"]["start_time"]
    if end_time is None:
        end_time = (available_until if available_until is not None
                    else result["analysis"]["end_time"])
    ranged = query_video_range(
        result["video"]["video_id"], start_time, end_time, available_until
    )
    if not ranged.get("ok"):
        return ranged
    triplets = {}
    for sample in ranged["samples"]:
        for prediction in sample.get("predictions") or []:
            key = (prediction.get("instrument"), prediction.get("verb"),
                   prediction.get("target"))
            current = triplets.setdefault(key, {
                "instrument": key[0], "verb": key[1], "target": key[2],
                "max_score": 0.0, "evidence_times": [],
            })
            current["max_score"] = max(
                current["max_score"], float(prediction.get("confidence", 0.0))
            )
            current["evidence_times"].append(sample["actual_time"])
    return {
        "ok": True,
        "status": "completed",
        "video_id": ranged["video_id"],
        "start_time": ranged["start_time"],
        "end_time": ranged["end_time"],
        "predicted_triplets": sorted(
            triplets.values(), key=lambda item: item["max_score"], reverse=True
        ),
        "events": ranged["events"],
        "source": ranged["source"],
        "warning": "预测事件汇总不是人工真值或真实手术阶段。",
    }


def query_recent_video(video_id: str, reference_time: float, duration: float) -> dict:
    result = load_video_result(video_id)
    if not result.get("ok"):
        return result
    try:
        reference = _validate_access_time(result, reference_time, reference_time)
        window = _as_number(duration, "duration")
        if window <= 0:
            raise ValueError("duration 必须大于 0")
        return query_video_range(
            result["video"]["video_id"], max(0.0, reference - window),
            reference, available_until=reference,
        )
    except (ValueError, PermissionError) as exc:
        return _error("INVALID_RECENT_QUERY", str(exc))


# Backward-compatible APIs used by the original agent and tests.
def get_temporal_memory() -> TemporalMemory:
    return _temporal_memory


def get_current_video_id() -> Optional[str]:
    return _CURRENT_VIDEO_ID


def query_surgical_timeline(
    query_type: str,
    start_time: float = None,
    end_time: float = None,
    count: int = 10,
) -> dict:
    if _CURRENT_VIDEO_ID:
        if query_type == "time_range":
            if start_time == end_time:
                return query_video_time(_CURRENT_VIDEO_ID, start_time)
            return query_video_range(_CURRENT_VIDEO_ID, start_time, end_time)
        if query_type == "recent":
            result = load_video_result(_CURRENT_VIDEO_ID)
            samples = result.get("samples", [])[-int(count):]
            return {
                "ok": True, "status": "completed", "query_type": "recent",
                "video_id": _CURRENT_VIDEO_ID, "samples": deepcopy(samples),
                "events": _build_events(samples), "source": result.get("source"),
            }
        if query_type in ("statistics", "longest_action", "transitions"):
            summary = summarize_video_events(_CURRENT_VIDEO_ID)
            summary["query_type"] = query_type
            return summary
    if query_type == "recent":
        events = _temporal_memory.get_recent_events(count=count)
        return {"query_type": "recent", "events": events,
                "context": _temporal_memory.build_context_text(count=count)}
    if query_type == "time_range":
        return {"query_type": "time_range", "start_time": start_time,
                "end_time": end_time,
                "events": _temporal_memory.get_events_by_time(start_time, end_time)}
    if query_type == "statistics":
        return {"query_type": query_type,
                "statistics": _temporal_memory.get_statistics()}
    if query_type == "longest_action":
        return {"query_type": query_type,
                "event": _temporal_memory.get_longest_action()}
    if query_type == "transitions":
        return {"query_type": query_type,
                "transitions": _temporal_memory.get_transitions()}
    return _error("UNSUPPORTED_QUERY", f"不支持的 query_type: {query_type}")

