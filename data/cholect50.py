"""Strict reader for the verified CholecT50 Release 2.0 JSON format."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


RDV_ORIGINAL_SPLIT = {
    "train": [1, 15, 26, 40, 52, 65, 79, 2, 18, 27, 43, 56, 66, 92,
              4, 22, 31, 47, 57, 68, 96, 5, 23, 35, 48, 60, 70, 103,
              13, 25, 36, 49, 62, 75, 110],
    "validation": [8, 12, 29, 50, 78],
    "test": [6, 51, 10, 73, 14, 74, 32, 80, 42, 111],
}

CHOLECT50_CHALLENGE_SPLIT = {
    "train": [1, 15, 26, 40, 52, 79, 2, 27, 43, 56, 66, 4, 22, 31,
              47, 57, 68, 23, 35, 48, 60, 70, 13, 25, 49, 62, 75, 8,
              12, 29, 50, 78, 6, 51, 10, 73, 14, 32, 80, 42],
    "validation": [5, 18, 36, 65, 74],
    "test": [92, 96, 103, 110, 111],
}

CHOLECT50_CROSSVAL_FOLDS = {
    1: [79, 2, 51, 6, 25, 14, 66, 23, 50, 111],
    2: [80, 32, 5, 15, 40, 47, 26, 48, 70, 96],
    3: [31, 57, 36, 18, 52, 68, 10, 8, 73, 103],
    4: [42, 29, 60, 27, 65, 75, 22, 49, 12, 110],
    5: [78, 43, 62, 35, 74, 1, 56, 4, 13, 92],
}


def video_name(video_id: int | str) -> str:
    if isinstance(video_id, str) and video_id.upper().startswith("VID"):
        number = int(video_id[3:])
    else:
        number = int(video_id)
    return f"VID{number:02d}"


def get_video_split(variant: str = "rdv_original", fold: int = 1) -> Dict[str, List[str]]:
    if variant == "rdv_original":
        raw = RDV_ORIGINAL_SPLIT
    elif variant == "challenge":
        raw = CHOLECT50_CHALLENGE_SPLIT
    elif variant == "crossval":
        if fold not in CHOLECT50_CROSSVAL_FOLDS:
            raise ValueError("crossval fold 必须是 1～5")
        test = list(CHOLECT50_CROSSVAL_FOLDS[fold])
        remaining = [
            item for key, values in CHOLECT50_CROSSVAL_FOLDS.items()
            if key != fold for item in values
        ]
        # Matches the official loader: last five remaining videos are val.
        raw = {"train": remaining[:-5], "validation": remaining[-5:], "test": test}
    else:
        raise ValueError(f"未知 CholecT50 split variant：{variant}")
    return {key: [video_name(value) for value in values] for key, values in raw.items()}


@dataclass(frozen=True)
class TripletInstance:
    triplet_id: int
    instrument_id: int
    instrument_score: float
    instrument_bbox: Tuple[float, float, float, float]
    verb_id: int
    target_id: int
    target_score: float
    target_bbox: Tuple[float, float, float, float]
    phase_id: int
    instrument: str
    verb: str
    target: str
    triplet: str
    phase: Optional[str]
    raw: Tuple[float, ...]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["instrument_bbox"] = list(self.instrument_bbox)
        value["target_bbox"] = list(self.target_bbox)
        value["raw"] = list(self.raw)
        return value


@dataclass(frozen=True)
class FrameAnnotation:
    video_id: str
    frame_id: int
    timestamp: float
    fps: float
    frame_path: Path
    triplets: Tuple[TripletInstance, ...]
    phase_id: Optional[int]
    phase: Optional[str]
    annotation_path: Path

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "fps": self.fps,
            "frame_path": str(self.frame_path),
            "triplets": [item.to_dict() for item in self.triplets],
            "phase_id": self.phase_id,
            "phase": self.phase,
            "annotation_path": str(self.annotation_path),
        }


@dataclass
class VideoAnnotation:
    video_id: str
    fps: float
    num_frames: int
    dataset_version: str
    categories: Dict[str, Dict[str, str]]
    annotation_path: Path
    frames_dir: Path
    annotations: Dict[str, list]
    mapping: Dict[int, Tuple[int, int, int]]

    def _name(self, category: str, category_id: int) -> Optional[str]:
        if category_id < 0:
            return None
        value = self.categories.get(category, {}).get(str(category_id))
        if value is None:
            raise ValueError(
                f"{self.video_id}: {category} id {category_id} 不在 categories 中"
            )
        return value

    def _decode_instance(self, row: list, frame_id: int) -> Optional[TripletInstance]:
        if not isinstance(row, list) or len(row) != 15:
            raise ValueError(
                f"{self.video_id} frame {frame_id}: 标注记录必须有15列"
            )
        triplet_id = int(row[0])
        phase_id = int(row[14])
        if triplet_id < 0:
            return None
        instrument_id, verb_id, target_id = int(row[1]), int(row[7]), int(row[8])
        expected = self.mapping.get(triplet_id)
        if expected is None:
            raise ValueError(f"{self.video_id}: triplet id {triplet_id} 不在映射中")
        actual = (instrument_id, verb_id, target_id)
        if actual != expected:
            raise ValueError(
                f"{self.video_id} frame {frame_id}: triplet {triplet_id} 组件"
                f"{actual} 与 label_mapping {expected} 不一致"
            )
        instrument = self._name("instrument", instrument_id)
        verb = self._name("verb", verb_id)
        target = self._name("target", target_id)
        triplet = self._name("triplet", triplet_id)
        expected_name = f"{instrument},{verb},{target}"
        if triplet != expected_name:
            raise ValueError(
                f"{self.video_id}: triplet名称 {triplet!r} 与组件 {expected_name!r} 不一致"
            )
        return TripletInstance(
            triplet_id=triplet_id,
            instrument_id=instrument_id,
            instrument_score=float(row[2]),
            instrument_bbox=tuple(float(v) for v in row[3:7]),
            verb_id=verb_id,
            target_id=target_id,
            target_score=float(row[9]),
            target_bbox=tuple(float(v) for v in row[10:14]),
            phase_id=phase_id,
            instrument=instrument,
            verb=verb,
            target=target,
            triplet=triplet,
            phase=self._name("phase", phase_id),
            raw=tuple(float(v) for v in row),
        )

    def frame(self, frame_id: int) -> FrameAnnotation:
        if frame_id < 0 or frame_id >= self.num_frames:
            raise IndexError(
                f"{self.video_id}: frame_id {frame_id} 超出 0～{self.num_frames - 1}"
            )
        rows = self.annotations.get(str(frame_id))
        if rows is None:
            raise ValueError(f"{self.video_id}: 缺少 frame {frame_id} 标注")
        decoded = tuple(
            item for item in (self._decode_instance(row, frame_id) for row in rows)
            if item is not None
        )
        phase_ids = {int(row[14]) for row in rows if int(row[14]) >= 0}
        if len(phase_ids) > 1:
            raise ValueError(f"{self.video_id} frame {frame_id}: 帧内phase不一致")
        phase_id = next(iter(phase_ids), None)
        return FrameAnnotation(
            video_id=self.video_id,
            frame_id=frame_id,
            timestamp=frame_id / self.fps,
            fps=self.fps,
            frame_path=self.frames_dir / f"{frame_id:06d}.png",
            triplets=decoded,
            phase_id=phase_id,
            phase=self._name("phase", phase_id) if phase_id is not None else None,
            annotation_path=self.annotation_path,
        )

    def iter_frames(self) -> Iterator[FrameAnnotation]:
        for frame_id in range(self.num_frames):
            yield self.frame(frame_id)


class CholecT50Reader:
    def __init__(self, root: str | Path):
        candidate = Path(root).expanduser().resolve()
        nested = candidate / "CholecT50"
        if not (candidate / "labels").is_dir() and (nested / "labels").is_dir():
            candidate = nested
        self.root = candidate
        self.labels_dir = candidate / "labels"
        self.videos_dir = candidate / "videos"
        self.mapping_path = candidate / "label_mapping.txt"
        if not self.labels_dir.is_dir() or not self.videos_dir.is_dir():
            raise FileNotFoundError(
                f"不是有效的CholecT50根目录（需要labels和videos）：{candidate}"
            )
        self.mapping = self._load_mapping()

    def _load_mapping(self) -> Dict[int, Tuple[int, int, int]]:
        mapping = {}
        for raw in self.mapping_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            values = [int(value.strip()) for value in line.split(",")]
            if len(values) != 6:
                raise ValueError(f"label_mapping行不是6列：{raw}")
            mapping[values[0]] = (values[1], values[2], values[3])
        if set(mapping) != set(range(100)):
            raise ValueError("label_mapping必须完整包含0～99三元组")
        return mapping

    def video_ids(self) -> List[str]:
        return sorted(path.stem for path in self.labels_dir.glob("VID*.json"))

    def read_video(self, video_id: int | str) -> VideoAnnotation:
        name = video_name(video_id)
        annotation_path = self.labels_dir / f"{name}.json"
        frames_dir = self.videos_dir / name
        if not annotation_path.is_file() or not frames_dir.is_dir():
            raise FileNotFoundError(f"缺少视频或标注：{name}")
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        required = {"annotations", "fps", "info", "categories", "num_frames", "video"}
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"{name}: JSON缺少字段 {missing}")
        version = str(data["info"].get("version"))
        if version != "2.0":
            raise ValueError(f"{name}: 仅支持已验证的Release 2.0，实际为 {version}")
        fps = float(data["fps"])
        if fps <= 0:
            raise ValueError(f"{name}: fps必须大于0")
        num_frames = int(data["num_frames"])
        if len(data["annotations"]) != num_frames:
            raise ValueError(f"{name}: annotations数量与num_frames不一致")
        expected_keys = {str(index) for index in range(num_frames)}
        if set(data["annotations"]) != expected_keys:
            raise ValueError(f"{name}: 标注帧键不连续")
        return VideoAnnotation(
            video_id=name,
            fps=fps,
            num_frames=num_frames,
            dataset_version=version,
            categories=data["categories"],
            annotation_path=annotation_path,
            frames_dir=frames_dir,
            annotations=data["annotations"],
            mapping=self.mapping,
        )

    def validate(self, decode_all: bool = True) -> dict:
        video_ids = self.video_ids()
        split_union = {
            item for values in get_video_split("rdv_original").values() for item in values
        }
        issues = []
        total_frames = 0
        total_instances = 0
        multi_instance_frames = 0
        empty_triplet_frames = 0
        for name in video_ids:
            video = self.read_video(name)
            frame_file_count = sum(1 for _ in video.frames_dir.glob("*.png"))
            if frame_file_count != video.num_frames:
                issues.append({
                    "video_id": name,
                    "code": "FRAME_COUNT_MISMATCH",
                    "declared": video.num_frames,
                    "files": frame_file_count,
                })
            total_frames += video.num_frames
            if decode_all:
                for frame in video.iter_frames():
                    total_instances += len(frame.triplets)
                    multi_instance_frames += int(len(frame.triplets) > 1)
                    empty_triplet_frames += int(len(frame.triplets) == 0)
                    if not frame.frame_path.is_file():
                        issues.append({
                            "video_id": name, "frame_id": frame.frame_id,
                            "code": "FRAME_FILE_MISSING",
                        })
        if set(video_ids) != split_union:
            issues.append({
                "code": "SPLIT_VIDEO_MISMATCH",
                "missing_from_split": sorted(set(video_ids) - split_union),
                "missing_from_dataset": sorted(split_union - set(video_ids)),
            })
        return {
            "dataset": "CholecT50",
            "version": "2.0",
            "root": str(self.root),
            "video_count": len(video_ids),
            "frame_count": total_frames,
            "triplet_instance_count": total_instances if decode_all else None,
            "multi_instance_frame_count": multi_instance_frames if decode_all else None,
            "empty_triplet_frame_count": empty_triplet_frames if decode_all else None,
            "split_variant": "rdv_original",
            "split_counts": {
                key: len(value) for key, value in get_video_split("rdv_original").items()
            },
            "valid": not issues,
            "issues": issues,
        }

