from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.query_router import parse_temporal_expression
from agent.tool_registry import validate_tool_arguments
import tools.video_tool as video_tool


def sample(requested, actual=None, confidence=0.8, kind="scheduled"):
    actual = requested if actual is None else actual
    triplet = {
        "id": 1, "instrument": "hook", "verb": "dissect",
        "target": "cystic_duct", "confidence": confidence,
    }
    return {
        "requested_time": requested,
        "actual_time": actual,
        "frame_id": int(actual * 25),
        "sampling_kind": kind,
        "timestamp_basis": "original_video_frame_index_divided_by_fps",
        "observation_status": "recognized_prediction",
        "predictions": [deepcopy(triplet)],
        "best_candidate": deepcopy(triplet),
        "components": {"instruments": [], "verbs": [], "targets": []},
        "source": {"type": "rdv_prediction", "is_ground_truth": False},
    }


def result_fixture(path="test.mp4"):
    video_id = "sha256:test-video"
    samples = [sample(0.0), sample(2.0), sample(4.0)]
    return {
        "ok": True,
        "status": "completed",
        "schema_version": video_tool.SCHEMA_VERSION,
        "video": {
            "video_id": video_id, "path": str(Path(path).resolve()),
            "name": "test.mp4", "content_sha256": "test-video",
            "size_bytes": 1, "fps": 25.0, "total_frames": 251,
            "duration": 10.04, "last_frame_time": 10.0,
        },
        "model": {"version": "test"},
        "analysis": {"start_time": 0.0, "end_time": 10.0,
                     "sample_interval": 2.0, "max_frames": None,
                     "timestamp_scope": "absolute_time_in_original_video"},
        "samples": samples,
        "events": video_tool._build_events(samples),
        "event_sequence": video_tool._build_events(samples),
        "source": {"type": "rdv_prediction", "is_ground_truth": False},
        "cache": {"hit": False, "key": "test-key"},
        "persistence": {"cache_path": str(Path("unused.json").resolve())},
    }


class VideoStageTwoTests(unittest.TestCase):
    def setUp(self):
        video_tool._RESULTS.clear()
        video_tool._CURRENT_VIDEO_ID = None

    def test_sampling_includes_absolute_end_boundary(self):
        self.assertEqual(video_tool._analysis_times(10, 15, 2), [10, 12, 14, 15])

    def test_inverted_range_is_not_silently_swapped(self):
        parsed = parse_temporal_expression("第30秒到第10秒发生了什么")
        self.assertEqual(parsed["start_time"], 30.0)
        self.assertEqual(parsed["end_time"], 10.0)

    def test_recent_requires_explicit_reference(self):
        parsed = parse_temporal_expression("过去20秒发生了什么")
        self.assertTrue(parsed["needs_reference_time"])
        parsed = parse_temporal_expression("过去20秒发生了什么", reference_time=35)
        self.assertEqual((parsed["start_time"], parsed["end_time"]), (15.0, 35.0))

    def test_event_group_is_prediction_not_phase(self):
        event = video_tool._build_events([sample(1.0)])[0]
        self.assertEqual(event["event_type"], "predicted_observation_group")
        self.assertFalse(event["source"]["is_ground_truth"])
        self.assertIn("不是真实手术阶段", event["warning"])
        self.assertEqual(len(event["sample_evidence"][0]["predictions"]), 1)

    def test_tool_validation_rejects_missing_and_boolean_time(self):
        missing = validate_tool_arguments("query_video_time", {"video_id": "x"})
        self.assertEqual(missing["error"]["code"], "MISSING_ARGUMENT")
        wrong = validate_tool_arguments(
            "query_video_time", {"video_id": "x", "time": True}
        )
        self.assertEqual(wrong["error"]["code"], "INVALID_ARGUMENT_TYPE")

    def test_realtime_boundary_rejects_future_query(self):
        result = result_fixture()
        video_tool._RESULTS[result["video"]["video_id"]] = result
        queried = video_tool.query_video_time(
            result["video"]["video_id"], 8.0, available_until=5.0
        )
        self.assertEqual(queried["error"]["code"], "REALTIME_FUTURE_ACCESS")

    def test_range_returns_only_available_samples(self):
        result = result_fixture()
        video_tool._RESULTS[result["video"]["video_id"]] = result
        queried = video_tool.query_video_range(
            result["video"]["video_id"], 0.0, 2.0, available_until=2.0
        )
        self.assertEqual([x["actual_time"] for x in queried["samples"]], [0.0, 2.0])

    def test_range_outside_analyzed_scope_is_not_silently_partial(self):
        result = result_fixture()
        result["analysis"]["end_time"] = 4.0
        video_tool._RESULTS[result["video"]["video_id"]] = result
        queried = video_tool.query_video_range(
            result["video"]["video_id"], 0.0, 8.0
        )
        self.assertEqual(queried["error"]["code"], "RANGE_NOT_ANALYZED")

    def test_invalid_analysis_range_has_structured_error(self):
        metadata = result_fixture()["video"]
        with patch.object(video_tool, "_video_metadata", return_value=metadata):
            result = video_tool.analyze_surgical_video(
                "ignored.mp4", start_time=5, end_time=2
            )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "VIDEO_ANALYSIS_INVALID")

    def test_persisted_result_can_reload(self):
        result = result_fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.json"
            result["persistence"]["cache_path"] = str(path)
            path.write_text(json.dumps(result), encoding="utf-8")
            loaded = video_tool.load_video_result(str(path))
        self.assertTrue(loaded["ok"])
        self.assertEqual(loaded["video"]["video_id"], "sha256:test-video")


if __name__ == "__main__":
    unittest.main()

