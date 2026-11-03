import json
from pathlib import Path
import tempfile
import unittest

from data.cholect50 import CholecT50Reader, get_video_split
from data.qa_pipeline import (
    _point_records,
    _range_tool_result,
    build_continuous_events,
    to_conversation,
    validate_record,
)


class CholecT50PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        (root / "labels").mkdir()
        frames = root / "videos" / "VID01"
        frames.mkdir(parents=True)
        mapping_source = Path(__file__).resolve().parents[1] / "knowledge" / "label_mapping.txt"
        (root / "label_mapping.txt").write_text(
            mapping_source.read_text(encoding="utf-8"), encoding="utf-8"
        )
        for frame_id in range(3):
            (frames / f"{frame_id:06d}.png").write_bytes(b"test")
        categories = {
            "instrument": {"0": "grasper", "2": "hook"},
            "verb": {"0": "grasp", "2": "dissect"},
            "target": {"0": "gallbladder", "1": "cystic_plate"},
            "triplet": {
                "0": "grasper,dissect,cystic_plate",
                "7": "grasper,grasp,gallbladder",
            },
            "phase": {"0": "preparation"},
        }
        row7 = [7, 0, 1, -1, -1, -1, -1, 0, 0, 1, -1, -1, -1, -1, 0]
        row0 = [0, 0, 1, -1, -1, -1, -1, 2, 1, 1, -1, -1, -1, -1, 0]
        payload = {
            "video": 1,
            "fps": 1,
            "num_frames": 3,
            "info": {"version": 2.0},
            "licenses": {},
            "categories": categories,
            "annotations": {"0": [row7], "1": [row7, row0], "2": [row0]},
        }
        (root / "labels" / "VID01.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        self.reader = CholecT50Reader(root)
        self.video = self.reader.read_video("VID01")

    def tearDown(self):
        self.temp.cleanup()

    def test_release2_reader_decodes_all_triplets_and_timestamp(self):
        frame = self.video.frame(1)
        self.assertEqual(frame.timestamp, 1.0)
        self.assertEqual(len(frame.triplets), 2)
        self.assertEqual(frame.triplets[0].triplet, "grasper,grasp,gallbladder")
        self.assertEqual(frame.triplets[1].triplet_id, 0)

    def test_continuous_events_use_consecutive_frames(self):
        events = build_continuous_events(self.video)
        event7 = next(item for item in events if item["triplet_id"] == 7)
        event0 = next(item for item in events if item["triplet_id"] == 0)
        self.assertEqual((event7["start_frame"], event7["end_frame"]), (0, 1))
        self.assertEqual((event0["start_frame"], event0["end_frame"]), (1, 2))

    def test_range_clipping_is_explicit(self):
        events = build_continuous_events(self.video)
        result = _range_tool_result(self.video, events, 1.0, 1.0)
        self.assertEqual(len(result["observed_triplets"]), 2)
        self.assertTrue(all(
            item["left_truncated"] or item["right_truncated"]
            for item in result["events"]
        ))

    def test_standard_record_and_conversation_are_traceable(self):
        record = next(_point_records(self.video.frame(1), "train"))
        self.assertEqual(record["evidence_source"]["type"], "human_annotation")
        self.assertEqual(validate_record(record), [])
        conversation = to_conversation(record)
        self.assertEqual(
            conversation["messages"][2]["tool_calls"][0]["function"]["name"],
            "query_video_time",
        )
        self.assertEqual(conversation["messages"][3]["role"], "tool")

    def test_rdv_original_split_is_video_disjoint(self):
        split = get_video_split("rdv_original")
        self.assertEqual({key: len(value) for key, value in split.items()}, {
            "train": 35, "validation": 5, "test": 10,
        })
        all_ids = split["train"] + split["validation"] + split["test"]
        self.assertEqual(len(all_ids), len(set(all_ids)))


if __name__ == "__main__":
    unittest.main()

