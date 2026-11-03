"""Command-line entry for deterministic offline or simulated-real-time QA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.video_qa import answer_video_question  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="video path; omit to reuse loaded cache")
    parser.add_argument("--video-id", help="persisted video identity")
    parser.add_argument("--question", required=True)
    parser.add_argument("--sample-interval", type=float)
    parser.add_argument("--reference-time", type=float)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    result = answer_video_question(
        question=args.question,
        video_path=args.video,
        video_id=args.video_id,
        sample_interval=args.sample_interval,
        reference_time=args.reference_time,
        simulated_realtime=args.realtime,
        force=args.force,
    )
    print(
        json.dumps(result, ensure_ascii=False, indent=2, default=str)
        if args.as_json else result["answer"]
    )


if __name__ == "__main__":
    main()

