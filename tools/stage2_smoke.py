"""Reproducible real-RDV smoke validation for stage two."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.video_tool import (  # noqa: E402
    analyze_surgical_video,
    query_video_time,
    query_video_range,
    summarize_video_events,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="test.mp4")
    parser.add_argument("--output", default="artifacts/stage2_smoke.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    analysis = analyze_surgical_video(
        args.video, start_time=10.0, end_time=14.0,
        sample_interval_seconds=2.0, force=args.force,
    )
    report = {"analysis": analysis}
    if analysis.get("ok"):
        video_id = analysis["video"]["video_id"]
        report["exact_unsampled_11s"] = query_video_time(video_id, 11.0)
        report["exact_unsampled_11s_again"] = query_video_time(video_id, 11.0)
        report["range_10_14"] = query_video_range(video_id, 10.0, 14.0)
        report["summary_10_14"] = summarize_video_events(
            video_id, 10.0, 14.0
        )
        report["realtime_future_rejection"] = query_video_time(
            video_id, 13.0, available_until=12.0
        )
        report["cache_reload"] = analyze_surgical_video(
            args.video, start_time=10.0, end_time=14.0,
            sample_interval_seconds=2.0, force=False,
        ).get("cache")
    report["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    concise = {
        "ok": analysis.get("ok"),
        "video_id": analysis.get("video", {}).get("video_id"),
        "sample_count": len(analysis.get("samples", [])),
        "prediction_counts": [
            len(item.get("predictions", [])) for item in analysis.get("samples", [])
        ],
        "exact_supplemented_first": report.get("exact_unsampled_11s", {}).get(
            "supplemented"
        ),
        "exact_supplemented_again": report.get(
            "exact_unsampled_11s_again", {}
        ).get("supplemented"),
        "future_error": report.get("realtime_future_rejection", {}).get(
            "error", {}
        ).get("code"),
        "cache_reload": report.get("cache_reload"),
        "elapsed_ms": report["elapsed_ms"],
        "output": str(output.resolve()),
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

