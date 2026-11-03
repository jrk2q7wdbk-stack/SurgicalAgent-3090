import os
import json

from tools.video_tool import (
    analyze_surgical_video,
    get_temporal_memory
)


def main():

    project_root = os.path.dirname(
        os.path.abspath(__file__)
    )

    video_path = os.path.join(
        project_root,
        "test.mp4"
    )

    if not os.path.exists(
        video_path
    ):
        raise FileNotFoundError(
            f"找不到测试视频：{video_path}"
        )

    result = analyze_surgical_video(
        video_path=video_path,

        # 先每 2 秒采样一帧
        sample_interval_seconds=2.0,

        # 调试阶段最多分析 10 帧
        max_frames=None
    )

    print()
    print("=" * 70)
    print("Video Result")
    print("=" * 70)

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )

    print()
    print("=" * 70)
    print("Temporal Context")
    print("=" * 70)

    memory = get_temporal_memory()

    print(
        memory.build_context_text(
            count=10
        )
    )


if __name__ == "__main__":
    main()