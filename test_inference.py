import json
import os

from tools.vision_tool import analyze_surgical_image


if __name__ == "__main__":

    # 当前 SurgicalAgent 项目根目录
    project_root = os.path.dirname(
        os.path.abspath(__file__)
    )

    # 测试图片路径
    image_path = os.path.join(
        project_root,
        "test.jpg"
    )

    print("=" * 60)
    print("Surgical Triplet Vision Tool Test")
    print("=" * 60)

    print(f"Test image: {image_path}")

    # 调用 Vision Tool
    result = analyze_surgical_image(
        image_path
    )

    # 打印识别结果
    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        )
    )