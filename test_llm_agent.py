import os

from agent.llm_agent import SurgicalLLMAgent


def main():

    # ========================================================
    # 项目根目录
    # ========================================================
    project_root = os.path.dirname(
        os.path.abspath(__file__)
    )

    # ========================================================
    # 测试图片
    # ========================================================
    image_path = os.path.join(
        project_root,
        "test.jpg"
    )

    # ========================================================
    # 检查图片
    # ========================================================
    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"找不到测试图像：{image_path}"
        )

    # ========================================================
    # 创建 Agent
    # ========================================================
    agent = SurgicalLLMAgent()

    # ========================================================
    # 测试问题
    # ========================================================
    questions = [

        "当前医生正在做什么？",

        "当前使用的是什么器械？",

        "当前操作作用于什么组织？",

        "当前最佳手术三元组是什么？",

        "这个操作是什么意思？"
    ]

    # ========================================================
    # 开始测试
    # ========================================================
    for question in questions:

        print()
        print("=" * 70)

        print(
            f"Question: {question}"
        )

        print("-" * 70)

        try:

            answer = agent.answer(
                question=question,
                image_path=image_path
            )

            print(
                f"Answer: {answer}"
            )

        except Exception as exc:

            print(
                f"[ERROR] {exc}"
            )


if __name__ == "__main__":
    main()