import os

from agent.llm_agent import SurgicalLLMAgent


def main():

    project_root = os.path.dirname(
        os.path.abspath(__file__)
    )

    image_path = os.path.join(
        project_root,
        "test.jpg"
    )

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"找不到测试图片：{image_path}"
        )

    # 创建 Agent
    agent = SurgicalLLMAgent(
        max_history_turns=10
    )

    # ========================================================
    # 开发阶段只测试 3 个核心问题
    # ========================================================

    questions = [
        (
            "当前医生正在做什么？",
            image_path
        ),
        (
            "这个器械是什么？",
            None
        ),
        (
            "这个操作是什么意思？",
            None
        ),
    ]

    # ========================================================
    # 开始测试
    # ========================================================

    for index, (question, current_image) in enumerate(
        questions,
        start=1
    ):

        print()
        print("=" * 70)
        print(
            f"Question {index}: {question}"
        )
        print("-" * 70)

        answer = agent.answer(
            question=question,
            image_path=current_image
        )

        print(
            f"Answer {index}: {answer}"
        )

    # ========================================================
    # 查看 Memory
    # ========================================================

    print()
    print("=" * 70)
    print("Current Memory")
    print("-" * 70)

    best_triplet = (
        agent.memory.get_best_triplet()
    )

    print(
        "Best triplet:",
        best_triplet
    )

    print(
        "History messages:",
        len(agent.memory.history)
    )


if __name__ == "__main__":
    main()