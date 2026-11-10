import os

from agent.llm_agent import (
    SurgicalLLMAgent
)


def main():

    # ========================================================
    # Project Path
    # ========================================================

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
            f"找不到测试视频："
            f"{video_path}"
        )

    # ========================================================
    # Create Agent
    # ========================================================

    agent = SurgicalLLMAgent(
        max_history_turns=10,
        max_tool_rounds=6
    )

    # ========================================================
    # Question 1
    # 第一次提供视频
    # ========================================================

    print()
    print("=" * 70)
    print(
        "Question 1: "
        "这段视频中发生了什么？"
    )
    print("=" * 70)

    answer_1 = agent.answer(
        question=(
            "请分析这段手术视频，"
            "告诉我视频中依次发生了哪些器械操作。"
        ),
        video_path=video_path
    )

    print()
    print(
        "Answer 1:"
    )

    print(
        answer_1
    )

    # ========================================================
    # Question 2
    # 不再提供视频
    # 应直接使用 Temporal Memory
    # ========================================================

    print()
    print("=" * 70)
    print(
        "Question 2: "
        "刚才主要进行了什么操作？"
    )
    print("=" * 70)

    answer_2 = agent.answer(
        question=(
            "刚才主要进行了什么操作？"
        )
    )

    print()
    print(
        "Answer 2:"
    )

    print(
        answer_2
    )

    # ========================================================
    # Question 3
    # 测试 no_action
    # ========================================================

    print()
    print("=" * 70)
    print(
        "Question 3: "
        "中间有没有一段时间没有明确器械操作？"
    )
    print("=" * 70)

    answer_3 = agent.answer(
        question=(
            "中间有没有一段时间"
            "没有检测到明确器械操作？"
        )
    )

    print()
    print(
        "Answer 3:"
    )

    print(
        answer_3
    )

    # ========================================================
    # Question 4
    # 操作变化
    # ========================================================

    print()
    print("=" * 70)
    print(
        "Question 4: "
        "前后操作发生了什么变化？"
    )
    print("=" * 70)

    answer_4 = agent.answer(
        question=(
            "前后操作发生了什么变化？"
        )
    )

    print()
    print(
        "Answer 4:"
    )

    print(
        answer_4
    )

    # ========================================================
    # Print Temporal Memory
    # ========================================================

    print()
    print("=" * 70)
    print(
        "Current Temporal Memory"
    )
    print("=" * 70)

    from tools.video_tool import (
        get_temporal_memory
    )

    temporal_memory = (
        get_temporal_memory()
    )

    print(
        temporal_memory
        .build_context_text(
            count=10
        )
    )

    print()
    print("=" * 70)
    print(
        "Conversation Memory"
    )
    print("=" * 70)

    print(
        "History messages:",
        len(
            agent.memory.history
        )
    )


if __name__ == "__main__":
    main()