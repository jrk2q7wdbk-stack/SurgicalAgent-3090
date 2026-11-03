from agent.llm_agent import SurgicalLLMAgent


def main():

    agent = SurgicalLLMAgent()

    video_path = "../test.mp4"

    # ========================================================
    # Q1
    # 首先分析视频
    # ========================================================

    print("\n" + "=" * 70)
    print("Q1：请分析这段手术视频。")
    print("=" * 70)

    answer = agent.answer(
        question="请分析这段手术视频。",
        video_path=video_path
    )

    print(answer)

    # ========================================================
    # Q2
    # Temporal + Knowledge
    # ========================================================

    print("\n" + "=" * 70)
    print("Q2：20秒时正在进行什么操作？这个操作是什么意思？")
    print("=" * 70)

    answer = agent.answer(
        question=(
            "20秒时正在进行什么操作？"
            "这个操作是什么意思？"
        )
    )

    print(answer)

    # ========================================================
    # Q3
    # Temporal + Knowledge
    # ========================================================

    print("\n" + "=" * 70)
    print("Q3：34秒时使用了什么器械？这个器械通常有什么作用？")
    print("=" * 70)

    answer = agent.answer(
        question=(
            "34秒时使用了什么器械？"
            "这个器械通常有什么作用？"
        )
    )

    print(answer)

    # ========================================================
    # Q4
    # Transition + Knowledge
    # ========================================================

    print("\n" + "=" * 70)
    print("Q4：视频后半段操作发生了什么变化？这些操作分别是什么意思？")
    print("=" * 70)

    answer = agent.answer(
        question=(
            "视频后半段操作发生了什么变化？"
            "这些操作分别是什么意思？"
        )
    )

    print(answer)


if __name__ == "__main__":
    main()