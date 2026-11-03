from capability_logging import run_logged


VIDEO_PATH = "test.mp4"
IMAGE_PATH = "test.jpg"


def ask(agent, number, category, question,
        image_path=None, video_path=None):

    print("\n")
    print("=" * 80)
    print(f"TEST {number}: {category}")
    print("=" * 80)

    print("\n[Question]")
    print(question)

    print("\n[Agent Answer]")

    answer = agent.answer(
        question=question,
        image_path=image_path,
        video_path=video_path
    )

    print(answer)

    return answer


def main():

    print("\n")
    print("=" * 80)
    print("               SurgicalAgent Capability Demonstration")
    print("=" * 80)

    from agent.llm_agent import SurgicalLLMAgent

    agent = SurgicalLLMAgent()

    # ========================================================
    # Part 1
    # Image Perception
    # ========================================================

    ask(
        agent,
        1,
        "Image Perception",
        "请分析这张手术图像，当前识别到的手术三元组是什么？",
        image_path=IMAGE_PATH
    )

    # ========================================================
    # Part 2
    # Conversation Memory
    # ========================================================

    ask(
        agent,
        2,
        "Conversation Memory",
        "刚才识别到的器械是什么？"
    )

    # ========================================================
    # Part 3
    # Surgical Knowledge
    # ========================================================

    ask(
        agent,
        3,
        "Surgical Knowledge",
        (
            "刚才识别到的器械、动作和目标分别是什么意思？"
            "请查询手术知识库回答。"
        )
    )

    # ========================================================
    # Part 4
    # Video Understanding
    # ========================================================

    ask(
        agent,
        4,
        "Video Understanding",
        (
            "请分析这段手术视频，"
            "建立手术事件时间线，并总结视频中识别到的操作。"
        ),
        video_path=VIDEO_PATH
    )

    # ========================================================
    # Part 5
    # Temporal Point Query
    # ========================================================

    ask(
        agent,
        5,
        "Temporal Point Query",
        "20秒时模型识别到了什么操作？"
    )

    # ========================================================
    # Part 6
    # no_action
    # ========================================================

    ask(
        agent,
        6,
        "No-action Detection",
        (
            "12秒时模型识别到了明确的器械操作吗？"
            "请根据视频时间线回答。"
        )
    )

    # ========================================================
    # Part 7
    # Temporal Transition
    # ========================================================

    ask(
        agent,
        7,
        "Temporal Reasoning",
        (
            "这个视频中操作过程发生了哪些变化？"
            "请说明器械和手术三元组的变化顺序。"
        )
    )

    # ========================================================
    # Part 8
    # Longest Action
    # ========================================================

    ask(
        agent,
        8,
        "Temporal Statistics",
        "视频中持续时间最长的明确操作是什么？"
    )

    # ========================================================
    # Part 9
    # KG Class Query
    # ========================================================

    ask(
        agent,
        9,
        "Knowledge Graph - IVT",
        (
            "CholecT50知识图谱中，"
            "IVT class 58对应什么手术三元组？"
        )
    )

    # ========================================================
    # Part 10
    # KG Relation Query
    # ========================================================

    ask(
        agent,
        10,
        "Knowledge Graph - Relation",
        (
            "在CholecT50知识图谱中，"
            "hook作用于cystic_duct时"
            "有哪些有效动作？"
        )
    )

    # ========================================================
    # Part 11
    # Temporal + Knowledge
    # ========================================================

    ask(
        agent,
        11,
        "Temporal + Surgical Knowledge",
        (
            "20秒时模型识别到了什么操作？"
            "这个操作中的器械、动作和目标分别是什么意思？"
        )
    )

    # ========================================================
    # Part 12
    # Temporal + KG
    # ========================================================

    ask(
        agent,
        12,
        "Temporal + Knowledge Graph",
        (
            "20秒时模型识别到了什么操作？"
            "这个器械在CholecT50中"
            "还能对gallbladder执行哪些有效动作？"
            "请区分视频识别结果和知识图谱关系。"
        )
    )

    # ========================================================
    # Part 13
    # Another Temporal + KG
    # ========================================================

    ask(
        agent,
        13,
        "Temporal + KG Multi-source Reasoning",
        (
            "34秒时模型识别到了什么三元组？"
            "对于这个器械和目标，"
            "CholecT50中还有哪些有效动作？"
        )
    )

    # ========================================================
    # Part 14
    # Evidence Boundary
    # ========================================================

    ask(
        agent,
        14,
        "Evidence Boundary",
        (
            "知识图谱中hook作用于cystic_duct时"
            "存在coagulate和dissect两种关系。"
            "那么是否可以认为视频0到8秒"
            "同时发生了coagulate和dissect？"
            "请分别根据视频证据和知识图谱回答。"
        )
    )

    # ========================================================
    # Part 15
    # Low confidence / evidence boundary
    # ========================================================

    ask(
        agent,
        15,
        "Uncertainty Awareness",
        (
            "为什么12秒不能直接认为"
            "医生正在使用grasper抓取gallbladder？"
            "请结合模型识别结果解释。"
        )
    )
    ask(
        agent,
        16,
        "Medium Confidence",
        "16秒时模型识别到了什么操作？请使用该采样点自身的置信度回答。"
    )

    ask(
        agent,
        17,
        "Video Source",
        "视频最后一个采样点是几秒？识别到了什么三元组？请报告该点的置信度。"
    )

    ask(
        agent,
        18,
        "Image Source",
        "之前的静态图片test.jpg识别到了什么三元组？请只根据这张图片的结果回答。"
    )

    print("\n")
    print("=" * 80)
    print("               Capability Demonstration Finished")
    print("=" * 80)


if __name__ == "__main__":
    run_logged(main)