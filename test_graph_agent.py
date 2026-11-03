from agent.llm_agent import SurgicalLLMAgent


def ask(agent, title, question):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    answer = agent.answer(
        question=question
    )

    print(answer)


def main():

    agent = SurgicalLLMAgent()

    # ========================================================
    # Q1: IVT ID
    # ========================================================

    ask(
        agent,
        "Q1：IVT 58",
        (
            "CholecT50 中 IVT class 58 "
            "对应什么手术三元组？"
            "请查询知识图谱回答。"
        )
    )

    # ========================================================
    # Q2: Instrument
    # ========================================================

    ask(
        agent,
        "Q2：hook 有哪些 Triplet",
        (
            "在 CholecT50 知识图谱中，"
            "hook 对应哪些有效手术三元组？"
            "请告诉我总数。"
        )
    )

    # ========================================================
    # Q3: Relation
    # ========================================================

    ask(
        agent,
        "Q3：hook + cystic_duct",
        (
            "在 CholecT50 中，"
            "hook 作用于 cystic_duct "
            "有哪些有效动作？"
            "只根据知识图谱回答。"
        )
    )

    # ========================================================
    # Q4: Supported Actions
    # ========================================================

    ask(
        agent,
        "Q4：hook 支持哪些动作",
        (
            "根据手术知识库中的图关系，"
            "hook 支持哪些动作？"
            "请不要把这些知识关系描述成"
            "当前视频中的实际操作。"
        )
    )

    # ========================================================
    # Q5: Distinguish KG from Video Evidence
    # ========================================================

    ask(
        agent,
        "Q5：证据边界测试",
        (
            "知识图谱中存在 "
            "hook-coagulate-cystic_duct，"
            "这是否意味着当前视频里医生"
            "正在用 hook coagulate cystic_duct？"
        )
    )


if __name__ == "__main__":
    main()