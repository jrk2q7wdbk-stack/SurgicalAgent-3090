from agent.llm_agent import SurgicalLLMAgent


def main():

    agent = SurgicalLLMAgent()

    print("\n" + "=" * 70)
    print("Q1")
    print("=" * 70)

    answer = agent.answer(
        question=(
            "grasper - grasp - gallbladder "
            "这个手术三元组是什么意思？"
            "请区分知识库提供的信息和你的补充解释。"
        )
    )

    print(answer)

    print("\n" + "=" * 70)
    print("Q2")
    print("=" * 70)

    answer = agent.answer(
        question=(
            "irrigator - aspirate - fluid "
            "是什么意思？"
            "请只根据知识库回答，不要补充知识库以外的信息。"
        )
    )

    print(answer)

    print("\n" + "=" * 70)
    print("Q3")
    print("=" * 70)

    answer = agent.answer(
        question=(
            "hook - dissect - cystic_duct "
            "是什么意思？"
            "哪些信息来自知识库，"
            "哪些属于额外解释？"
        )
    )

    print(answer)


if __name__ == "__main__":
    main()