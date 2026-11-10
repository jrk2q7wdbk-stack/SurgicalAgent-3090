import os

from agent.agent import (
    SurgicalAgent
)


if __name__ == "__main__":

    project_root = os.path.dirname(
        os.path.abspath(__file__)
    )

    image_path = os.path.join(
        project_root,
        "test.jpg"
    )

    agent = SurgicalAgent()

    questions = [
        "当前使用的是什么器械？",
        "医生现在在做什么？",
        "作用于什么组织？",
        "当前三元组是什么？"
    ]

    for question in questions:

        print("=" * 60)

        print(
            "Question:",
            question
        )

        answer = agent.answer(
            question=question,
            image_path=image_path
        )

        print(
            "Answer:",
            answer
        )