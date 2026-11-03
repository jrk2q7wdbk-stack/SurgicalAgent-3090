from tools.vision_tool import (
    analyze_surgical_image
)

from tools.knowledge_tool import (
    get_triplet_knowledge
)

from agent.confidence import (
    build_grounded_visual_answer
)


class SurgicalAgent:

    def analyze(
        self,
        image_path: str
    ) -> dict:

        vision_result = (
            analyze_surgical_image(
                image_path
            )
        )

        best_triplet = (
            vision_result.get(
                "best_triplet"
            )
        )

        knowledge = {}

        if best_triplet is not None:

            knowledge = (
                get_triplet_knowledge(
                    best_triplet[
                        "instrument"
                    ],
                    best_triplet[
                        "verb"
                    ],
                    best_triplet[
                        "target"
                    ]
                )
            )

        return {
            "vision": vision_result,
            "knowledge": knowledge
        }


    def answer(
        self,
        question: str,
        image_path: str
    ) -> str:

        result = self.analyze(
            image_path
        )

        vision = result[
            "vision"
        ]

        best = vision.get(
            "best_triplet"
        )

        if best is None:
            return (
                "当前图像中没有识别到"
                "明确的手术操作三元组。"
            )

        return build_grounded_visual_answer(
            question=question,
            triplet=best
        )
