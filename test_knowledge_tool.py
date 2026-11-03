import json

from tools.knowledge_tool import (
    get_triplet_knowledge,
    get_knowledge_base_info
)


def main():

    print("=" * 70)
    print("Knowledge Base Info")
    print("=" * 70)

    info = get_knowledge_base_info()

    print(
        json.dumps(
            info,
            ensure_ascii=False,
            indent=2
        )
    )

    print("\n" + "=" * 70)
    print("Test 1: grasper-grasp-gallbladder")
    print("=" * 70)

    result = get_triplet_knowledge(
        instrument="grasper",
        verb="grasp",
        target="gallbladder"
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )
    )

    print("\n" + "=" * 70)
    print("Test 2: hook-dissect-cystic_duct")
    print("=" * 70)

    result = get_triplet_knowledge(
        instrument="hook",
        verb="dissect",
        target="cystic_duct"
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )
    )

    print("\n" + "=" * 70)
    print("Test 3: irrigator-aspirate-fluid")
    print("=" * 70)

    result = get_triplet_knowledge(
        instrument="irrigator",
        verb="aspirate",
        target="fluid"
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )
    )


if __name__ == "__main__":
    main()