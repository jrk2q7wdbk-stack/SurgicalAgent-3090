"""Generate grounded question-answer pairs from RDV labels or predictions."""

import argparse
import json
import os
import sys

try:
    from agent.confidence import build_grounded_visual_answer, enrich_triplet
except ModuleNotFoundError:
    # Support both ``python -m tools.qa_generator`` and direct script usage.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.confidence import build_grounded_visual_answer, enrich_triplet

from data.qa_pipeline import GenerationConfig, generate_dataset


QUESTION_TEMPLATES = (
    ("triplet", "当前RDV识别到的手术三元组是什么？"),
    ("instrument", "当前使用的主要器械是什么？"),
    ("verb", "当前识别到的手术动作是什么？"),
    ("target", "当前操作作用于什么目标组织？"),
)


def generate_qa_pairs_from_prediction(vision_result, source_id=None, timestamp=None):
    """Create confidence-aware QA pairs directly from one RDV output."""

    best = (vision_result or {}).get("best_triplet")
    if not best:
        return [
            {
                "source_id": source_id,
                "timestamp": timestamp,
                "question_type": "triplet",
                "question": QUESTION_TEMPLATES[0][1],
                "answer": "当前RDV模型没有返回可用的手术三元组，无法作出确定判断。",
                "answerable": False,
                "evidence": None,
                "evidence_source": "rdv_prediction",
            }
        ]

    evidence = enrich_triplet(best)
    answerable = evidence["confidence_level"] != "low"
    pairs = []
    for question_type, question in QUESTION_TEMPLATES:
        pairs.append(
            {
                "source_id": source_id,
                "timestamp": timestamp,
                "question_type": question_type,
                "question": question,
                "answer": build_grounded_visual_answer(question, evidence),
                "answerable": answerable,
                "evidence": evidence,
                "evidence_source": "rdv_prediction",
            }
        )
    return pairs


def generate_qa_pairs_from_labels(labels):
    """Create label-grounded QA pairs for every class in triplet_classes.json."""

    pairs = []
    for class_id, label in enumerate(labels):
        triplet = {
            "id": class_id,
            "instrument": label["instrument"],
            "verb": label["verb"],
            "target": label["target"],
            "confidence": 1.0,
        }
        evidence = enrich_triplet(triplet)
        evidence["evidence_type"] = "label_definition"
        answers = {
            "triplet": (
                f"IVT class {class_id} 对应的三元组是 "
                f"{label['instrument']} - {label['verb']} - {label['target']}。"
            ),
            "instrument": f"该标签中的器械是 {label['instrument']}。",
            "verb": f"该标签中的动作是 {label['verb']}。",
            "target": f"该标签中的目标组织是 {label['target']}。",
        }
        for question_type, base_question in QUESTION_TEMPLATES:
            pairs.append(
                {
                    "source_id": f"ivt_class_{class_id}",
                    "timestamp": None,
                    "question_type": question_type,
                    "question": (
                        f"IVT class {class_id}：{base_question}"
                    ),
                    "answer": answers[question_type],
                    "answerable": True,
                    "evidence": evidence,
                    "class_id": class_id,
                    "label_source": "class_definition",
                    "evidence_source": "class_definition",
                }
            )
    return pairs


def write_jsonl(records, output_path):
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", help="legacy triplet_classes.json path")
    parser.add_argument("--output", help="legacy class-definition JSONL path")
    parser.add_argument("--dataset-root", help="CholecT50 Release 2.0 root")
    parser.add_argument("--output-dir", help="human-annotation dataset output directory")
    parser.add_argument(
        "--split-variant",
        choices=("rdv_original", "challenge", "crossval"),
        default="rdv_original",
    )
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--max-point-frames-per-video", type=int, default=40)
    parser.add_argument("--max-per-signature", type=int, default=2)
    parser.add_argument("--range-windows", type=int, nargs="+", default=[10, 30])
    parser.add_argument("--range-stride", type=int, default=60)
    parser.add_argument("--max-ranges-per-video", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.dataset_root:
        if not args.output_dir:
            parser.error("--dataset-root 模式必须提供 --output-dir")
        manifest = generate_dataset(
            args.dataset_root,
            args.output_dir,
            GenerationConfig(
                split_variant=args.split_variant,
                fold=args.fold,
                max_point_frames_per_video=args.max_point_frames_per_video,
                max_per_signature=args.max_per_signature,
                range_windows=tuple(args.range_windows),
                range_stride=args.range_stride,
                max_ranges_per_video=args.max_ranges_per_video,
                seed=args.seed,
            ),
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    if not args.labels or not args.output:
        parser.error("类别定义模式需要 --labels 和 --output")

    with open(args.labels, "r", encoding="utf-8") as handle:
        labels = json.load(handle)
    records = generate_qa_pairs_from_labels(labels)
    write_jsonl(records, args.output)
    print(f"generated {len(records)} QA pairs -> {args.output}")


if __name__ == "__main__":
    main()
