"""Verify that compact training tool results retain answer-bearing facts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.tool_call_data import compact_tool_result  # noqa: E402


def _rows(paths: list[str]):
    for value in paths:
        with Path(value).open(encoding="utf-8") as handle:
            for raw in handle:
                if raw.strip():
                    yield json.loads(raw)


def _tool_results(row: dict) -> list[dict]:
    if "messages" not in row:
        return [row["tool_result"]] if row.get("tool_result") is not None else []
    results = []
    for message in row.get("messages") or []:
        if message.get("role") != "tool":
            continue
        value = message.get("content")
        results.append(json.loads(value) if isinstance(value, str) else value)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--compact", required=True)
    parser.add_argument("--token-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    compact_rows = {
        row.get("metadata", {}).get("sample_id"): row
        for row in _rows([args.compact])
    }
    counts = Counter()
    by_type: dict[str, Counter] = {}
    issues = []
    for source in _rows(args.source):
        counts["source_samples"] += 1
        metadata = source.get("metadata") or source
        sample_id = metadata.get("sample_id")
        question_type = str(metadata.get("question_type") or metadata.get("abnormal_type") or "unknown")
        type_counts = by_type.setdefault(question_type, Counter())
        type_counts["source"] += 1
        target = compact_rows.get(sample_id)
        if target is None:
            counts["missing_compact_sample"] += 1
            type_counts["missing_compact_sample"] += 1
            issues.append({"sample_id": sample_id, "question_type": question_type, "issue": "missing_compact_sample"})
            continue
        counts["matched_samples"] += 1
        type_counts["matched"] += 1
        source_results = _tool_results(source)
        compact_results = _tool_results(target)
        if len(source_results) != len(compact_results):
            issues.append({"sample_id": sample_id, "question_type": question_type, "issue": "tool_result_count_changed"})
            type_counts["fact_mismatch"] += 1
            continue
        mismatch = False
        for original, actual in zip(source_results, compact_results):
            expected = compact_tool_result(original, question_type)
            if expected != actual:
                mismatch = True
                issues.append({"sample_id": sample_id, "question_type": question_type, "issue": "answer_fact_or_compaction_mismatch"})
                break
        if mismatch:
            type_counts["fact_mismatch"] += 1
            counts["fact_mismatch"] += 1
        else:
            type_counts["facts_preserved"] += 1
            counts["facts_preserved"] += 1

    token_audit = json.loads(Path(args.token_audit).read_text(encoding="utf-8"))
    report = {
        "schema_version": "surgical-training-compaction-audit/v1",
        "source_files": args.source,
        "compact_file": args.compact,
        "compaction_view": "surgical-agent-model-evidence/v2",
        "preserved_answer_facts": [
            "tool status and errors", "video/time/frame identity", "all triplet identities and instrument-verb-target mappings",
            "all event triplet mappings, clipped bounds, truncation flags, and original bounds when clipped", "traceable source type/dataset/version/ground-truth flag",
        ],
        "removed_non_answer_diagnostics": [
            "confidence and duplicate instance counts for human ground truth", "raw frame samples and bounding boxes", "event sample counts and repeated boundary-rule strings", "private annotation paths",
        ],
        "counts": dict(counts),
        "counts_by_question_type": {key: dict(value) for key, value in sorted(by_type.items())},
        "token_filter_counts": token_audit.get("counts"),
        "token_filter_counts_by_question_type": token_audit.get("counts_by_question_type"),
        "valid": not issues and counts["source_samples"] == len(compact_rows),
        "issues": issues[:100],
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"valid": report["valid"], "counts": report["counts"], "token_filter_counts": report["token_filter_counts"]}, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
