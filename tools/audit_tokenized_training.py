"""Tokenize every local-training sample and audit masks/pairing/length."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.tool_call_data import build_assistant_labels, qwen_tools, validate_conversation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--input", required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--output", default="artifacts/stage5/tokenized_data_audit.json")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
    counts = Counter()
    by_question_type: dict[str, Counter] = {}
    lengths_by_type: dict[str, list[int]] = {}
    accepted_lengths_by_type: dict[str, list[int]] = {}
    lengths = []
    role_tokens = Counter()
    role_supervised = Counter()
    issues = []
    with Path(args.input).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            counts["total"] += 1
            row = json.loads(raw)
            question_type = str(row.get("metadata", {}).get("question_type") or row.get("metadata", {}).get("abnormal_type") or "unknown")
            type_counts = by_question_type.setdefault(question_type, Counter())
            type_counts["total"] += 1
            pairing = validate_conversation(row.get("messages") or [])
            if pairing:
                counts["invalid_pairing"] += 1
                type_counts["invalid_pairing"] += 1
                issues.append({"line": line_number, "sample_id": row.get("metadata", {}).get("sample_id"), "errors": pairing})
                continue
            encoded = build_assistant_labels(tokenizer, row["messages"], qwen_tools(), args.max_length)
            if not encoded.get("ok"):
                counts[encoded.get("status", "unknown_failure")] += 1
                type_counts[encoded.get("status", "unknown_failure")] += 1
                issues.append({"line": line_number, "sample_id": row.get("metadata", {}).get("sample_id"), "status": encoded.get("status"), "token_count": encoded.get("token_count")})
                if encoded.get("token_count") is not None:
                    lengths_by_type.setdefault(question_type, []).append(int(encoded["token_count"]))
                continue
            counts["accepted"] += 1
            type_counts["accepted"] += 1
            lengths.append(encoded["token_count"])
            lengths_by_type.setdefault(question_type, []).append(encoded["token_count"])
            accepted_lengths_by_type.setdefault(question_type, []).append(encoded["token_count"])
            role_tokens.update(encoded["role_token_counts"])
            role_supervised.update(encoded["role_supervised_counts"])
            if encoded["assistant_token_count"] <= 0:
                counts["no_supervised_tokens"] += 1
    non_assistant_supervised = sum(value for role, value in role_supervised.items() if role != "assistant")
    def distribution(values: list[int]) -> dict:
        if not values:
            return {"count": 0, "min": None, "p50": None, "p95": None, "max": None, "mean": None}
        ordered = sorted(values)
        return {
            "count": len(values), "min": ordered[0],
            "p50": ordered[math.ceil(.50 * len(ordered)) - 1],
            "p95": ordered[math.ceil(.95 * len(ordered)) - 1],
            "max": ordered[-1], "mean": sum(ordered) / len(ordered),
        }
    report = {
        "schema_version": "surgical-tokenized-data-audit/v1",
        "model": args.model, "tokenizer_revision": getattr(tokenizer, "_commit_hash", None),
        "input": str(Path(args.input).resolve()), "max_length": args.max_length,
        "counts": dict(counts),
        "counts_by_question_type": {key: dict(value) for key, value in sorted(by_question_type.items())},
        "length_distribution_by_question_type": {
            key: {"all": distribution(values), "accepted": distribution(accepted_lengths_by_type.get(key, []))}
            for key, values in sorted(lengths_by_type.items())
        },
        "lengths": {"min": min(lengths) if lengths else None, "max": max(lengths) if lengths else None, "mean": sum(lengths) / len(lengths) if lengths else None},
        "role_token_counts": dict(role_tokens), "role_supervised_counts": dict(role_supervised),
        "non_assistant_supervised_tokens": non_assistant_supervised,
        "mask_valid": non_assistant_supervised == 0 and counts["no_supervised_tokens"] == 0,
        "issues_count": len(issues), "issues": issues,
    }
    output = Path(args.output)
    if not output.is_absolute(): output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("model", "max_length", "counts", "lengths", "role_supervised_counts", "non_assistant_supervised_tokens", "mask_valid", "issues_count")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
