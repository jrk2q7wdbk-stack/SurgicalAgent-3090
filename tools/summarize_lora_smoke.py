"""Create a compact, reviewable summary from the full LoRA smoke report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    history = source.get("trainer_log_history") or []
    nonfinite_steps = [
        item.get("step") for item in history
        if "grad_norm" in item and not math.isfinite(float(item["grad_norm"]))
    ]
    closure = source.get("reload_tool_closure") or {}
    effective_status = source.get("status")
    if effective_status == "completed" and nonfinite_steps:
        effective_status = "completed_with_numerical_warning"
    summary = {
        "schema_version": "surgical-lora-smoke-summary/v1",
        "source_report": str(Path(args.input).resolve()),
        "source_status": source.get("status"),
        "effective_status": effective_status,
        "full_training": source.get("full_training"),
        "steps": source.get("steps"),
        "dependencies": source.get("dependencies"),
        "cuda": source.get("cuda"),
        "resolved_model_revision": source.get("resolved_model_revision"),
        "train_metrics": source.get("train_metrics"),
        "validation_metrics": source.get("validation_metrics"),
        "per_step_log": history,
        "numerical": {
            "nonfinite_gradient_norm_steps": nonfinite_steps,
            "classification": "fp16_numerical_warning" if nonfinite_steps else "no_nonfinite_gradient_norm_observed",
            "training_loss_finite": math.isfinite(float((source.get("train_metrics") or {}).get("train_loss", float("nan")))),
            "validation_loss_finite": math.isfinite(float((source.get("validation_metrics") or {}).get("eval_loss", float("nan")))),
        },
        "memory": {
            "peak_allocated_mib": source.get("peak_gpu_allocated_mib"),
            "peak_reserved_mib": source.get("peak_gpu_reserved_mib"),
            "note": "Allocated is live tensor memory. Reserved is the PyTorch allocator/WDDM virtual reservation and can exceed physical VRAM.",
        },
        "adapter": source.get("adapter"),
        "tool_closure": {
            "tool_call_parsed": bool(closure.get("generated_tool_calls")) and not closure.get("tool_call_parse_error"),
            "generated_tool_calls": closure.get("generated_tool_calls"),
            "tool_executed": closure.get("tool_executed"),
            "tool_result_status": (closure.get("tool_result") or {}).get("status"),
            "final_visible_answer_generated": bool(closure.get("visible_answer")),
            "final_structured_answer_generated": closure.get("structured_answer") is not None,
            "final_parse_error": closure.get("final_parse_error"),
            "task_closure_success": bool(closure.get("structured_answer")) and (closure.get("tool_result") or {}).get("ok") is True,
        },
        "interpretation": "This proves the five-step pipeline executed. It does not establish task learning or performance improvement.",
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
