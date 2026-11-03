"""Reproduce FP16 scaler behavior and verify finite LoRA updates.

This is a bounded diagnostic, not a training entry point. Every invocation loads
the frozen base model and initializes a new LoRA adapter.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.tool_call_data import build_assistant_labels, qwen_tools  # noqa: E402


def _selected_ids(report_path: Path) -> list[str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return [item["sample_id"] for item in report["selected_samples"]["train"]]


def _load_rows(path: Path, wanted: list[str], tokenizer, max_length: int) -> list[dict]:
    indexed = {}
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            sample_id = row.get("metadata", {}).get("sample_id")
            if sample_id not in wanted:
                continue
            encoded = build_assistant_labels(tokenizer, row["messages"], qwen_tools(), max_length)
            if not encoded.get("ok"):
                raise RuntimeError(f"selected sample {sample_id} is not usable: {encoded}")
            indexed[sample_id] = {
                **{key: encoded[key] for key in ("input_ids", "attention_mask", "labels")},
                "sample_id": sample_id,
                "question_type": row.get("metadata", {}).get("question_type") or row.get("metadata", {}).get("abnormal_type") or "unknown",
                "token_count": encoded["token_count"],
            }
    missing = [sample_id for sample_id in wanted if sample_id not in indexed]
    if missing:
        raise RuntimeError(f"selected sample ids missing: {missing}")
    return [indexed[sample_id] for sample_id in wanted]


def _finite_gradients(parameters) -> tuple[bool, int, int]:
    finite, values, nonfinite = True, 0, 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        values += parameter.grad.numel()
        bad = int((~parameter.grad.isfinite()).sum().item())
        nonfinite += bad
        finite = finite and bad == 0
    return finite, values, nonfinite


def _finite_parameters(parameters) -> bool:
    return all(bool(parameter.detach().isfinite().all().cpu()) for parameter in parameters)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--selection-report", required=True)
    parser.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--initial-scale", type=float, default=65536.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = _load_rows(Path(args.train_data), _selected_ids(Path(args.selection_report)), tokenizer, args.max_length)
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, local_files_only=True, low_cpu_mem_usage=True).cuda()
    base.gradient_checkpointing_enable()
    base.config.use_cache = False
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(init_scale=args.initial_scale, growth_interval=2000)

    dtype_counts = {}
    for name, parameter in model.named_parameters():
        bucket = "lora_trainable" if parameter.requires_grad else "base_frozen"
        dtype_counts.setdefault(bucket, {})[str(parameter.dtype)] = dtype_counts.setdefault(bucket, {}).get(str(parameter.dtype), 0) + parameter.numel()

    order = []
    generator = torch.Generator().manual_seed(42)
    while len(order) < args.steps * args.gradient_accumulation:
        order.extend(torch.randperm(len(rows), generator=generator).tolist())
    order = order[:args.steps * args.gradient_accumulation]
    step_logs = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        micro_logs = []
        before = [parameter.detach().clone() for parameter in trainable]
        for offset in range(args.gradient_accumulation):
            row = rows[order[(step - 1) * args.gradient_accumulation + offset]]
            batch = {
                key: torch.tensor([row[key]], device="cuda")
                for key in ("input_ids", "attention_mask", "labels")
            }
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = model(**batch).loss / args.gradient_accumulation
            scaler.scale(loss).backward()
            scaled_finite, grad_values, scaled_nonfinite = _finite_gradients(trainable)
            micro_logs.append({
                "sample_id": row["sample_id"], "question_type": row["question_type"], "token_count": row["token_count"],
                "loss_before_accumulation_division": float(loss.detach().cpu()) * args.gradient_accumulation,
                "scaled_gradients_finite_after_microbatch": scaled_finite,
                "scaled_nonfinite_gradient_values": scaled_nonfinite,
                "gradient_values": grad_values,
            })
        scale_before = float(scaler.get_scale())
        scaler.unscale_(optimizer)
        unscaled_finite, grad_values, unscaled_nonfinite = _finite_gradients(trainable)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        changed_tensors = 0
        max_abs_update = 0.0
        all_updates_finite = True
        for previous, current in zip(before, trainable):
            delta = current.detach() - previous
            if bool(torch.count_nonzero(delta).cpu()):
                changed_tensors += 1
            max_abs_update = max(max_abs_update, float(delta.abs().max().cpu()))
            all_updates_finite = all_updates_finite and bool(delta.isfinite().all().cpu())
        update_skipped = changed_tensors == 0
        step_logs.append({
            "step": step,
            "microbatches": micro_logs,
            "scale_before": scale_before,
            "scale_after": scale_after,
            "unscaled_gradients_finite": unscaled_finite,
            "unscaled_nonfinite_gradient_values": unscaled_nonfinite,
            "gradient_values": grad_values,
            "gradient_norm": float(grad_norm.detach().cpu()),
            "optimizer_update_skipped": update_skipped,
            "changed_lora_tensors": changed_tensors,
            "max_abs_lora_update": max_abs_update,
            "lora_parameters_finite_after_step": _finite_parameters(trainable),
            "lora_updates_finite": all_updates_finite,
        })
        optimizer.zero_grad(set_to_none=True)

    report = {
        "schema_version": "surgical-fp16-lora-diagnostic/v1",
        "fresh_base_model_each_run": True,
        "model": args.model,
        "config": vars(args),
        "parameter_dtypes": dtype_counts,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "selected_samples": [{"sample_id": row["sample_id"], "question_type": row["question_type"], "token_count": row["token_count"]} for row in rows],
        "step_logs": step_logs,
        "summary": {
            "nonfinite_gradient_steps": [item["step"] for item in step_logs if not item["unscaled_gradients_finite"]],
            "skipped_update_steps": [item["step"] for item in step_logs if item["optimizer_update_skipped"]],
            "updated_steps": [item["step"] for item in step_logs if not item["optimizer_update_skipped"]],
            "all_lora_parameters_always_finite": all(item["lora_parameters_finite_after_step"] for item in step_logs),
            "all_applied_updates_finite": all(item["lora_updates_finite"] for item in step_logs),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"config": report["config"], "parameter_dtypes": dtype_counts, "summary": report["summary"], "steps": [{key: item[key] for key in ("step", "scale_before", "scale_after", "gradient_norm", "optimizer_update_skipped", "changed_lora_tensors", "max_abs_lora_update")} for item in step_logs]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
