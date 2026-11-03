"""Bounded preflight: diverse stability run plus small-sample learning check."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.backends import LocalModelBackend  # noqa: E402
from training.tool_call_data import build_assistant_labels, qwen_tools  # noqa: E402

NORMAL_TYPES = ["instrument", "action", "target", "triplet", "time_point_operation", "time_range_summary", "continuous_event"]


def _scan(path: Path, tokenizer, max_length: int) -> dict[str, list[tuple[int, str]]]:
    groups = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            encoded = build_assistant_labels(tokenizer, row["messages"], qwen_tools(), max_length)
            if not encoded.get("ok"):
                raise RuntimeError(f"preflight requires all samples to fit: {row.get('metadata', {}).get('sample_id')} {encoded}")
            metadata = row.get("metadata") or {}
            question_type = metadata.get("question_type") or metadata.get("abnormal_type") or "unknown"
            groups[question_type].append((encoded["token_count"], metadata.get("sample_id")))
    return groups


def _choose(groups: dict[str, list[tuple[int, str]]], modes=("min", "median", "max")) -> list[str]:
    selected = []
    for question_type in sorted(groups):
        values = sorted(groups[question_type])
        indices = {"min": 0, "median": len(values) // 2, "max": len(values) - 1}
        for mode in modes:
            sample_id = values[indices[mode]][1]
            if sample_id not in selected:
                selected.append(sample_id)
    return selected


def _normal_medians(groups: dict[str, list[tuple[int, str]]]) -> list[str]:
    return [sorted(groups[name])[len(groups[name]) // 2][1] for name in NORMAL_TYPES]


def _load(path: Path, ids: list[str], tokenizer, max_length: int) -> list[dict]:
    wanted, indexed = set(ids), {}
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            metadata = row.get("metadata") or {}
            sample_id = metadata.get("sample_id")
            if sample_id not in wanted:
                continue
            encoded = build_assistant_labels(tokenizer, row["messages"], qwen_tools(), max_length)
            indexed[sample_id] = {"sample_id": sample_id, "question_type": metadata.get("question_type") or metadata.get("abnormal_type") or "unknown", "messages": row["messages"], **encoded}
    return [indexed[sample_id] for sample_id in ids]


def _batch(row: dict, torch) -> dict:
    return {key: torch.tensor([row[key]], device="cuda") for key in ("input_ids", "attention_mask", "labels")}


def _losses(model, rows, torch) -> dict:
    model.eval()
    values = []
    with torch.inference_mode():
        for row in rows:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = model(**_batch(row, torch)).loss
            values.append({"sample_id": row["sample_id"], "question_type": row["question_type"], "tokens": row["token_count"], "loss": float(loss.cpu())})
    model.train()
    return {"mean": sum(item["loss"] for item in values) / len(values), "samples": values}


def _tool_generation(model, tokenizer, rows, torch) -> dict:
    model.eval()
    details = []
    for row in rows:
        messages = row["messages"][:2]
        reference_message = next(item for item in row["messages"] if item.get("role") == "assistant" and item.get("tool_calls"))
        reference_function = reference_message["tool_calls"][0]["function"]
        reference = {"name": reference_function["name"], "arguments": reference_function.get("arguments") or {}}
        prompt = tokenizer.apply_chat_template(messages, tools=qwen_tools(), tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False).replace("<|im_end|>", "").strip()
        calls, parse_error = LocalModelBackend._parse_tool_calls(text)
        actual = calls[0] if calls else None
        details.append({"sample_id": row["sample_id"], "question_type": row["question_type"], "reference": reference, "actual": actual, "parse_error": parse_error, "exact": actual == reference, "raw": text})
    model.train()
    return {"correct": sum(item["exact"] for item in details), "total": len(details), "details": details}


def _train_phase(model, optimizer, scaler, trainable, rows, torch, steps: int, accumulation: int, learning_rate: float, start_index: int = 0) -> tuple[list[dict], int]:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    logs, index = [], start_index
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, steps + 1):
        before = [parameter.detach().clone() for parameter in trainable]
        samples, losses = [], []
        for _ in range(accumulation):
            row = rows[index % len(rows)]
            index += 1
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = model(**_batch(row, torch)).loss / accumulation
            scaler.scale(loss).backward()
            samples.append({"sample_id": row["sample_id"], "question_type": row["question_type"], "tokens": row["token_count"]})
            losses.append(float(loss.detach().cpu()) * accumulation)
        scale_before = float(scaler.get_scale())
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        gradients_finite = all(parameter.grad is None or bool(parameter.grad.isfinite().all().cpu()) for parameter in trainable)
        scaler.step(optimizer)
        scaler.update()
        changed = sum(bool(torch.count_nonzero(parameter.detach() - previous).cpu()) for previous, parameter in zip(before, trainable))
        params_finite = all(bool(parameter.detach().isfinite().all().cpu()) for parameter in trainable)
        logs.append({"step": step, "samples": samples, "mean_microbatch_loss": sum(losses) / len(losses), "scale_before": scale_before, "scale_after": float(scaler.get_scale()), "gradient_norm": float(grad_norm.detach().cpu()), "gradients_finite": gradients_finite, "update_skipped": changed == 0, "changed_lora_tensors": changed, "lora_parameters_finite": params_finite})
        optimizer.zero_grad(set_to_none=True)
    return logs, index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--stability-steps", type=int, default=6)
    parser.add_argument("--learning-steps", type=int, default=28)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "surgical-lora-preflight/v1", "status": "started", "fresh_base_model": True, "config": vars(args)}
    try:
        set_seed(20261102)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        train_groups = _scan(Path(args.train_data), tokenizer, args.max_length)
        validation_groups = _scan(Path(args.validation_data), tokenizer, args.max_length)
        stability_ids = _choose(train_groups)
        # Stress long contexts first, while retaining every type and length tier.
        stability_rows = sorted(_load(Path(args.train_data), stability_ids, tokenizer, args.max_length), key=lambda row: row["token_count"], reverse=True)
        learning_rows = _load(Path(args.train_data), _normal_medians(train_groups), tokenizer, args.max_length)
        independent_rows = _load(Path(args.validation_data), _normal_medians(validation_groups), tokenizer, args.max_length)

        base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, local_files_only=True, low_cpu_mem_usage=True).cuda()
        base.gradient_checkpointing_enable()
        base.config.use_cache = False
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
        model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
        model.train()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=5e-5)
        scaler = torch.cuda.amp.GradScaler(init_scale=8192.0, growth_interval=2000)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()

        stability_logs, index = _train_phase(model, optimizer, scaler, trainable, stability_rows, torch, args.stability_steps, 4, 5e-5)
        before_learning = {"train_loss": _losses(model, learning_rows, torch), "validation_loss": _losses(model, independent_rows, torch), "train_tool_generation": _tool_generation(model, tokenizer, learning_rows, torch), "validation_tool_generation": _tool_generation(model, tokenizer, independent_rows, torch)}
        learning_logs, _ = _train_phase(model, optimizer, scaler, trainable, learning_rows, torch, args.learning_steps, 1, 1e-4, index)
        after_learning = {"train_loss": _losses(model, learning_rows, torch), "validation_loss": _losses(model, independent_rows, torch), "train_tool_generation": _tool_generation(model, tokenizer, learning_rows, torch), "validation_tool_generation": _tool_generation(model, tokenizer, independent_rows, torch)}
        adapter = output / "adapter"
        model.save_pretrained(adapter)

        all_logs = stability_logs + learning_logs
        report.update({
            "status": "completed" if all(item["gradients_finite"] and item["lora_parameters_finite"] and not item["update_skipped"] for item in all_logs) else "completed_with_stability_failure",
            "resolved_model_revision": getattr(model.config, "_commit_hash", None),
            "parameter_precision": {"base": "float16", "lora": str(trainable[0].dtype)},
            "stability_selection": [{"sample_id": row["sample_id"], "question_type": row["question_type"], "tokens": row["token_count"]} for row in stability_rows],
            "learning_selection": [{"sample_id": row["sample_id"], "question_type": row["question_type"], "tokens": row["token_count"]} for row in learning_rows],
            "independent_validation_selection": [{"sample_id": row["sample_id"], "question_type": row["question_type"], "tokens": row["token_count"]} for row in independent_rows],
            "stability_logs": stability_logs, "learning_logs": learning_logs,
            "before_learning": before_learning, "after_learning": after_learning,
            "summary": {
                "stability_nonfinite_steps": [item["step"] for item in stability_logs if not item["gradients_finite"]],
                "stability_skipped_steps": [item["step"] for item in stability_logs if item["update_skipped"]],
                "learning_nonfinite_steps": [item["step"] for item in learning_logs if not item["gradients_finite"]],
                "learning_skipped_steps": [item["step"] for item in learning_logs if item["update_skipped"]],
                "train_loss_before": before_learning["train_loss"]["mean"], "train_loss_after": after_learning["train_loss"]["mean"],
                "validation_loss_before": before_learning["validation_loss"]["mean"], "validation_loss_after": after_learning["validation_loss"]["mean"],
                "train_tool_before": {key: before_learning["train_tool_generation"][key] for key in ("correct", "total")},
                "train_tool_after": {key: after_learning["train_tool_generation"][key] for key in ("correct", "total")},
                "validation_tool_before": {key: before_learning["validation_tool_generation"][key] for key in ("correct", "total")},
                "validation_tool_after": {key: after_learning["validation_tool_generation"][key] for key in ("correct", "total")},
            },
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "elapsed_seconds": time.perf_counter() - started,
            "adapter": str(adapter),
            "interpretation": "Bounded preflight only; not full training and not a test-set result.",
        })
    except torch.cuda.OutOfMemoryError as exc:
        report.update({"status": "oom", "error_category": "gpu_memory", "error": str(exc), "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20})
    except Exception as exc:
        report.update({"status": "failed", "error_category": "runtime_or_dependency", "error": f"{type(exc).__name__}: {exc}"})
    (output / "preflight_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report.get(key) for key in ("status", "summary", "peak_allocated_mib", "peak_reserved_mib", "elapsed_seconds", "adapter", "error_category", "error")}, ensure_ascii=False, indent=2))
    if report["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
