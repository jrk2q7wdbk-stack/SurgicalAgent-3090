"""Tiny LoRA feasibility run. It never starts full training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from importlib import metadata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.tool_call_data import build_assistant_labels, qwen_tools  # noqa: E402
from evaluation.backends import LocalModelBackend  # noqa: E402
from evaluation.protocol import SYSTEM_PROMPT, parse_final_output  # noqa: E402
from agent.tool_registry import validate_tool_arguments  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/local_lora_smoke.json")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--output-dir", default="artifacts/training_feasibility/lora_smoke")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    report = {"schema_version": "surgical-lora-smoke/v1", "status": "not_started", "config": config, "full_training": False}
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
        from peft import LoraConfig, get_peft_model, PeftModel
    except Exception as exc:
        report.update({"status": "blocked_missing_dependency", "error": f"{type(exc).__name__}: {exc}"})
        (output / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    report["dependencies"] = {
        name: metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate")
    }
    if not torch.cuda.is_available():
        report.update({"status": "blocked_no_cuda"})
        (output / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(2)
    try:
        probe = torch.tensor([1.0], device="cuda") + 1
        torch.cuda.synchronize()
        report["cuda_kernel_probe"] = {"passed": True, "value": float(probe.cpu()[0])}
        del probe
        torch.cuda.empty_cache()
    except Exception as exc:
        report.update({"status": "blocked_cuda_runtime", "cuda_kernel_probe": {"passed": False, "error": f"{type(exc).__name__}: {exc}"}, "torch": torch.__version__, "compiled_cuda": torch.version.cuda})
        (output / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    model_id = config["model"].get("local_path") or config["model"]["id"]
    local_only = not args.allow_download
    try:
        revision = config["model"].get("revision") or "main"
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=local_only)
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=torch.float16, local_files_only=local_only, low_cpu_mem_usage=True).cuda()
    except Exception as exc:
        report.update({"status": "blocked_model_unavailable", "allow_download": args.allow_download, "error": f"{type(exc).__name__}: {exc}"})
        (output / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    training = config["training"]
    report["resolved_model_revision"] = getattr(model.config, "_commit_hash", None) or config["model"].get("revision")
    report["cuda"] = {"torch": torch.__version__, "runtime": torch.version.cuda, "device": torch.cuda.get_device_name(0)}
    if training.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    peft_config = LoraConfig(r=training["lora_r"], lora_alpha=training["lora_alpha"], lora_dropout=training["lora_dropout"], target_modules=training["target_modules"], task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)

    def load(path: str, limit: int):
        candidates, filtered = [], []
        for raw in Path(path).open(encoding="utf-8"):
            if not raw.strip():
                continue
            row = json.loads(raw)
            encoded = build_assistant_labels(tokenizer, row["messages"], qwen_tools(), training["max_length"])
            if encoded.get("ok"):
                candidates.append({
                    **{key: encoded[key] for key in ("input_ids", "attention_mask", "labels")},
                    "sample_id": row.get("metadata", {}).get("sample_id"),
                    "question_type": row.get("metadata", {}).get("question_type") or row.get("metadata", {}).get("abnormal_type") or "unknown",
                })
            else:
                filtered.append({"sample_id": row.get("metadata", {}).get("sample_id"), "reason": encoded.get("status"), "details": encoded})
        # First take one item per type, then fill deterministically in file order.
        selected, selected_ids, selected_types = [], set(), set()
        for row in candidates:
            if row["question_type"] not in selected_types:
                selected.append(row)
                selected_ids.add(row["sample_id"])
                selected_types.add(row["question_type"])
                if len(selected) >= limit:
                    break
        if len(selected) < limit:
            for row in candidates:
                if row["sample_id"] in selected_ids:
                    continue
                selected.append(row)
                selected_ids.add(row["sample_id"])
                if len(selected) >= limit:
                    break
        return selected, filtered
    train_rows, train_filtered = load(args.train_data, 16)
    val_rows, val_filtered = load(args.validation_data, 4)
    if not train_rows or not val_rows:
        report.update({"status": "blocked_no_tokenized_samples", "train_filtered": train_filtered, "validation_filtered": val_filtered})
        (output / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(2)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, index):
            return {key: self.rows[index][key] for key in ("input_ids", "attention_mask", "labels")}

    def collate(batch):
        width = max(len(row["input_ids"]) for row in batch)
        pad = tokenizer.pad_token_id or tokenizer.eos_token_id
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in batch:
            amount = width - len(row["input_ids"])
            result["input_ids"].append(row["input_ids"] + [pad] * amount)
            result["attention_mask"].append(row["attention_mask"] + [0] * amount)
            result["labels"].append(row["labels"] + [-100] * amount)
        return {key: torch.tensor(value) for key, value in result.items()}

    report["selected_samples"] = {
        "train": [{"sample_id": row["sample_id"], "question_type": row["question_type"], "tokens": len(row["input_ids"])} for row in train_rows],
        "validation": [{"sample_id": row["sample_id"], "question_type": row["question_type"], "tokens": len(row["input_ids"])} for row in val_rows],
    }
    arguments = TrainingArguments(output_dir=str(output / "trainer"), per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=training["gradient_accumulation_steps"], max_steps=min(5, training["max_steps"]), learning_rate=training["learning_rate"], fp16=True, logging_steps=1, save_strategy="no", report_to=[], remove_unused_columns=False)
    trainer = Trainer(model=model, args=arguments, train_dataset=Dataset(train_rows), eval_dataset=Dataset(val_rows), data_collator=collate)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        train_result = trainer.train()
        validation = trainer.evaluate()
        trainer_history = list(trainer.state.log_history)
        nonfinite_gradient_steps = [
            entry.get("step") for entry in trainer_history
            if "grad_norm" in entry and not math.isfinite(float(entry["grad_norm"]))
        ]
        adapter = output / "adapter"
        model.save_pretrained(adapter)
        del trainer, model
        torch.cuda.empty_cache()
        base = AutoModelForCausalLM.from_pretrained(model_id, revision=report["resolved_model_revision"] or revision, torch_dtype=torch.float16, local_files_only=True, low_cpu_mem_usage=True).cuda()
        reloaded = PeftModel.from_pretrained(base, adapter)
        reloaded.eval()
        first_validation = json.loads(next(raw for raw in Path(args.validation_data).open(encoding="utf-8") if raw.strip()))
        source_messages = first_validation["messages"]
        user_message = next(item for item in source_messages if item["role"] == "user")
        reference_assistant = next((item for item in source_messages if item["role"] == "assistant" and item.get("tool_calls")), None)
        reference_tool = next((item for item in source_messages if item["role"] == "tool"), None)
        closure_messages = [{"role": "system", "content": SYSTEM_PROMPT}, user_message]

        def generate_once(messages):
            prompt = tokenizer.apply_chat_template(messages, tools=qwen_tools(), tokenize=False, add_generation_prompt=True)
            tokens = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = reloaded.generate(**tokens, max_new_tokens=192, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            return tokenizer.decode(generated[0][tokens["input_ids"].shape[1]:], skip_special_tokens=False).replace("<|im_end|>", "").strip()

        first_output = generate_once(closure_messages)
        generated_calls, call_parse_error = LocalModelBackend._parse_tool_calls(first_output)
        closure = {"question": user_message["content"], "first_output": first_output, "generated_tool_calls": generated_calls, "tool_call_parse_error": call_parse_error, "tool_executed": False}
        if generated_calls:
            call = generated_calls[0]
            validation_error = validate_tool_arguments(call["name"], call["arguments"])
            closure["tool_validation_error"] = validation_error
            reference_call = (reference_assistant.get("tool_calls") or [])[0]["function"] if reference_assistant else None
            matches_reference = bool(reference_call and call["name"] == reference_call.get("name") and call["arguments"] == reference_call.get("arguments"))
            if validation_error:
                executed = validation_error
            elif matches_reference and reference_tool:
                executed = json.loads(reference_tool["content"])
            else:
                executed = {"ok": False, "status": "error", "error": {"code": "ANNOTATION_SIMULATOR_MISS", "message": "generated call has no matching frozen annotation evidence"}}
            closure["tool_executed"] = True
            closure["tool_result"] = executed
            closure_messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": "smoke_call", "type": "function", "function": call}]})
            closure_messages.append({"role": "tool", "tool_call_id": "smoke_call", "name": call["name"], "content": json.dumps(executed, ensure_ascii=False)})
            final_output = generate_once(closure_messages)
            visible, structured, parse_error = parse_final_output(final_output)
            closure.update({"final_output": final_output, "visible_answer": visible, "structured_answer": structured, "final_parse_error": parse_error})
        smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=False)
        final_status = "completed_with_numerical_warning" if nonfinite_gradient_steps else "completed"
        report.update({"status": final_status, "steps": train_result.global_step, "train_metrics": train_result.metrics, "validation_metrics": validation, "trainer_log_history": trainer_history, "numerical_warnings": {"nonfinite_gradient_norm_steps": nonfinite_gradient_steps, "classification": "fp16_numerical" if nonfinite_gradient_steps else None}, "elapsed_seconds": time.perf_counter() - started, "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved() / 2**20, "peak_memory_note": "allocated is live tensor memory; reserved is allocator/WDDM virtual reservation and can exceed physical VRAM", "nvidia_smi_after": smi.stdout.strip(), "adapter": str(adapter), "reload_tool_closure": closure, "train_filtered": train_filtered, "validation_filtered": val_filtered, "selected_train_samples": len(train_rows), "selected_validation_samples": len(val_rows)})
    except torch.cuda.OutOfMemoryError as exc:
        report.update({"status": "oom", "error_category": "gpu_memory", "error": str(exc), "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved() / 2**20})
    except (FloatingPointError, ArithmeticError) as exc:
        report.update({"status": "numerical_error", "error_category": "numerical", "error": f"{type(exc).__name__}: {exc}"})
    except Exception as exc:
        report.update({"status": "failed", "error_category": "runtime_or_dependency", "error": f"{type(exc).__name__}: {exc}", "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved() / 2**20})
    (output / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if report["status"] not in {"completed", "completed_with_numerical_warning"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
