"""First formal LoRA training round with frozen-test safeguards."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.backends import AnnotationToolSimulator, LocalModelBackend  # noqa: E402
from evaluation.runner import evaluate, read_jsonl, stratified_subset, write_evaluation  # noqa: E402
from training.token_cache import TokenCacheDataset, build_token_cache  # noqa: E402
from training.tool_call_data import qwen_tools  # noqa: E402


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _batch(row: dict, torch) -> dict:
    ids = torch.as_tensor(row["input_ids"], dtype=torch.long, device="cuda").unsqueeze(0)
    labels = torch.as_tensor(row["labels"], dtype=torch.long, device="cuda").unsqueeze(0)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def _all_finite(parameters) -> bool:
    return all(bool(parameter.detach().isfinite().all().cpu()) for parameter in parameters)


def _save_checkpoint(root: Path, model, optimizer, scheduler, scaler, state: dict, torch) -> Path:
    checkpoint = root / f"checkpoint_step_{state['optimizer_step']:06d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint / "adapter")
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "state": state}, checkpoint / "training_state.pt")
    _write_json(checkpoint / "state.json", state)
    return checkpoint


def _validation_loss(model, dataset, torch) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.inference_mode():
        for index in range(len(dataset)):
            row = dataset.row(index)
            supervised = int((row["labels"] != -100).sum())
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = model(**_batch(row, torch)).loss
            total_loss += float(loss.cpu()) * supervised
            total_tokens += supervised
    model.train()
    return total_loss / total_tokens


def _in_memory_backend(model, tokenizer, executor, adapter_label: str, max_rounds: int, max_new_tokens: int):
    backend = LocalModelBackend.__new__(LocalModelBackend)
    backend.model_path = "models/Qwen2.5-0.5B-Instruct"
    backend.executor = executor
    backend.evidence_backend = "annotation_simulator"
    backend.adapter_path = adapter_label
    backend.max_rounds = max_rounds
    backend.max_new_tokens = max_new_tokens
    backend.backend_id = f"formal-round1-in-memory/{adapter_label}"
    backend.tokenizer = tokenizer
    backend.model = model
    backend.device = "cuda"
    backend.load_error = None
    backend.config = {"protocol_version": "surgical-agent-output/v2", "model_backend": "local_transformers_training_checkpoint", "evidence_backend": "annotation_simulator", "adapter_path": adapter_label, "device": "cuda", "max_new_tokens": max_new_tokens, "max_tool_rounds": max_rounds}
    return backend


def _evaluate_checkpoint(output: Path, epoch: int, model, tokenizer, validation_dataset, validation_rows: list[dict], config: dict, torch) -> tuple[dict, float]:
    validation_loss = _validation_loss(model, validation_dataset, torch)
    subset = stratified_subset(list(validation_rows), config["validation"]["generation_subset_limit"])
    simulator = AnnotationToolSimulator(validation_rows)
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    backend = _in_memory_backend(model, tokenizer, simulator, f"epoch-{epoch}", config["validation"]["max_tool_rounds"], config["validation"]["max_new_tokens"])
    results, summary, review = evaluate(subset, backend)
    write_evaluation(output / f"validation_epoch_{epoch}", results, summary, review)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.train()
    summary["teacher_forced_validation_loss"] = validation_loss
    _write_json(output / f"validation_epoch_{epoch}" / "summary.json", summary)
    return summary, validation_loss


def _metric_tuple(summary: dict, validation_loss: float) -> tuple:
    return (
        float(summary["joint_tool_parameter_accuracy"]["value"] or 0.0),
        float(summary["answer_scoring"]["automatic_coverage"] or 0.0),
        float(summary["answer_scoring"]["provisional_complete_answer_accuracy"] or 0.0),
        float(summary["overall_task_success_rate"]["value"] or 0.0),
        -float(validation_loss),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/local_lora_round1.json")
    parser.add_argument("--confirm-full-training", action="store_true")
    args = parser.parse_args()
    if not args.confirm_full_training:
        raise SystemExit("Refusing to start formal training without --confirm-full-training")
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if "test" in json.dumps(config.get("data", {}), ensure_ascii=False).lower() and config["data"].get("test_policy") != "frozen and not accepted by this runner":
        raise SystemExit("Test data is forbidden in the training runner")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

    output = ROOT / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    event_log = output / "training_log.jsonl"
    report = {"schema_version": "surgical-formal-training/v1", "status": "initializing", "config": config, "test_split_used": False, "fresh_base_model": True, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _write_json(output / "run_report.json", report)
    try:
        set_seed(config["training"]["seed"])
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["local_path"], revision=config["model"]["revision"], local_files_only=True)
        train_cache_report = build_token_cache(ROOT / config["data"]["train"], output / "token_cache_train", tokenizer, config["data"]["max_length"])
        validation_cache_report = build_token_cache(ROOT / config["data"]["validation"], output / "token_cache_validation", tokenizer, config["data"]["max_length"])
        if train_cache_report["sample_count"] != config["data"]["expected_train_samples"] or validation_cache_report["sample_count"] != config["data"]["expected_validation_samples"]:
            raise RuntimeError("token cache sample count does not match frozen config")
        train_dataset, validation_dataset = TokenCacheDataset(output / "token_cache_train"), TokenCacheDataset(output / "token_cache_validation")
        validation_rows = []
        for path in config["data"]["validation_standardized"]:
            validation_rows.extend(read_jsonl(ROOT / path))
        if len(validation_rows) != config["data"]["expected_validation_samples"]:
            raise RuntimeError("standardized validation count mismatch")

        base = AutoModelForCausalLM.from_pretrained(config["model"]["local_path"], revision=config["model"]["revision"], torch_dtype=torch.float16, local_files_only=True, low_cpu_mem_usage=True).cuda()
        if config["training"]["gradient_checkpointing"]:
            base.gradient_checkpointing_enable()
            base.config.use_cache = False
            if hasattr(base, "enable_input_require_grads"):
                base.enable_input_require_grads()
        model = get_peft_model(base, LoraConfig(r=config["training"]["lora_r"], lora_alpha=config["training"]["lora_alpha"], lora_dropout=config["training"]["lora_dropout"], target_modules=config["training"]["target_modules"], task_type="CAUSAL_LM"))
        model.train()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
        steps_per_epoch = math.ceil(len(train_dataset) / config["training"]["gradient_accumulation_steps"])
        total_steps = steps_per_epoch * config["training"]["epochs"]
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config["training"]["warmup_ratio"]), total_steps)
        scaler = torch.cuda.amp.GradScaler(init_scale=config["training"]["amp_initial_scale"], growth_interval=config["training"]["amp_growth_interval"])
        state = {"epoch": 0, "micro_step": 0, "optimizer_step": 0, "successful_optimizer_steps": 0, "skipped_updates": 0, "total_steps": total_steps, "steps_per_epoch": steps_per_epoch}
        best = None
        epoch_summaries = []
        torch.cuda.reset_peak_memory_stats()
        training_started = time.perf_counter()

        for epoch in range(1, config["training"]["epochs"] + 1):
            generator = torch.Generator().manual_seed(config["training"]["seed"] + epoch)
            order = torch.randperm(len(train_dataset), generator=generator).tolist()
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss, accumulated_micro = 0.0, 0
            epoch_started = time.perf_counter()
            for position, index in enumerate(order, 1):
                row = train_dataset.row(index)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    loss = model(**_batch(row, torch)).loss / config["training"]["gradient_accumulation_steps"]
                scaler.scale(loss).backward()
                accumulated_loss += float(loss.detach().cpu()) * config["training"]["gradient_accumulation_steps"]
                accumulated_micro += 1
                state["micro_step"] += 1
                boundary = accumulated_micro == config["training"]["gradient_accumulation_steps"] or position == len(order)
                if not boundary:
                    continue
                scale_before = float(scaler.get_scale())
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, config["training"]["max_grad_norm"])
                gradients_finite = math.isfinite(float(grad_norm.detach().cpu()))
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
                skipped = scale_after < scale_before
                state["optimizer_step"] += 1
                if skipped:
                    state["skipped_updates"] += 1
                else:
                    state["successful_optimizer_steps"] += 1
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                log = {"epoch": epoch, "optimizer_step": state["optimizer_step"], "micro_step": state["micro_step"], "loss": accumulated_loss / accumulated_micro, "grad_norm": float(grad_norm.detach().cpu()), "gradients_finite": gradients_finite, "scale_before": scale_before, "scale_after": scale_after, "update_skipped": skipped, "learning_rate": scheduler.get_last_lr()[0], "elapsed_seconds": time.perf_counter() - training_started, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
                if state["optimizer_step"] % config["training"]["log_steps"] == 0:
                    with event_log.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(log, ensure_ascii=False) + "\n")
                    print(json.dumps(log, ensure_ascii=False), flush=True)
                if (not gradients_finite or skipped or not _all_finite(trainable)) and config["training"]["abort_on_nonfinite_or_skipped_update"]:
                    raise FloatingPointError(f"nonfinite or skipped update at optimizer step {state['optimizer_step']}")
                if state["optimizer_step"] % config["training"]["checkpoint_steps"] == 0:
                    _save_checkpoint(output, model, optimizer, scheduler, scaler, state, torch)
                accumulated_loss, accumulated_micro = 0.0, 0

            state["epoch"] = epoch
            checkpoint = _save_checkpoint(output, model, optimizer, scheduler, scaler, state, torch)
            summary, validation_loss = _evaluate_checkpoint(output, epoch, model, tokenizer, validation_dataset, validation_rows, config, torch)
            metrics = {"epoch": epoch, "checkpoint": str(checkpoint), "validation_loss": validation_loss, "joint_tool_parameter_accuracy": summary["joint_tool_parameter_accuracy"], "answer_scoring": summary["answer_scoring"], "overall_task_success_rate": summary["overall_task_success_rate"], "epoch_seconds": time.perf_counter() - epoch_started}
            epoch_summaries.append(metrics)
            score = _metric_tuple(summary, validation_loss)
            if best is None or score > best["score"]:
                best_dir = output / "best_adapter"
                model.save_pretrained(best_dir)
                best = {"epoch": epoch, "score": score, "adapter": str(best_dir), "metrics": metrics}
                _write_json(output / "best_checkpoint.json", best)
            report.update({"status": "training", "state": state, "epoch_summaries": epoch_summaries, "best_checkpoint": best, "elapsed_seconds": time.perf_counter() - training_started, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20})
            _write_json(output / "run_report.json", report)

        report.update({"status": "completed", "state": state, "epoch_summaries": epoch_summaries, "best_checkpoint": best, "elapsed_seconds": time.perf_counter() - training_started, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    except torch.cuda.OutOfMemoryError as exc:
        report.update({"status": "oom", "error_category": "gpu_memory", "error": str(exc), "test_split_used": False})
    except FloatingPointError as exc:
        report.update({"status": "aborted_numerical", "error_category": "numerical", "error": str(exc), "test_split_used": False})
    except Exception as exc:
        report.update({"status": "failed", "error_category": "runtime_or_dependency", "error": f"{type(exc).__name__}: {exc}", "test_split_used": False})
    _write_json(output / "run_report.json", report)
    print(json.dumps({key: report.get(key) for key in ("status", "state", "best_checkpoint", "elapsed_seconds", "peak_allocated_mib", "error_category", "error", "test_split_used")}, ensure_ascii=False, indent=2), flush=True)
    if report["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
