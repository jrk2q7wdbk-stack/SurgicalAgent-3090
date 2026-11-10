"""First formal LoRA training round with frozen-test safeguards."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
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
    # Token-cache arrays are read-only memory maps; copy before giving them to PyTorch.
    ids = torch.tensor(row["input_ids"], dtype=torch.long, device="cuda").unsqueeze(0)
    labels = torch.tensor(row["labels"], dtype=torch.long, device="cuda").unsqueeze(0)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def _all_finite(parameters) -> bool:
    return all(bool(parameter.detach().isfinite().all().cpu()) for parameter in parameters)


def _resolve_path(value: str | Path, base: Path = ROOT) -> Path:
    """Resolve config paths without accidentally prefixing absolute paths."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _capture_rng_state(torch) -> dict:
    state = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        state["numpy"] = None
    return state


def _restore_rng_state(state: dict, torch) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if state.get("numpy") is not None:
        import numpy as np

        np.random.set_state(state["numpy"])


def _torch_save(path: Path, value, torch) -> None:
    """Use a Python file handle to avoid the PyTorch 1.13 Windows Unicode path bug."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        torch.save(value, handle)


def _torch_load(path: Path, torch, *, map_location="cpu"):
    with path.open("rb") as handle:
        return torch.load(handle, map_location=map_location)


def _save_checkpoint(root: Path, model, optimizer, scheduler, scaler, state: dict, torch) -> Path:
    checkpoint = root / f"checkpoint_step_{state['optimizer_step']:06d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint / "adapter")
    payload = {
        "schema_version": "surgical-lora-checkpoint/v2",
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "state": dict(state),
        "rng_state": _capture_rng_state(torch),
    }
    _torch_save(checkpoint / "training_state.pt", payload, torch)
    _write_json(checkpoint / "state.json", state)
    files = []
    for path in sorted(
        item
        for item in checkpoint.rglob("*")
        if item.is_file() and item.name != "checkpoint_manifest.json"
    ):
        files.append({"path": path.relative_to(checkpoint).as_posix(), "bytes": path.stat().st_size})
    _write_json(
        checkpoint / "checkpoint_manifest.json",
        {
            "schema_version": "surgical-lora-checkpoint-manifest/v1",
            "complete": True,
            "optimizer_step": state["optimizer_step"],
            "successful_optimizer_steps": state["successful_optimizer_steps"],
            "files": files,
        },
    )
    return checkpoint


def _load_checkpoint_state(checkpoint: Path, optimizer, scheduler, scaler, torch) -> dict:
    required = [
        checkpoint / "adapter" / "adapter_config.json",
        checkpoint / "adapter" / "adapter_model.safetensors",
        checkpoint / "training_state.pt",
        checkpoint / "state.json",
        checkpoint / "checkpoint_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"incomplete checkpoint; missing: {missing}")
    manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise RuntimeError("checkpoint manifest is not marked complete")
    payload = _torch_load(checkpoint / "training_state.pt", torch, map_location="cpu")
    if payload.get("schema_version") != "surgical-lora-checkpoint/v2":
        raise RuntimeError("unsupported or diagnostic-only checkpoint format")
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    _restore_rng_state(payload["rng_state"], torch)
    return dict(payload["state"])


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
    parser.add_argument("--output-dir", help="Absolute ASCII-only output directory override")
    parser.add_argument("--resume-from-checkpoint", help="Complete v2 checkpoint to resume")
    parser.add_argument(
        "--verification-max-optimizer-steps",
        type=int,
        help="Stop successfully after at most 20 optimizer steps; for checkpoint-flow verification only",
    )
    args = parser.parse_args()
    verification_limit = args.verification_max_optimizer_steps
    if verification_limit is None and not args.confirm_full_training:
        raise SystemExit("Refusing to start formal training without --confirm-full-training")
    if verification_limit is not None and (verification_limit < 1 or verification_limit > 20):
        raise SystemExit("--verification-max-optimizer-steps must be in [1, 20]")
    config_path = _resolve_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "test" in json.dumps(config.get("data", {}), ensure_ascii=False).lower() and config["data"].get("test_policy") != "frozen and not accepted by this runner":
        raise SystemExit("Test data is forbidden in the training runner")

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

    output = _resolve_path(args.output_dir or config["output_dir"])
    if sys.platform == "win32" and not str(output).isascii():
        raise SystemExit("Windows training output must be an ASCII-only absolute path; use --output-dir D:/SurgicalAgentRuns/<run>")
    resume_checkpoint = _resolve_path(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    output.mkdir(parents=True, exist_ok=True)
    event_log = output / "training_log.jsonl"
    report = {
        "schema_version": "surgical-formal-training/v2",
        "status": "initializing",
        "config": config,
        "runtime": {
            "config_path": str(config_path),
            "output_dir": str(output),
            "output_path_ascii": str(output).isascii(),
            "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
            "verification_max_optimizer_steps": verification_limit,
        },
        "test_split_used": False,
        "fresh_base_model": resume_checkpoint is None,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _write_json(output / "run_report.json", report)
    try:
        set_seed(config["training"]["seed"])
        model_path = _resolve_path(config["model"]["local_path"])
        tokenizer = AutoTokenizer.from_pretrained(model_path, revision=config["model"]["revision"], local_files_only=True)
        train_cache_report = build_token_cache(_resolve_path(config["data"]["train"]), output / "token_cache_train", tokenizer, config["data"]["max_length"])
        validation_cache_report = build_token_cache(_resolve_path(config["data"]["validation"]), output / "token_cache_validation", tokenizer, config["data"]["max_length"])
        if train_cache_report["sample_count"] != config["data"]["expected_train_samples"] or validation_cache_report["sample_count"] != config["data"]["expected_validation_samples"]:
            raise RuntimeError("token cache sample count does not match frozen config")
        train_dataset, validation_dataset = TokenCacheDataset(output / "token_cache_train"), TokenCacheDataset(output / "token_cache_validation")
        validation_rows = []
        for path in config["data"]["validation_standardized"]:
            validation_rows.extend(read_jsonl(_resolve_path(path)))
        if len(validation_rows) != config["data"]["expected_validation_samples"]:
            raise RuntimeError("standardized validation count mismatch")

        base = AutoModelForCausalLM.from_pretrained(model_path, revision=config["model"]["revision"], torch_dtype=torch.float16, local_files_only=True, low_cpu_mem_usage=True).cuda()
        if config["training"]["gradient_checkpointing"]:
            base.gradient_checkpointing_enable()
            base.config.use_cache = False
            if hasattr(base, "enable_input_require_grads"):
                base.enable_input_require_grads()
        if resume_checkpoint is None:
            model = get_peft_model(base, LoraConfig(r=config["training"]["lora_r"], lora_alpha=config["training"]["lora_alpha"], lora_dropout=config["training"]["lora_dropout"], target_modules=config["training"]["target_modules"], task_type="CAUSAL_LM"))
        else:
            model = PeftModel.from_pretrained(base, resume_checkpoint / "adapter", is_trainable=True)
        model.train()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
        steps_per_epoch = math.ceil(len(train_dataset) / config["training"]["gradient_accumulation_steps"])
        total_steps = steps_per_epoch * config["training"]["epochs"]
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config["training"]["warmup_ratio"]), total_steps)
        scaler = torch.cuda.amp.GradScaler(init_scale=config["training"]["amp_initial_scale"], growth_interval=config["training"]["amp_growth_interval"])
        state = {"epoch": 0, "epoch_position": 0, "micro_step": 0, "optimizer_step": 0, "successful_optimizer_steps": 0, "skipped_updates": 0, "total_steps": total_steps, "steps_per_epoch": steps_per_epoch}
        if resume_checkpoint is not None:
            state = _load_checkpoint_state(resume_checkpoint, optimizer, scheduler, scaler, torch)
            if state["total_steps"] != total_steps or state["steps_per_epoch"] != steps_per_epoch:
                raise RuntimeError("checkpoint schedule does not match the current frozen data/config")
            if verification_limit is not None and verification_limit <= state["optimizer_step"]:
                raise RuntimeError("verification step limit must be greater than the resumed optimizer step")
            report["resume"] = {
                "checkpoint": str(resume_checkpoint),
                "optimizer_step": state["optimizer_step"],
                "successful_optimizer_steps": state["successful_optimizer_steps"],
                "scheduler_last_epoch": scheduler.last_epoch,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        if not _all_finite(trainable):
            raise FloatingPointError("LoRA parameters are non-finite before training")
        best = None
        epoch_summaries = []
        torch.cuda.reset_peak_memory_stats()
        training_started = time.perf_counter()

        if state["epoch"] and state.get("epoch_position", 0) >= len(train_dataset):
            first_epoch = state["epoch"] + 1
        else:
            first_epoch = max(1, state["epoch"])
        verification_checkpoint = None
        stop_for_verification = False
        for epoch in range(first_epoch, config["training"]["epochs"] + 1):
            generator = torch.Generator().manual_seed(config["training"]["seed"] + epoch)
            order = torch.randperm(len(train_dataset), generator=generator).tolist()
            start_position = state.get("epoch_position", 0) if state["epoch"] == epoch else 0
            state["epoch"] = epoch
            state["epoch_position"] = start_position
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss, accumulated_micro = 0.0, 0
            epoch_started = time.perf_counter()
            for position in range(start_position + 1, len(order) + 1):
                index = order[position - 1]
                row = train_dataset.row(index)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    loss = model(**_batch(row, torch)).loss / config["training"]["gradient_accumulation_steps"]
                scaler.scale(loss).backward()
                accumulated_loss += float(loss.detach().cpu()) * config["training"]["gradient_accumulation_steps"]
                accumulated_micro += 1
                state["micro_step"] += 1
                state["epoch_position"] = position
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
                if verification_limit is not None and state["optimizer_step"] >= verification_limit:
                    verification_checkpoint = _save_checkpoint(output, model, optimizer, scheduler, scaler, state, torch)
                    stop_for_verification = True
                    break
                accumulated_loss, accumulated_micro = 0.0, 0

            if stop_for_verification:
                break
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

        if stop_for_verification:
            report.update({"status": "verification_completed", "state": state, "verification_checkpoint": str(verification_checkpoint), "epoch_summaries": epoch_summaries, "best_checkpoint": best, "elapsed_seconds": time.perf_counter() - training_started, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        else:
            report.update({"status": "completed", "state": state, "epoch_summaries": epoch_summaries, "best_checkpoint": best, "elapsed_seconds": time.perf_counter() - training_started, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    except torch.cuda.OutOfMemoryError as exc:
        report.update({"status": "oom", "error_category": "gpu_memory", "error": str(exc), "test_split_used": False})
    except FloatingPointError as exc:
        report.update({"status": "aborted_numerical", "error_category": "numerical", "error": str(exc), "test_split_used": False})
    except Exception as exc:
        report.update({"status": "failed", "error_category": "runtime_or_dependency", "error": f"{type(exc).__name__}: {exc}", "test_split_used": False})
    _write_json(output / "run_report.json", report)
    print(json.dumps({key: report.get(key) for key in ("status", "state", "best_checkpoint", "elapsed_seconds", "peak_allocated_mib", "error_category", "error", "test_split_used")}, ensure_ascii=False, indent=2), flush=True)
    if report["status"] not in {"completed", "verification_completed"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
