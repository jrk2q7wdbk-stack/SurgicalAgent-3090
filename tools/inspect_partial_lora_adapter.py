"""Load and classify a LoRA adapter left by an interrupted training run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter = _resolve(args.adapter)
    model_path = _resolve(args.model)
    output = _resolve(args.output)
    weight = adapter / "adapter_model.safetensors"
    config = adapter / "adapter_config.json"
    if not weight.is_file() or not config.is_file():
        raise SystemExit("adapter_config.json or adapter_model.safetensors is missing")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    started = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    lora_parameters = [(name, parameter) for name, parameter in model.named_parameters() if "lora_" in name]
    nonfinite = [name for name, parameter in lora_parameters if not bool(parameter.detach().isfinite().all().cpu())]
    report = {
        "schema_version": "partial-lora-diagnostic/v1",
        "classification": "diagnostic_partial_training_weights",
        "resumable_checkpoint": False,
        "reason": "adapter exists, but optimizer/scheduler/AMP scaler/global step/RNG checkpoint is absent",
        "adapter": str(adapter),
        "adapter_weight_bytes": weight.stat().st_size,
        "adapter_sha256": _sha256(weight),
        "base_model": str(model_path),
        "device": device,
        "load_succeeded": True,
        "lora_parameter_tensors": len(lora_parameters),
        "lora_parameter_values": sum(parameter.numel() for _, parameter in lora_parameters),
        "all_lora_parameters_finite": not nonfinite,
        "nonfinite_parameter_names": nonfinite,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if nonfinite:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
