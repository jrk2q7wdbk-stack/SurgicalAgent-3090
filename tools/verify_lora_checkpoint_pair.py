"""Verify that a resumed LoRA checkpoint advanced all required training state."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load_torch(path: Path, torch):
    with path.open("rb") as handle:
        return torch.load(handle, map_location="cpu")


def _tensors(value):
    if hasattr(value, "isfinite"):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _tensors(child)


def _read_checkpoint(path: Path, torch, load_file) -> dict:
    manifest = json.loads((path / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    payload = _load_torch(path / "training_state.pt", torch)
    adapter = load_file(str(path / "adapter" / "adapter_model.safetensors"), device="cpu")
    optimizer_tensors = list(_tensors(payload["optimizer"]["state"]))
    return {
        "manifest": manifest,
        "payload": payload,
        "adapter": adapter,
        "adapter_finite": all(bool(tensor.isfinite().all()) for tensor in adapter.values()),
        "optimizer_state_tensor_count": len(optimizer_tensors),
        "optimizer_state_finite": all(bool(tensor.isfinite().all()) for tensor in optimizer_tensors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch
    from safetensors.torch import load_file

    before_path = Path(args.before).resolve()
    after_path = Path(args.after).resolve()
    before = _read_checkpoint(before_path, torch, load_file)
    after = _read_checkpoint(after_path, torch, load_file)
    common = sorted(set(before["adapter"]) & set(after["adapter"]))
    changed = [name for name in common if not torch.equal(before["adapter"][name], after["adapter"][name])]
    state_before = before["payload"]["state"]
    state_after = after["payload"]["state"]
    scheduler_before = before["payload"]["scheduler"]
    scheduler_after = after["payload"]["scheduler"]
    scaler_before = before["payload"]["scaler"]
    scaler_after = after["payload"]["scaler"]
    required_payload = {"optimizer", "scheduler", "scaler", "state", "rng_state"}
    checks = {
        "manifests_complete": bool(before["manifest"].get("complete") and after["manifest"].get("complete")),
        "payload_fields_complete": required_payload.issubset(before["payload"]) and required_payload.issubset(after["payload"]),
        "step_delta_correct": state_after["optimizer_step"] - state_before["optimizer_step"] == 2,
        "successful_step_delta_correct": state_after["successful_optimizer_steps"] - state_before["successful_optimizer_steps"] == 2,
        "micro_step_delta_correct": state_after["micro_step"] - state_before["micro_step"] == 16,
        "scheduler_progress_correct": scheduler_before["last_epoch"] == state_before["successful_optimizer_steps"] and scheduler_after["last_epoch"] == state_after["successful_optimizer_steps"],
        "learning_rate_progressed": scheduler_after["_last_lr"][0] > scheduler_before["_last_lr"][0],
        "amp_scale_finite_and_preserved": math.isfinite(float(scaler_before["scale"])) and scaler_after["scale"] == scaler_before["scale"],
        "rng_state_present": all(name in before["payload"]["rng_state"] and name in after["payload"]["rng_state"] for name in ("python", "torch_cpu", "torch_cuda", "numpy")),
        "adapter_finite": before["adapter_finite"] and after["adapter_finite"],
        "optimizer_state_finite": before["optimizer_state_finite"] and after["optimizer_state_finite"],
        "adapter_parameters_updated": bool(changed),
    }
    report = {
        "schema_version": "lora-checkpoint-resume-verification/v1",
        "valid": all(checks.values()),
        "checks": checks,
        "before": {
            "path": str(before_path),
            "state": state_before,
            "scheduler_last_epoch": scheduler_before["last_epoch"],
            "learning_rate": scheduler_before["_last_lr"][0],
            "amp_scale": scaler_before["scale"],
            "optimizer_state_tensor_count": before["optimizer_state_tensor_count"],
        },
        "after": {
            "path": str(after_path),
            "state": state_after,
            "scheduler_last_epoch": scheduler_after["last_epoch"],
            "learning_rate": scheduler_after["_last_lr"][0],
            "amp_scale": scaler_after["scale"],
            "optimizer_state_tensor_count": after["optimizer_state_tensor_count"],
        },
        "adapter_tensor_count": len(common),
        "adapter_tensors_changed": len(changed),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
