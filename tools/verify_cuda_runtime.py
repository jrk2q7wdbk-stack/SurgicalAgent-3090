"""Run a real CUDA kernel and record driver/runtime compatibility evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def verify() -> dict:
    report = {"schema_version": "surgical-cuda-runtime/v1", "python": sys.version, "executable": sys.executable, "os": platform.platform(), "status": "not_started"}
    try:
        import torch
        report["torch"] = {"version": torch.__version__, "compiled_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=False)
        report["nvidia_smi"] = smi.stdout.strip() or smi.stderr.strip()
        if not torch.cuda.is_available():
            report.update({"status": "cuda_unavailable"})
            return report
        torch.cuda.empty_cache()
        started = time.perf_counter()
        allocation = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        left = (torch.arange(1024 * 1024, dtype=torch.float32, device="cuda").reshape(1024, 1024) / 1024).requires_grad_(True)
        right = torch.eye(1024, dtype=torch.float32, device="cuda")
        result = left @ right
        loss = result.square().mean()
        loss.backward()
        checksum = float(result.sum().cpu())
        torch.cuda.synchronize()
        expected = float(left.sum().cpu())
        report.update({
            "status": "passed" if abs(checksum - expected) < 1.0 else "numerical_mismatch",
            "operations": ["scalar tensor allocation", "1024x1024 CUDA float32 matrix multiplication", "backward propagation", "host transfer"],
            "scalar_allocation": float(allocation.cpu()[0]),
            "gradient_finite": bool(torch.isfinite(left.grad).all().cpu()),
            "checksum": checksum,
            "expected": expected,
            "absolute_error": abs(checksum - expected),
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "device": torch.cuda.get_device_name(0),
            "allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        })
    except Exception as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/stage5/cuda_runtime.json")
    args = parser.parse_args()
    report = verify()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
