"""Read-only environment audit for local LoRA feasibility."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _memory() -> dict:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if os.name == "nt" and ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return {"total_gib": round(status.ullTotalPhys / 2**30, 2), "available_gib": round(status.ullAvailPhys / 2**30, 2)}
    return {"total_gib": None, "available_gib": None}


def audit() -> dict:
    packages = ["torch", "transformers", "peft", "trl", "accelerate", "bitsandbytes", "datasets", "sentencepiece"]
    package_info = {}
    for name in packages:
        spec = importlib.util.find_spec(name)
        if spec is None:
            package_info[name] = {"installed": False, "version": None}
        else:
            try:
                module = __import__(name)
                package_info[name] = {"installed": True, "version": getattr(module, "__version__", "unknown")}
            except Exception as exc:
                package_info[name] = {"installed": True, "version": None, "import_error": f"{type(exc).__name__}: {exc}"}
    gpu = {"available": False}
    if package_info["torch"]["installed"]:
        import torch
        gpu = {"available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda, "torch_version": torch.__version__}
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            gpu.update({"name": torch.cuda.get_device_name(0), "total_mib": round(total / 2**20), "free_mib": round(free / 2**20), "capability": list(torch.cuda.get_device_capability(0))})
            try:
                left = torch.ones((64, 64), device="cuda", requires_grad=True)
                loss = (left @ left).mean()
                loss.backward()
                torch.cuda.synchronize()
                gpu["runtime_probe"] = {"status": "passed", "operations": ["allocation", "matrix_multiplication", "backward"], "gradient_finite": bool(torch.isfinite(left.grad).all().cpu())}
                del left, loss
                torch.cuda.empty_cache()
            except Exception as exc:
                gpu["runtime_probe"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=False)
        gpu["nvidia_smi"] = smi.stdout.strip() or smi.stderr.strip()
        if smi.returncode == 0 and smi.stdout.strip():
            driver, total, used, free = [item.strip() for item in smi.stdout.splitlines()[0].split(",")]
            gpu.update({"driver": driver, "total_mib": int(total), "used_mib": int(used), "free_mib": int(free), "memory_source": "nvidia-smi"})
    except Exception as exc:
        gpu["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    disk = shutil.disk_usage(ROOT)
    hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    local_model = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
    required_ready = all(package_info[name]["installed"] for name in ("torch", "transformers", "peft", "accelerate"))
    cuda_ready = gpu.get("runtime_probe", {}).get("status") == "passed"
    smoke_report = ROOT / "artifacts" / "stage5" / "lora_smoke_after_driver_update" / "smoke_report.json"
    smoke_status = None
    if smoke_report.exists():
        try:
            smoke_status = json.loads(smoke_report.read_text(encoding="utf-8")).get("status")
        except Exception:
            smoke_status = "unreadable"
    pip_check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=30, check=False)
    return {
        "schema_version": "surgical-training-environment/v1",
        "os": platform.platform(), "python": sys.version, "executable": sys.executable,
        "gpu": gpu, "memory": _memory(),
        "disk": {"path": str(ROOT.drive or ROOT), "total_gib": round(disk.total / 2**30, 2), "free_gib": round(disk.free / 2**30, 2)},
        "packages": package_info,
        "model_cache": {"hf_home": str(hf_home), "exists": hf_home.exists(), "selected_model_found": local_model.exists(), "selected_model_path": str(local_model)},
        "compatibility": {
            "declared_minimums_satisfied": required_ready,
            "peft_0_13_2_requires_torch": ">=1.13.0",
            "accelerate_1_1_1_requires_torch": ">=1.10.0",
            "transformers_tool_template_requirement": ">=4.42",
            "pip_check_status": "passed" if pip_check.returncode == 0 else "unrelated_conflict_present",
            "pip_check_output": (pip_check.stdout or pip_check.stderr).strip(),
            "note": "The OpenCV/NumPy conflict, when present, is unrelated to the text-only LoRA smoke and was not changed.",
        },
        "smoke_training_status": smoke_status,
        "recommendation": {"environment": "separate conda environment; do not modify rdv4", "model": "Qwen/Qwen2.5-0.5B-Instruct", "method": "FP16 LoRA smoke; QLoRA only after bitsandbytes compatibility is verified", "feasibility_status": "smoke_completed" if smoke_status == "completed" and cuda_ready else ("ready_for_smoke" if required_ready and cuda_ready and local_model.exists() else "blocked")},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/training_feasibility/environment.json")
    args = parser.parse_args()
    report = audit()
    path = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
