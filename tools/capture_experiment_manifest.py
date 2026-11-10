#!/usr/bin/env python3
"""Capture reproducibility metadata without copying datasets or model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime() -> dict:
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        result["torch"] = torch.__version__
        result["cuda_runtime"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            result["gpu"] = props.name
            result["gpu_memory_bytes"] = props.total_memory
    except ImportError:
        result["torch"] = None
        result["cuda_available"] = False
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/manifests/experiment.json")
    parser.add_argument("--file", action="append", default=[], help="File to hash; repeatable")
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()

    files = []
    for value in args.file:
        path = Path(value).resolve()
        files.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sha256(path) if path.is_file() else None,
            }
        )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "dirty": bool(git_value("status", "--porcelain")),
        },
        "runtime": runtime(),
        "files": files,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
