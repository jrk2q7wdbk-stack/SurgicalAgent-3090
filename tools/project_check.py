"""Read-only project and external-dependency inventory.

This check never treats a present checkpoint as proof that inference works.
It can emit a reproducible JSON report for stage hand-off.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import platform
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.config import get_config  # noqa: E402


REQUIRED_PACKAGES = ("torch", "torchvision", "PIL", "numpy", "cv2", "openai")
PATH_KEYS = (
    "model_checkpoint",
    "instrument_labels",
    "verb_labels",
    "target_labels",
    "triplet_labels",
    "knowledge_base",
    "knowledge_graph",
)


def _file_info(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def _json_info(path: Path) -> dict:
    info = _file_info(path)
    if not info["exists"]:
        return info
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        info["valid_json"] = True
        info["top_level_type"] = type(data).__name__
        info["item_count"] = len(data) if hasattr(data, "__len__") else None
    except Exception as exc:
        info["valid_json"] = False
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def build_report() -> dict:
    config = get_config(reload=True)
    files = {}
    for key in PATH_KEYS:
        path = config.path(key)
        files[key] = (
            _file_info(path) if key == "model_checkpoint" else _json_info(path)
        )

    duplicate_suffixes = sorted(
        str(path.relative_to(PROJECT_ROOT))
        for path in PROJECT_ROOT.rglob("*")
        if path.is_file() and re.search(r"\s*\(\d+\)(?=\.[^.]+$)", path.name)
    )
    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in REQUIRED_PACKAGES
    }
    required_files_ok = all(item["exists"] for item in files.values())
    json_files_ok = all(
        item.get("valid_json", True) for item in files.values()
    )
    runtime_ready = required_files_ok and json_files_ok and all(packages.values())

    blockers = []
    for name, available in packages.items():
        if not available:
            blockers.append(f"missing_python_package:{name}")
    for name, info in files.items():
        if not info["exists"]:
            blockers.append(f"missing_external_file:{name}")
        elif info.get("valid_json") is False:
            blockers.append(f"invalid_json:{name}")
    if not config.llm_api_key():
        blockers.append(
            "missing_optional_llm_api_key:"
            + str(config.section("llm").get("api_key_env"))
        )

    return {
        "schema_version": "surgical-agent-project-check/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "config": config.public_summary(),
        "entry_points": {
            "capability_demo": "test_agent_capabilities.py",
            "image_baseline": "test_inference.py",
            "video_llm_demo": "tests/test_video_llm_agent.py",
            "project_check": "python -m tools.project_check",
        },
        "configured_files": files,
        "python_packages": packages,
        "attachment_duplicate_suffix_files": duplicate_suffixes,
        "runtime_ready_for_real_rdv_inference": runtime_ready,
        "blockers": blockers,
        "notes": [
            "Checkpoint presence is not an inference success result.",
            "No annotation dataset was found in the current project tree.",
            "API-key presence is reported only as a boolean; its value is never logged.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="optional JSON report path")
    args = parser.parse_args()
    report = build_report()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"project check report: {output.resolve()}")
    print(rendered)


if __name__ == "__main__":
    main()

