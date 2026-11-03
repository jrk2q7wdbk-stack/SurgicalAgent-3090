"""Structured, append-only audit logging for questions and answer routes."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict

from agent.config import get_config


_LOCK = Lock()


def write_audit_record(record: Dict[str, Any]) -> Path:
    log_dir = get_config().path("audit_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "agent_audit.jsonl"
    payload = dict(record)
    payload.setdefault(
        "timestamp_utc", datetime.now(timezone.utc).isoformat()
    )
    line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
    with _LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    return path

