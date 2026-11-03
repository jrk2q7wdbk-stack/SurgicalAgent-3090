"""Disk-backed assistant-only token cache for local LoRA training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from training.tool_call_data import build_assistant_labels, qwen_tools


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_token_cache(input_path: str | Path, cache_dir: str | Path, tokenizer, max_length: int) -> dict:
    source, target = Path(input_path), Path(cache_dir)
    target.mkdir(parents=True, exist_ok=True)
    fingerprint = {"input": str(source.resolve()), "input_sha256": _sha256(source), "max_length": max_length, "tokenizer": getattr(tokenizer, "name_or_path", None), "tokenizer_revision": getattr(tokenizer, "_commit_hash", None)}
    metadata_path = target / "metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint and all((target / name).exists() for name in ("input_ids.npy", "labels.npy", "offsets.npy")):
            return existing
    input_parts, label_parts, offsets, samples = [], [], [0], []
    with source.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            encoded = build_assistant_labels(tokenizer, row["messages"], qwen_tools(), max_length)
            if not encoded.get("ok"):
                raise ValueError(f"token cache rejected {row.get('metadata', {}).get('sample_id')}: {encoded}")
            ids = np.asarray(encoded["input_ids"], dtype=np.int32)
            labels = np.asarray(encoded["labels"], dtype=np.int32)
            input_parts.append(ids)
            label_parts.append(labels)
            offsets.append(offsets[-1] + len(ids))
            meta = row.get("metadata") or {}
            samples.append({"sample_id": meta.get("sample_id"), "question_type": meta.get("question_type") or meta.get("abnormal_type") or "unknown", "token_count": len(ids)})
    np.save(target / "input_ids.npy", np.concatenate(input_parts))
    np.save(target / "labels.npy", np.concatenate(label_parts))
    np.save(target / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    report = {"schema_version": "surgical-token-cache/v1", "fingerprint": fingerprint, "sample_count": len(samples), "token_count": offsets[-1], "samples": samples}
    metadata_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


class TokenCacheDataset:
    def __init__(self, cache_dir: str | Path):
        root = Path(cache_dir)
        self.input_ids = np.load(root / "input_ids.npy", mmap_mode="r")
        self.labels = np.load(root / "labels.npy", mmap_mode="r")
        self.offsets = np.load(root / "offsets.npy", mmap_mode="r")
        self.metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def row(self, index: int) -> dict:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        return {"input_ids": self.input_ids[start:end], "labels": self.labels[start:end], **self.metadata["samples"][index]}
