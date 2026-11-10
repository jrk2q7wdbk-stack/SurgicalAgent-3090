from pathlib import Path

import pytest
import torch

from tools.train_lora_round1 import _load_checkpoint_state, _resolve_path, _torch_load, _torch_save


def test_absolute_output_path_is_not_prefixed_by_project_root(tmp_path: Path):
    absolute = (tmp_path / "run").resolve()
    assert _resolve_path(absolute) == absolute


def test_torch_file_handle_save_supports_unicode_parent(tmp_path: Path):
    destination = tmp_path / "中文目录" / "training_state.pt"
    _torch_save(destination, {"step": 3, "tensor": torch.tensor([1.0])}, torch)
    restored = _torch_load(destination, torch)
    assert restored["step"] == 3
    assert torch.equal(restored["tensor"], torch.tensor([1.0]))


def test_partial_adapter_is_not_accepted_as_resumable_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint_step_000200"
    adapter = checkpoint / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"diagnostic-only")
    with pytest.raises(RuntimeError, match="incomplete checkpoint"):
        _load_checkpoint_state(checkpoint, None, None, None, torch)
