import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rtx3090_baseline_keeps_test_frozen_and_effective_batch_consistent():
    config = json.loads((ROOT / "config" / "local_lora_3090_baseline.json").read_text(encoding="utf-8"))
    training = config["training"]

    assert "frozen" in config["data"]["test_policy"]
    assert training["effective_batch_size"] == training["batch_size"] * training["gradient_accumulation_steps"]
    assert config["validation"]["generation_subset_limit"] == 100
    assert {"q_proj", "k_proj", "v_proj", "o_proj"}.issubset(training["target_modules"])
