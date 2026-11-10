# RTX 3090 baseline workflow

This workflow reproduces the 0.5B LoRA baseline before changing the model family,
data, prompts, or evaluation rules. The test split stays frozen.

## 1. Environment

Create a separate Python 3.10 environment and install
`requirements-local-training.txt`. Record the exact package versions with
`pip freeze`. Confirm that PyTorch sees the RTX 3090 before training.

## 2. Reproducibility manifest

Generate hashes for every local input used by the run. Paths are recorded, but
file contents are not copied:

```powershell
python tools/capture_experiment_manifest.py `
  --label rtx3090-baseline `
  --output artifacts/manifests/rtx3090-baseline.json `
  --file models/Qwen2.5-0.5B-Instruct/model.safetensors `
  --file artifacts/local_training/train.qwen_tools.jsonl `
  --file artifacts/local_training/validation.qwen_tools.jsonl
```

Add the RDV checkpoint with another `--file` when running end-to-end vision
evaluation. Do not commit private datasets, base-model weights, or credentials.

## 3. Preflight and training

Run the existing preflight first, then train with the 3090 configuration:

```powershell
python tools/run_lora_preflight.py --config config/local_lora_3090_baseline.json
python tools/train_lora_round1.py --config config/local_lora_3090_baseline.json
```

The current trainer processes one tokenized sample per micro-step, so the
configuration keeps `batch_size=1` and raises gradient accumulation to 16. A
larger configured batch size would currently be misleading because the trainer
does not construct batched rows.

## 4. Acceptance checks

Before comparing a larger model, require all of the following:

- no non-finite or skipped optimizer update;
- all unit tests pass except explicitly documented external-service skips;
- validation is completed without reading the test split;
- the 100-sample deterministic generation subset is saved per epoch;
- pending manual reviews are completed before reporting final answer accuracy;
- peak GPU memory, elapsed time, checkpoint hash, and Git commit are recorded.

The `baseline` branch is the immutable copy of the original repository. All new
work belongs on `main` or a dedicated experiment branch.
