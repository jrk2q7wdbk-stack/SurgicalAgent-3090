"""CLI for the reproducible CholecT50 training-data audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.data_audit import audit_training_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--training-dir", default="artifacts/training_data/cholect50_rdv_original")
    parser.add_argument("--output-dir", default="artifacts/data_audit/cholect50_rdv_original")
    parser.add_argument("--checkpoint", default="models/rendezvous_l8_cholectcholect50_k0_batchnorm_lowres.pth")
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--per-type", type=int, default=8)
    args = parser.parse_args()
    report = audit_training_data(args.dataset_root, args.training_dir, args.output_dir, args.checkpoint, args.seed, args.per_type)
    print(json.dumps({"valid": report["valid"], "counts": report["counts"], "issues": report["issues"]}, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
