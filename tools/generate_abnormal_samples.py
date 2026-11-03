from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.abnormal_samples import AbnormalConfig, generate_abnormal_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="artifacts/training_data/cholect50_rdv_original")
    parser.add_argument("--output-dir", default="artifacts/training_data/cholect50_rdv_original_abnormal")
    parser.add_argument("--train-per-type", type=int, default=5)
    parser.add_argument("--validation-per-type", type=int, default=2)
    parser.add_argument("--test-per-type", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    report = generate_abnormal_data(args.base_dir, args.output_dir, AbnormalConfig(args.train_per_type, args.validation_per_type, args.test_per_type, args.seed))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
