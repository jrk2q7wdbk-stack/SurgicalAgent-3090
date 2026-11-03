from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.backends import AnnotationToolSimulator, CurrentAgentPolicyBackend, DeepSeekAutonomousBackend, LocalModelBackend, ReferenceProtocolBackend  # noqa: E402
from evaluation.runner import evaluate, read_jsonl, stratified_subset, write_evaluation  # noqa: E402
from agent.tool_registry import execute_registered_tool  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", choices=["mock", "current-agent", "deepseek-autonomous", "local"], default="mock")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--limit", type=int, default=35, help="Defaults to a small validation smoke subset; pass 0 for all only after protocol freeze.")
    parser.add_argument("--time-tolerance", type=float, default=0.01)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--evidence-backend", choices=["annotation-simulator", "registered-tools"], default="annotation-simulator")
    parser.add_argument("--adapter")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    all_rows = read_jsonl(args.input)
    rows = stratified_subset(all_rows, None if args.limit == 0 else args.limit)
    if args.backend == "mock":
        backend = ReferenceProtocolBackend()
    elif args.backend == "current-agent":
        backend = CurrentAgentPolicyBackend()
    elif args.backend == "deepseek-autonomous":
        simulator = AnnotationToolSimulator(all_rows)
        executor = simulator if args.evidence_backend == "annotation-simulator" else execute_registered_tool
        backend = DeepSeekAutonomousBackend(model=args.model, timeout=args.timeout, retries=args.retries, max_rounds=args.max_rounds, executor=executor, evidence_backend=args.evidence_backend)
    else:
        simulator = AnnotationToolSimulator(all_rows)
        executor = simulator if args.evidence_backend == "annotation-simulator" else execute_registered_tool
        backend = LocalModelBackend(args.model, executor=executor, evidence_backend=args.evidence_backend, adapter_path=args.adapter, max_rounds=args.max_rounds, max_new_tokens=args.max_new_tokens, local_files_only=not args.allow_model_download, device=args.device)
    results, summary, review = evaluate(rows, backend, args.time_tolerance)
    write_evaluation(args.output_dir, results, summary, review)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
