"""Local-only comparison CLI. See the Chinese implementation guide for ordered steps."""

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID, uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "export", "smoke", "run", "verify", "replay"))
    parser.add_argument("--run", type=UUID, help="Existing local comparison UUID")
    parser.add_argument("--source-run", type=UUID)
    parser.add_argument("--dataset-hash")
    args = parser.parse_args()
    from app.services.model_comparison_protocol import REVISION, freeze_protocol

    if args.command == "freeze":
        if not args.source_run or not args.dataset_hash or args.run:
            parser.error("freeze requires --source-run and --dataset-hash, and generates a fresh run UUID")
        folder = ROOT / ".local-runs" / f"model-comparison-{uuid4()}"
        folder.mkdir(parents=True)
        protocol = freeze_protocol(folder, args.source_run, args.dataset_hash, ROOT / "requirements-chronos2.txt")
        result = {"run": folder.name, "protocol_hash": protocol["protocol_hash"]}
    else:
        if not args.run:
            parser.error("--run is required")
        folder = ROOT / ".local-runs" / f"model-comparison-{args.run}"
        checkpoint = ROOT / ".local-runs" / "model-cache" / REVISION
        if args.command == "export":
            from app.services.model_comparison_dataset import export_dataset

            manifest = export_dataset(folder)
            result = {
                "inputs": manifest["input_count"],
                "answers": manifest["answer_count"],
                "manifest_hash": manifest["manifest_hash"],
            }
        else:
            from app.services.model_comparison_runner import replay_smoke, run_comparison, smoke, verify_run

            if args.command == "smoke":
                report = smoke(folder, checkpoint)
                result = {
                    "real_samples": len(report["rows"]),
                    "real_lags": report["real_lags"],
                    "no_exam_answers_read": report["no_exam_answers_read"],
                }
            elif args.command == "run":
                report = run_comparison(folder, checkpoint)
                result = {"conclusion": report["conclusion"]}
            elif args.command == "replay":
                result = replay_smoke(folder, checkpoint)
            else:
                result = verify_run(folder)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
