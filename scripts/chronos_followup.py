"""Ordered offline CLI: freeze, run, verify. Original runs and database are read-only."""

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "verify"))
    parser.add_argument("--run", type=UUID)
    args = parser.parse_args()
    from app.services.chronos_followup import execute_followup, freeze_followup, verify_followup

    if args.command == "freeze":
        if args.run:
            parser.error("freeze generates a new run UUID")
        folder = freeze_followup()
        result = {"run": folder.name}
    else:
        if not args.run:
            parser.error("--run is required")
        folder = ROOT / ".local-runs" / f"chronos-followup-{args.run}"
        if args.command == "run":
            report = execute_followup(folder)
            result = {
                "planned": report["planned"],
                "mature_answers": report["mature_answers"],
                "raw_direction_conclusion": report["raw_direction"]["conclusion"],
            }
        else:
            result = verify_followup(folder)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
