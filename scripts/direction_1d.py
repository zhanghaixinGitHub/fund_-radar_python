"""1日实验受控CLI；训练离线，补数必须显式使用sync子命令。"""

import argparse
import json
from pathlib import Path


def main():
    from app.services import direction_1d_training as training
    from app.services.direction_1d_data import history, inventory

    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    for action in ("inventory", "sync", "freeze"):
        sub = subs.add_parser(action)
        sub.add_argument("--scope", type=Path, required=True)
        sub.add_argument("--out", type=Path, required=True)
    for action in ("build", "train", "evaluate", "verify", "replay", "register"):
        sub = subs.add_parser(action)
        sub.add_argument("--run-dir", type=Path, required=True)
        if action == "replay":
            sub.add_argument("--out", type=Path, required=True)
        if action == "register":
            sub.add_argument("--experimental", action="store_true", required=True)
    args = parser.parse_args()
    if args.action in {"inventory", "sync", "freeze"}:
        scope = json.loads(args.scope.read_text(encoding="utf-8"))
        codes = scope["fund_codes"]
        if args.action == "inventory":
            result = inventory(codes)
            args.out.mkdir(exist_ok=True, parents=True)
            training.write_new(args.out / "coverage.json", result)
        elif args.action == "sync":
            from app.services.direction_1d_jobs import sync_missing

            result = sync_missing(codes)
            args.out.mkdir(exist_ok=True, parents=True)
            training.write_new(args.out / "sync-receipt.json", result)
        else:
            training.freeze(history(codes), args.out)
            result = {"frozen": str(args.out)}
    elif args.action == "replay":
        result = training.replay(args.run_dir, args.out)
    elif args.action == "evaluate":
        training.verify(args.run_dir)
        result = training.read(args.run_dir / "metrics.json")
    else:
        result = getattr(training, args.action)(args.run_dir)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
