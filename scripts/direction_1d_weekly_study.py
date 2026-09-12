"""周更频率固定离线对照：先冻结，主训练一次，同配方复现一次，随后只读核验。"""

import argparse
from pathlib import Path

from app.services import direction_1d_weekly_study as study
from app.services.direction_1d_protocol import canonical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "train", "replay", "verify", "evaluate"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--base-run", type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        if args.base_run is None or args.plan is None:
            parser.error("freeze需要--base-run和--plan")
        result = study.freeze(args.base_run, args.plan, args.run_dir)
    elif args.command in ("train", "replay"):
        result = study.run(args.run_dir, replay=args.command == "replay")
    elif args.command == "evaluate":
        report = study.evaluate(args.run_dir)
        result = {k: report[k] for k in ("overall", "paired")}
    else:
        result = study.verify(args.run_dir)
    print(canonical(result))


if __name__ == "__main__":
    main()
