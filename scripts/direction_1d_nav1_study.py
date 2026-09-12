"""原树增加基金自身1日幅度的固定离线对照；入口不接受可调超参数，不登记模型。"""

import argparse
from pathlib import Path

from app.services import direction_1d_nav1_study as study
from app.services.direction_1d_protocol import canonical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "train", "replay", "evaluate", "verify"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-run", type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        if args.base_run is None or args.plan is None:
            parser.error("freeze需要--base-run及--plan")
        result = study.freeze(args.base_run, args.plan, args.run_dir)
    elif args.command in ("train", "replay"):
        result = study.run(args.run_dir, replay=args.command == "replay")
    elif args.command == "evaluate":
        report = study.evaluate(args.run_dir)
        result = {k: report[k] for k in ("overall", "paired_primary", "paired_linear")}
    else:
        result = study.verify(args.run_dir)
    print(canonical(result))


if __name__ == "__main__":
    main()
