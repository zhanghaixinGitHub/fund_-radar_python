"""沪深300成交活跃度的固定离线对照；有限补数、冻结、训练、复现和只读恢复。"""

import argparse
from pathlib import Path

from app.services import direction_1d_activity_data as data
from app.services import direction_1d_activity_study as study
from app.services.direction_1d_protocol import canonical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare-data", "acquire-data", "freeze", "train", "replay", "evaluate", "verify")
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-run", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.command == "prepare-data":
        if args.base_run is None or args.plan is None or args.audit is None:
            parser.error("prepare-data需要--base-run、--plan及--audit")
        result = data.initialize(args.run_dir, args.base_run, args.audit, args.plan)
    elif args.command == "acquire-data":
        result = data.acquire(args.run_dir)
    elif args.command == "freeze":
        if args.base_run is None or args.plan is None or args.data_dir is None:
            parser.error("freeze需要--base-run、--plan及--data-dir")
        result = study.freeze(args.base_run, args.data_dir, args.plan, args.run_dir)
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
