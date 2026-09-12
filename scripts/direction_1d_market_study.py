"""市场信息研究CLI；旧研究包只读、候选固定、真实预测与在用模型不变。"""

import argparse
from pathlib import Path

from app.services import direction_1d_market_study as study
from app.services.direction_1d_protocol import canonical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "train", "replay", "evaluate", "verify"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-run", type=Path)
    parser.add_argument("--market-dir", type=Path)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--shared-origin", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        if not all((args.base_run, args.market_dir, args.source_audit, args.shared_origin)):
            parser.error("freeze需要--base-run、--market-dir、--source-audit、--shared-origin")
        result = study.freeze(args.base_run, args.market_dir, args.source_audit, args.shared_origin, args.run_dir)
    elif args.command in ("train", "replay"):
        result = study.run(args.run_dir, replay=args.command == "replay")
    elif args.command == "evaluate":
        report = study.evaluate(args.run_dir)
        result = {k: report[k] for k in ("overall", "paired_primary", "paired_old_group", "runtime_model_changed")}
    else:
        result = study.verify(args.run_dir)
    print(canonical(result))


if __name__ == "__main__":
    main()
