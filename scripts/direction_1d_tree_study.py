"""固定浅树离线对照入口，诊断与冻结先于拟合，不注册模型。"""

import argparse
from pathlib import Path

from app.services import direction_1d_tree_audit as audit
from app.services import direction_1d_tree_study as study
from app.services.direction_1d_protocol import canonical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "freeze", "train", "replay", "evaluate", "verify"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-run", type=Path)
    parser.add_argument("--audit-dir", type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        if args.base_run is None:
            parser.error("audit需要--base-run")
        report = audit.collect(args.base_run, args.run_dir)
        result = {
            k: report[k] for k in ("exam_count", "frozen_nav_count", "nav_mismatch_count", "requires_data_repair")
        }
    elif args.command == "freeze":
        if args.base_run is None or args.audit_dir is None:
            parser.error("freeze需要--base-run和--audit-dir")
        result = study.freeze(args.base_run, args.audit_dir, args.run_dir)
    elif args.command in ("train", "replay"):
        result = study.run(args.run_dir, replay=args.command == "replay")
    elif args.command == "evaluate":
        report = study.evaluate(args.run_dir)
        result = {k: report[k] for k in ("overall", "paired_primary", "paired_old_group")}
    else:
        result = study.verify(args.run_dir)
    print(canonical(result))


if __name__ == "__main__":
    main()
