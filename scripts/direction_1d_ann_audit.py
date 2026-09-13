"""只读公告日期专项审计；prepare后最多两次fund_nav原始响应检查，禁止训练和入库。"""

import argparse
from pathlib import Path

from app.services import direction_1d_ann_audit as audit
from app.services.direction_1d_protocol import canonical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "probe", "finish", "verify"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scope", type=Path)
    parser.add_argument("--base-run", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--probe-funds", nargs=2)
    args = parser.parse_args()
    if args.command == "prepare":
        if any(v is None for v in (args.scope, args.base_run, args.plan, args.probe_funds)):
            parser.error("prepare需要--scope、--base-run、--plan、--probe-funds")
        result = audit.initialize(args.scope, args.base_run, args.plan, args.run_dir, args.probe_funds)
    else:
        result = getattr(audit, args.command)(args.run_dir)
    print(canonical(result))


if __name__ == "__main__":
    main()
