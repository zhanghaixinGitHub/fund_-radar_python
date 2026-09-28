"""独立修复协议 CLI：只支持准入、离线拟合、状态和验证，不提供采用入口。"""

import argparse
import json

from app.services import fund_002112_repaired_experiment as run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "status", "verify", "_fit", "_restore"))
    parser.add_argument("--run-id")
    parser.add_argument("--slot", choices=(*run.INHERITED, *run.NEW_SLOTS))
    parser.add_argument("--interrupt-after", choices=run.NEW_SLOTS)
    args = parser.parse_args()
    if args.command == "preflight":
        path = run.preflight()
    else:
        if not args.run_id:
            parser.error("必须指定新协议绑定的运行编号")
        path = run.directory(args.run_id)
        if args.command == "run":
            run.run(path, interrupt_after=args.interrupt_after)
        elif args.command == "verify":
            print(json.dumps(run.verify(path), ensure_ascii=False))
        elif args.command in ("_fit", "_restore"):
            if not args.slot:
                parser.error("内部操作缺少固定槽位")
            (run.fit_worker if args.command == "_fit" else run.restore_worker)(path, args.slot)
    print(json.dumps(run.status(path), ensure_ascii=False))


if __name__ == "__main__":
    main()
