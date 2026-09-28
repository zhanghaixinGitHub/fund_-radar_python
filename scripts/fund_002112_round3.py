"""独立离线 CLI；不提供登记、采用、同步或外部模型加载操作。"""

import argparse
import json

from app.services import fund_002112_round3_run as run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preflight", "run", "status", "verify", "report", "news", "_fit", "_restore")
    )
    parser.add_argument("--run-id")
    parser.add_argument("--slot", choices=run.SLOTS)
    parser.add_argument("--interrupt-after", choices=run.SLOTS)
    args = parser.parse_args()
    if args.command == "preflight":
        p = run.preflight()
    else:
        if not args.run_id:
            parser.error("恢复、运行和状态查询必须显式指定原 --run-id")
        p = run.directory(args.run_id)
        if args.command == "run":
            run.run(p, interrupt_after=args.interrupt_after)
        elif args.command == "verify":
            print(json.dumps(run.verify(p), ensure_ascii=False))
        elif args.command == "report":
            print(run.report(p))
        elif args.command == "news":
            from app.services.fund_002112_round3_news import build

            print(json.dumps(build(p / "news"), ensure_ascii=False))
        elif args.command in ("_fit", "_restore"):
            if not args.slot:
                parser.error("内部操作必须指定 --slot")
            (run.fit_worker if args.command == "_fit" else run.restore_worker)(p, args.slot)
    print(json.dumps(run.status(p), ensure_ascii=False))


if __name__ == "__main__":
    main()
