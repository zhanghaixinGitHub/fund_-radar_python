"""显式执行第五版资料计划；共享排他锁，失败来源不重试，不触发拟合。"""

import argparse
import json
import os
from datetime import datetime

from app.services.fund_earnings_batch_v5 import RUNS, ZONE, EarningsBatchV5, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "collect", "audit"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--previous")
    args = parser.parse_args()
    if args.command == "prepare" and not args.previous:
        parser.error("prepare requires --previous")
    batch = EarningsBatchV5(args.run)
    lock = RUNS / ".earnings-preparation.lock"
    owner = {"pid": os.getpid(), "run": args.run, "command": args.command, "at": datetime.now(ZONE).isoformat()}
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(owner, stream)
    try:
        result = batch.prepare(args.previous) if args.command == "prepare" else getattr(batch, args.command)()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if read(lock) == owner:
            lock.unlink()


if __name__ == "__main__":
    main()
