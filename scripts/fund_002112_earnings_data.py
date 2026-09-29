"""通过显式计划名执行资料批次；进程间共享排他锁，不启动服务或模型拟合。"""

import argparse
import json
import os
from datetime import datetime

from app.services.fund_earnings_batch_v3 import RUNS, EarningsBatch
from app.services.fund_information_history_v1 import ZONE, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "collect", "audit"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--previous")
    args = parser.parse_args()
    if args.command == "prepare" and not args.previous:
        parser.error("prepare requires --previous")
    batch = EarningsBatch(args.run)
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
