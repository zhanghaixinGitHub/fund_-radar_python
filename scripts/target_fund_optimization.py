"""002112 第二轮独立 CLI：只运行预定实验，不登记或采用模型。"""

import argparse
import json

from app.services import fund_002112_training as original
from app.services import target_fund_optimization as optimization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status", "verify"))
    parser.add_argument("--interrupt-after", choices=("2023Q2-A7-main",), help="仅用于检查点中断恢复验收")
    args = parser.parse_args()
    if args.interrupt_after and args.command != "run":
        parser.error("中断参数仅适用于 run")
    if args.command == "prepare":
        with original.run_lock():
            optimization.prepare(original.load_frozen(optimization.BASE))
            result = optimization.status()
    elif args.command == "run":
        result = optimization.run(args.interrupt_after)
    elif args.command == "status":
        result = optimization.status()
    else:
        result = optimization.verify_report()
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result.get("status") == "PARTIAL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
