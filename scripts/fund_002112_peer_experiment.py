"""运行唯一已授权的补资料实验；固定目录与 12 次总预算，不开放新预算参数。"""

import argparse
import json

from app.services import fund_002112_peer_experiment as experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "train", "verify", "status", "_fit", "_restore"))
    parser.add_argument("--slot", choices=experiment.SLOTS)
    parser.add_argument("--interrupt-after", type=int, choices=range(1, 13))
    args = parser.parse_args()
    path = experiment.RUN
    if args.command == "freeze":
        with experiment.lock(path):
            experiment.freeze(path)
        result = {"frozen": True, "budget": experiment.BUDGET, "path": str(path)}
    elif args.command == "train":
        result = experiment.run(path, interrupt_after=args.interrupt_after)
    elif args.command == "verify":
        result = experiment.verify(path)
    elif args.command == "status":
        experiment.load_frozen(path)
        result = experiment.counts(path)
    else:
        if args.slot is None:
            parser.error("worker requires --slot")
        function = experiment.fit_worker if args.command == "_fit" else experiment.restore_worker
        result = function(path, args.slot)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
