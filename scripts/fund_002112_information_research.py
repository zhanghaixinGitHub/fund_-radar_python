"""执行 002112 新信息研究；预算来自已登记的 24 次授权，无扩大预算选项。"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "freeze", "train", "verify", "status", "_fit", "_restore"))
    parser.add_argument(
        "--hypothesis", choices=("H1_EARLY_OMISSION", "H2_EVENT_INFORMATION"), default="H1_EARLY_OMISSION"
    )
    parser.add_argument("--slot")
    parser.add_argument("--interrupt-after", type=int, choices=range(1, 13))
    args = parser.parse_args()
    if args.command == "prepare":
        from app.services.fund_002112_information_admission import prepare

        result = prepare()
    else:
        from app.services import fund_002112_information_experiment as experiment

        if args.command in ("_fit", "_restore"):
            if args.slot not in experiment.SLOTS:
                parser.error("worker requires a registered slot")
            result = experiment.worker(args.hypothesis, args.slot, restore=args.command == "_restore")
        elif args.command == "freeze":
            with experiment.lock():
                experiment.freeze(args.hypothesis)
            result = {"frozen": True, **experiment.status()}
        elif args.command == "train":
            result = experiment.run(args.hypothesis, args.interrupt_after)
        elif args.command == "verify":
            with experiment.lock():
                result = experiment.verify(args.hypothesis)
        else:
            result = experiment.status()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
