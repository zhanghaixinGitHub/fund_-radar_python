"""统一模型研究入口；训练完全离线，原始运行包与在用模型只读。"""

import argparse
from pathlib import Path

from app.services import direction_1d_comparison as study
from app.services.direction_1d_protocol import canonical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "train", "replay", "evaluate", "verify"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-run", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        if args.base_run is None:
            parser.error("freeze需要--base-run")
        result = study.freeze(args.base_run, args.run_dir)
    elif args.command in ("train", "replay"):
        result = study.run(args.run_dir, replay=args.command == "replay")
    elif args.command == "evaluate":
        report = study.evaluate(args.run_dir)
        result = {k: report[k] for k in ("overall", "paired", "observed_best", "runtime_model_changed")}
    else:
        result = study.verify(args.run_dir)
    print(canonical(result))


if __name__ == "__main__":
    main()
