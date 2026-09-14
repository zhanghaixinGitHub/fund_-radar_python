"""一日固定校准研究入口：封存、4次校准、唯一复现、只读核验和评估。"""

import argparse
from pathlib import Path

from app.services import direction_1d_calibration_study as study
from app.services.direction_1d_protocol import canonical
from threadpoolctl import threadpool_limits


@threadpool_limits.wrap(limits=1)
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "train", "replay", "verify", "evaluate"))
    parser.add_argument("--run-dir", type=Path, required=True)
    for name in ("base-run", "plan", "scope"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        if any(v is None for v in (args.base_run, args.plan, args.scope)):
            parser.error("freeze需要--base-run、--plan、--scope")
        result = study.freeze(args.base_run, args.plan, args.scope, args.run_dir)
    elif args.command in ("train", "replay"):
        result = study.run(args.run_dir, replay=args.command == "replay")
    elif args.command == "evaluate":
        report = study.evaluate(args.run_dir)
        result = {k: report[k] for k in ("paired", "conditions", "status", "reliability_metric_differences")}
        result["overall"] = {
            b: {k: v for k, v in m.items() if not k.endswith("_bins")} for b, m in report["overall"].items()
        }
    else:
        result = study.verify(args.run_dir)
    print(canonical(result), flush=True)


if __name__ == "__main__":
    main()
