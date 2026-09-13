"""一日树复杂度研究入口：离线有界执行，不注册或替换页面模型。"""

import argparse
from pathlib import Path

from app.services import direction_1d_complexity_study as study
from app.services.direction_1d_protocol import canonical
from app.services.direction_1d_training import write_new
from threadpoolctl import threadpool_limits


@threadpool_limits.wrap(limits=1)
def main() -> None:
    """统一训练/复核的线程数，避免并行求和造成尺度末位变化。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("diagnose", "freeze", "train", "replay", "verify", "evaluate"))
    parser.add_argument("--run-dir", type=Path, required=True)
    for name in ("base-run", "plan", "diagnosis", "scope"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.command == "diagnose":
        if args.base_run is None:
            parser.error("diagnose需要--base-run")
        report = study.diagnose(args.base_run)
        args.run_dir.mkdir(parents=True, exist_ok=True)
        write_new(args.run_dir / "diagnosis.json", report)
        result = report["overall"]
    elif args.command == "freeze":
        if any(v is None for v in (args.base_run, args.plan, args.diagnosis, args.scope)):
            parser.error("freeze需要--base-run、--plan、--diagnosis、--scope")
        result = study.freeze(args.base_run, args.plan, args.diagnosis, args.scope, args.run_dir)
    elif args.command in ("train", "replay"):
        result = study.run(args.run_dir, replay=args.command == "replay")
    elif args.command == "evaluate":
        report = study.evaluate(args.run_dir)
        result = {k: report[k] for k in ("overall", "paired", "conditions", "status")}
    else:
        result = study.verify(args.run_dir)
    print(canonical(result), flush=True)


if __name__ == "__main__":
    main()
