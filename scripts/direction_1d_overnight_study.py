"""一日SPX隔夜假设性研究入口；仅本地离线训练，不注册模型或触发真实预测。"""

import argparse
from pathlib import Path

from app.services import direction_1d_overnight_study as study
from app.services.direction_1d_protocol import canonical
from threadpoolctl import threadpool_limits


@threadpool_limits.wrap(limits=1)
def main() -> None:
    """训练与只读复核统一单线程，避免并行求和末位差异触发严格尺度一致性检查。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "train", "replay", "verify", "evaluate"))
    parser.add_argument("--run-dir", type=Path, required=True)
    for name in ("base-run", "data-run", "plan", "scope", "source-status"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        if any(v is None for v in (args.base_run, args.data_run, args.plan, args.scope, args.source_status)):
            parser.error("freeze需要--base-run、--data-run、--plan、--scope、--source-status")
        result = study.freeze(args.base_run, args.data_run, args.plan, args.scope, args.source_status, args.run_dir)
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
