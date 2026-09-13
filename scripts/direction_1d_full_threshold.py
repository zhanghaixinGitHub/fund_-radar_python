"""完整样本重训与较早验证段选阈值；所有输出独占创建，研究模型不注册到业务服务。"""

import argparse
from pathlib import Path

from app.services import direction_1d_full_threshold as study
from app.services.direction_1d_protocol import canonical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "train", "replay", "verify", "evaluate"))
    parser.add_argument("--run-dir", type=Path, required=True)
    for name in ("base-run", "audit-run", "plan", "scope"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        if any(v is None for v in (args.base_run, args.audit_run, args.plan, args.scope)):
            parser.error("freeze需要--base-run、--audit-run、--plan、--scope")
        result = study.freeze(args.base_run, args.audit_run, args.plan, args.scope, args.run_dir)
    elif args.command in ("train", "replay"):
        result = study.run(args.run_dir, replay=args.command == "replay")
    elif args.command == "evaluate":
        value = study.evaluate(args.run_dir)
        result = {k: value[k] for k in ("overall", "paired")}
    else:
        result = study.verify(args.run_dir)
    print(canonical(result))


if __name__ == "__main__":
    main()
