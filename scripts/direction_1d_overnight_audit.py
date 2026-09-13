"""一日隔夜特征只读资格审计入口；只支持审计和复核，不提供训练或联网采集命令。"""

import argparse
from pathlib import Path

from app.services import direction_1d_overnight_audit as audit
from app.services.direction_1d_protocol import canonical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "verify"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--base-run", type=Path)
    parser.add_argument("--data-run", type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        if args.base_run is None or args.data_run is None:
            parser.error("audit需要--base-run与--data-run")
        result = audit.save_audit(args.base_run, args.data_run, args.run_dir)
    else:
        result = audit.verify(args.run_dir)
    print(canonical(result), flush=True)


if __name__ == "__main__":
    main()
