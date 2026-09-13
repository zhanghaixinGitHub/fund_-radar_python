"""隔夜数据真实留档入口；供Windows计划任务调用，凭据沿用本机环境且不写入参数或日志。"""

import argparse
import sys
from pathlib import Path

# 计划任务直接调用pythonw而不是交互shell，显式加入项目根，沿用该项目.env和虚拟环境。
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from app.services import direction_1d_overnight_collect as collect  # noqa: E402
from app.services.direction_1d_protocol import canonical  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "tick", "status", "probe"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--capability-proof", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "init":
            if args.capability_proof is None:
                parser.error("init需要原成功SPX最小验权回执--capability-proof")
            result = collect.initialize(args.run_dir, args.capability_proof)
        elif args.command == "tick":
            result = collect.tick(args.run_dir)
        elif args.command == "probe":
            result = collect.probe(args.run_dir)
        else:
            at = collect.now()
            result = {"at": at.isoformat(), "day": collect.day_status(args.run_dir, at.date(), at)}
        if sys.stdout is not None:
            print(canonical(result), flush=True)
        if args.command == "tick" and (
            result["day"]["status"] == "MISSED_DEADLINE" or (result.get("attempt") or {}).get("status") == "FAILED"
        ):
            raise SystemExit(1)
    except Exception as error:
        result = {
            "at": collect.now().isoformat(),
            "status": "FAILED",
            "error_type": type(error).__name__,
            "error_code": str(error)
            if isinstance(error, ValueError) and str(error).startswith("OVERNIGHT_")
            else "OVERNIGHT_COMMAND_FAILED",
        }
        if args.run_dir.is_dir():
            collect.write_health(args.run_dir, result)
        if sys.stdout is not None:
            print(canonical(result), flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
