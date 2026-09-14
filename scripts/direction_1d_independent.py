"""一日独立验证CLI：P0审计、只读恢复及最小真实输入诊断；无训练/发布入口。"""

import argparse
import sys
from pathlib import Path
from uuid import uuid4

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from app.services import direction_1d_independent as protocol  # noqa: E402
from app.services import direction_1d_independent_audit as audit  # noqa: E402
from app.services import direction_1d_independent_data as data  # noqa: E402
from app.services import direction_1d_independent_runtime as runtime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    prepare = subs.add_parser("prepare")
    prepare.add_argument("--candidate-root", type=Path, required=True)
    prepare.add_argument("--initial-root", type=Path, required=True)
    prepare.add_argument("--identity-evidence", type=Path, required=True)
    prepare.add_argument("--core-env", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    for action in ("verify", "probe", "start-check", "input-acceptance", "activate", "tick", "report", "active-check"):
        command = subs.add_parser(action)
        command.add_argument("--run-dir", type=Path, required=True)
        if action == "probe":
            command.add_argument("--out", type=Path, required=True)
            command.add_argument("--market-evidence", type=Path, help="复用已保存原始响应及接收回执，避免重复请求")
    args = parser.parse_args()

    def emit(value):
        if sys.stdout is not None:
            print(protocol.canonical(value), flush=True)
        if args.action == "tick" and args.run_dir.exists():
            protocol.seal(
                args.run_dir / "invocations" / (uuid4().hex + ".json"), {"at": protocol.now().isoformat(), **value}
            )

    try:
        if args.action == "prepare":
            result = audit.prepare(
                args.out, args.candidate_root, args.initial_root, args.identity_evidence, args.core_env
            )
        elif args.action == "probe":
            result = data.probe(args.run_dir, args.out, args.market_evidence)
        elif args.action == "start-check":
            audit.require_startable(args.run_dir)
            result = {"status": "READY_TO_ACTIVATE"}
        elif args.action == "input-acceptance":
            result = runtime.input_acceptance(args.run_dir)
        elif args.action == "activate":
            result = runtime.activate(args.run_dir)
        elif args.action == "tick":
            result = runtime.tick(args.run_dir)
        elif args.action == "report":
            result = runtime.report(args.run_dir)
        elif args.action == "active-check":
            runtime.load(args.run_dir)
            result = {"status": "ACTIVE_CONTRACT_VERIFIED"}
        else:
            state = audit.verify(args.run_dir)
            draft = state["draft"]
            result = {
                "status": "VERIFIED_P0_EVIDENCE",
                "study_status": draft["status"],
                "blockers": draft["blockers"],
                "owner_count": len(draft["coverage"]),
                "eligible_count": len(draft["members"]),
                "new_fits": 0,
                "trial_prediction_count": 0,
            }
        emit(result)
        return 1 if result.get("status") == "DEGRADED" else 0
    except ValueError as exc:
        # 仅输出本模块稳定错误码；SQL/HTTP异常不输出带参数的异常正文。
        message = str(exc)
        safe = (
            message
            if message and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789:," for c in message)
            else "VALIDATION_FAILED"
        )
        emit({"status": "BLOCKED", "reason": safe, **runtime.failure(exc)})
        return 2
    except Exception as exc:
        emit({"status": "FAILED", "error_type": type(exc).__name__, **runtime.failure(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
