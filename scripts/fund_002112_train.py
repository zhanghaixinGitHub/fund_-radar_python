"""002112 一日离线训练 CLI；只创建本地候选，禁止自动登记或发布。"""

import argparse
import json
import time

from app.services import fund_002112_training as training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "status", "resume", "report"))
    parser.add_argument("--run-id")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument(
        "--interrupt-after-checkpoint",
        choices=[f"{v}:{p}" for v in training.numerical.VARIANTS for p in ("main", "replay")],
        help="仅用于中断恢复验收：模型检查点落盘后立即以 75 退出，不重置拟合预算",
    )
    args = parser.parse_args()
    if args.command != "start" and not args.run_id:
        parser.error("status/resume/report 必须指定 --run-id")
    if args.command != "start" and (args.freeze_only or args.interrupt_after_checkpoint):
        parser.error("验收开关仅允许 start 使用")
    last = 0

    def progress(done, total, code, message):
        nonlocal last
        if time.monotonic() - last >= 15:
            print(json.dumps({"stage": "FREEZE_VERIFY", "done": done, "total": total}, ensure_ascii=False), flush=True)
            last = time.monotonic()

    if args.command == "status":
        result = training.status(args.run_id)
    elif args.command == "report":
        result = training.export_report(args.run_id)
    else:
        result = training.execute(
            args.run_id,
            freeze_only=args.freeze_only,
            interrupt_after_checkpoint=args.interrupt_after_checkpoint,
            progress=progress,
        )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result.get("status") in {"PARTIAL", "FAILED", "BLOCKED"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
