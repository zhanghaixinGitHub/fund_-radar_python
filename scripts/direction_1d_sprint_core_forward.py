"""在原Windows任务内优先执行当前一日模型；共享进程锁，失败保存脱敏调用栈。"""

import argparse
import json
import msvcrt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import direction_1d_sprint as base  # noqa: E402
from app.services import direction_1d_sprint_core_forward as core  # noqa: E402

from scripts.direction_1d_sprint_market_kernel_direction import failure  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "run"))
    action = parser.parse_args().action
    with (base.ROOT / "run.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print('{"status":"ALREADY_RUNNING"}')
            return
        try:
            result = getattr(core, action)()
            base.save(
                core.root() / "last-run.json",
                {"at": base.now().isoformat(), "action": action, "status": "SUCCEEDED"},
                replace=True,
            )
            print(json.dumps(result, ensure_ascii=True))
        except Exception as exc:
            value = failure(exc, action)
            base.save(core.root() / "last-error.json", value, replace=True)
            print(json.dumps(value))
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
