"""第103轮研究及前向入口；运行仅接CORE3当前模型，准备训练使用独立互斥锁。"""

import argparse
import json
import msvcrt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import direction_1d_sprint as base  # noqa: E402
from app.services import direction_1d_sprint_market_futures_style as model  # noqa: E402

from scripts.direction_1d_sprint_market_kernel_direction import failure  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "prepare", "preflight", "run", "report"))
    action = parser.parse_args().action
    path = base.ROOT / ("round103-prepare.lock" if action in ("plan", "prepare") else "run.lock")
    with path.open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print('{"status":"ALREADY_RUNNING"}')
            return
        try:
            result = model.tick() if action == "run" else getattr(model, action)()
            if action == "run":
                base.save(
                    model.root() / "last-run.json", {"at": base.now().isoformat(), "status": "SUCCEEDED"}, replace=True
                )
            print(json.dumps(result, ensure_ascii=True))
        except Exception as exc:
            error = failure(exc, action)
            base.save(model.root() / "last-error.json", error, replace=True)
            print(json.dumps(error))
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
