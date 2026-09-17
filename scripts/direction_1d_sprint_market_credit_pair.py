"""第112轮独立训练入口；前向层另行绑定，不改变Windows运行链。"""

import argparse
import json
import msvcrt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import direction_1d_sprint as base  # noqa: E402
from app.services import direction_1d_sprint_market_credit_pair as model  # noqa: E402

from scripts.direction_1d_sprint_market_kernel_direction import failure  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "prepare"))
    action = parser.parse_args().action
    path = base.ROOT / "round112-prepare.lock"
    with path.open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print('{"status":"ALREADY_RUNNING"}')
            return
        try:
            result = getattr(model, action)()
            print(json.dumps(result, ensure_ascii=True))
        except Exception as exc:
            error = failure(exc, action)
            base.save(model.root() / "last-error.json", error, replace=True)
            print(json.dumps(error))
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
