"""第60轮历史更新频率实验入口；当前预测复用已有相同模型，不新建采集任务。"""

import argparse
import json
import msvcrt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services import direction_1d_sprint as base  # noqa: E402
from app.services import direction_1d_sprint_monthly_direct as monthly  # noqa: E402

from scripts.direction_1d_sprint_stacking_monotonic import failure  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "train"))
    action = parser.parse_args().action
    with (base.ROOT / "round60-training.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print('{"status":"ALREADY_RUNNING"}')
            return
        try:
            result = getattr(monthly, action)()
            print(json.dumps(result, ensure_ascii=True))
        except Exception as exc:
            base.save(monthly.root() / "last-error.json", failure(exc, action), replace=True)
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
