"""独立的IXIC历史采集锁，不占用五轮真实预测的run.lock。"""

import json
import msvcrt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import direction_1d_sprint as base  # noqa: E402
from app.services import direction_1d_sprint_ixic_data as data  # noqa: E402


def main():
    data.root().mkdir(parents=True, exist_ok=True)
    with (data.root() / "acquire.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print('{"status":"ALREADY_RUNNING"}')
            return
        try:
            print(json.dumps(data.acquire(), ensure_ascii=True))
        except Exception as exc:
            error = {"at": base.now().isoformat(), "error": base.error_code(exc)}
            base.save(data.root() / "last-error.json", error, replace=True)
            print(json.dumps(error))
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
