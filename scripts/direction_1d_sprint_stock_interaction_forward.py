"""第114轮已训模型独立前向入口：共用研究运行锁，不包含任何训练或调参命令。"""

import argparse
import json
import msvcrt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import direction_1d_sprint as base  # noqa: E402
from app.services import direction_1d_sprint_market_stock_interaction as model  # noqa: E402
from app.services import direction_1d_sprint_stock_interaction_forward as forward  # noqa: E402
from app.services import direction_1d_sprint_stock_interaction_runtime as runtime  # noqa: E402

from scripts.direction_1d_sprint_market_kernel_direction import failure  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "preflight", "run", "report"))
    action = parser.parse_args().action
    with (base.ROOT / "run.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print('{"status":"ALREADY_RUNNING"}')
            return
        try:
            result = runtime.plan() if action == "plan" else getattr(forward, action)()
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
