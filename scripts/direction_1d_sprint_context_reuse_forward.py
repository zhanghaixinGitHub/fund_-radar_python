"""原研究run.lock下运行独立上下文复用版本，无训练命令，不自动创建或变更任何任务。"""

import argparse
import json
import msvcrt
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import direction_1d_sprint as base  # noqa: E402
from app.services import direction_1d_sprint_context_reuse_runtime as runtime  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "run"))
    parser.add_argument("--round", dest="number", required=True, type=int, choices=tuple(runtime.TERMINALS))
    args = parser.parse_args()
    with (base.ROOT / "run.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print('{"status":"ALREADY_RUNNING"}')
            return
        try:
            print(json.dumps(runtime.execute(args.number, args.action), ensure_ascii=True))
        except Exception as exc:
            value = {
                "at": base.now().isoformat(),
                "type": type(exc).__name__,
                "code": base.error_code(exc),
                "action": args.action,
                "traceback": traceback.format_exc(),
            }
            base.save(runtime.root(args.number) / "last-error.json", value, replace=True)
            print(json.dumps({k: v for k, v in value.items() if k != "traceback"}))
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
