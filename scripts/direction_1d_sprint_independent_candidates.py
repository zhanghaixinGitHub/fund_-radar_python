"""原run.lock下执行独立候选接续；不创建任务、不训练、不自动切换入口。"""

import argparse
import json
import msvcrt
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import direction_1d_sprint as base  # noqa: E402
from app.services import direction_1d_sprint_independent_candidates as runtime  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "run"))
    args = parser.parse_args()
    with (base.ROOT / "run.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print('{"status":"ALREADY_RUNNING"}')
            return
        try:
            result = runtime.execute(args.action)
            print(json.dumps({k: v for k, v in result.items() if k != "result"}, ensure_ascii=True))
        except Exception as exc:
            value = {
                "at": base.now().isoformat(),
                "type": type(exc).__name__,
                "code": base.error_code(exc),
                "action": args.action,
                "traceback": traceback.format_exc(),
            }
            base.save(runtime.root() / "last-error.json", value, replace=True)
            print(json.dumps({k: v for k, v in value.items() if k != "traceback"}))
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
