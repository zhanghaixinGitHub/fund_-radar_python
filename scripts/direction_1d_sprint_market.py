"""第二轮命令及Windows合并入口；与首轮命令共用同一个Windows文件锁。"""

import argparse
import json
import msvcrt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import direction_1d_sprint as base  # noqa: E402
from app.services import direction_1d_sprint_market as market  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "acquire", "train", "preflight", "tick", "run", "report"))
    args = parser.parse_args()
    with (base.ROOT / "run.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print('{"status":"ALREADY_RUNNING"}')
            return
        try:
            if args.action == "run":
                original = base.tick()
                result = {"original": original, "market": market.tick()}
            else:
                result = getattr(market, args.action)()
            print(json.dumps(result, ensure_ascii=True))
            if args.action in ("run", "tick"):
                base.save(
                    market.root() / "last-run.json", {"at": base.now().isoformat(), "status": "SUCCEEDED"}, replace=True
                )
        except Exception as exc:
            error = {"at": base.now().isoformat(), "action": args.action, "error": base.error_code(exc)}
            base.save(market.root() / "last-error.json", error, replace=True)
            print(json.dumps(error))
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
