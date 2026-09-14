"""六轮顺序执行入口：原始答案先落盘，第六轮只追加截止前的新信息。"""

import argparse
import json
import msvcrt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import direction_1d_sprint as base  # noqa: E402
from app.services import direction_1d_sprint_dual_us as dual  # noqa: E402
from app.services import direction_1d_sprint_fund_response as response  # noqa: E402
from app.services import direction_1d_sprint_market as market  # noqa: E402
from app.services import direction_1d_sprint_overnight as overnight  # noqa: E402
from app.services import direction_1d_sprint_sparse as sparse  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "train", "preflight", "tick", "run", "report"))
    action = parser.parse_args().action
    with (base.ROOT / "run.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print('{"status":"ALREADY_RUNNING"}')
            return
        try:
            result = (
                {
                    "original": base.tick(),
                    "market": market.tick(),
                    "overnight": overnight.tick(),
                    "sparse": sparse.tick(),
                    "fund_response": response.tick(),
                    "dual_us": dual.tick(),
                }
                if action == "run"
                else getattr(dual, action)()
            )
            print(json.dumps(result, ensure_ascii=True))
            if action in ("run", "tick"):
                base.save(
                    dual.root() / "last-run.json", {"at": base.now().isoformat(), "status": "SUCCEEDED"}, replace=True
                )
        except Exception as exc:
            error = {"at": base.now().isoformat(), "action": action, "error": base.error_code(exc)}
            base.save(dual.root() / "last-error.json", error, replace=True)
            print(json.dumps(error))
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
