"""七轮执行入口：第七轮复用第五轮输入，先保存它，再执行需要额外IXIC数据的第六轮。"""

import argparse
import json
import msvcrt
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import direction_1d_sprint as base  # noqa: E402
from app.services import direction_1d_sprint_dual_us as dual  # noqa: E402
from app.services import direction_1d_sprint_fund_response as response  # noqa: E402
from app.services import direction_1d_sprint_market as market  # noqa: E402
from app.services import direction_1d_sprint_overnight as overnight  # noqa: E402
from app.services import direction_1d_sprint_return_target as regression  # noqa: E402
from app.services import direction_1d_sprint_sparse as sparse  # noqa: E402


def failure(exc, action):
    """保留异常调用栈的位置，不写可能包含连接信息的异常全文、源码行或局部变量。"""
    return {
        "at": base.now().isoformat(),
        "action": action,
        "error": base.error_code(exc),
        "exception_type": type(exc).__name__,
        "stack": [
            {"file": frame.filename, "line": frame.lineno, "function": frame.name}
            for frame in traceback.extract_tb(exc.__traceback__)
        ],
    }


def run():
    result = {
        "original": base.tick(),
        "market": market.tick(),
        "overnight": overnight.tick(),
        "sparse": sparse.tick(),
        "fund_response": response.tick(),
    }
    errors = {}
    # 两个新分支各自留异常，任一失败都不能令另一个少一次截止前的机会。
    for name, module in (("return_target", regression), ("dual_us", dual)):
        try:
            result[name] = module.tick()
        except Exception as exc:
            error = failure(exc, "tick")
            base.save(module.root() / "last-error.json", error, replace=True)
            errors[name] = error
    if errors:
        raise ValueError("INDEPENDENT_BRANCH_FAILED")
    return result


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
            result = run() if action == "run" else getattr(regression, action)()
            print(json.dumps(result, ensure_ascii=True))
            if action in ("run", "tick"):
                base.save(
                    regression.root() / "last-run.json",
                    {"at": base.now().isoformat(), "status": "SUCCEEDED"},
                    replace=True,
                )
        except Exception as exc:
            error = failure(exc, action)
            base.save(regression.root() / "last-error.json", error, replace=True)
            print(json.dumps(error))
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
