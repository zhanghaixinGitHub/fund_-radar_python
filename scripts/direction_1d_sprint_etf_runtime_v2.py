"""ETF预测验证第二版入口：保留第47轮之前的执行链，独立接续两个冻结模型。"""

import argparse
import json
import msvcrt
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import direction_1d_sprint as base  # noqa: E402
from app.services import direction_1d_sprint_etf_runtime_v2 as runtime  # noqa: E402

from scripts import direction_1d_sprint_convertible as existing  # noqa: E402


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
    result, errors = {}, {}
    # 新旧分支独立尝试，任一分支失败均留下明确失败状态。
    for name, action in (
        ("existing_through_round47", existing.run),
        ("round48", lambda: runtime.tick(runtime.r48)),
        ("round49", lambda: runtime.tick(runtime.r49)),
    ):
        try:
            result[name] = action()
        except Exception as exc:
            errors[name] = failure(exc, "run")
    if errors:
        base.save(runtime.folder() / "branch-errors.json", errors, replace=True)
        raise ValueError("INDEPENDENT_BRANCH_FAILED")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "preflight", "run"))
    action = parser.parse_args().action
    lock_path = (
        base.ROOT / "run.lock" if action in ("run", "tick", "report") else base.ROOT / "runtime-etf-v2-preflight.lock"
    )
    with lock_path.open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print('{"status":"ALREADY_RUNNING"}')
            return
        try:
            result = run() if action == "run" else getattr(runtime, action)()
            print(json.dumps(result, ensure_ascii=True))
            if action in ("run", "tick"):
                base.save(
                    runtime.folder() / "last-run.json",
                    {"at": base.now().isoformat(), "status": "SUCCEEDED"},
                    replace=True,
                )
        except Exception as exc:
            error = failure(exc, action)
            base.save(runtime.folder() / "last-error.json", error, replace=True)
            print(json.dumps(error))
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
