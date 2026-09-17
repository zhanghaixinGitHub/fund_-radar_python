"""第四十七轮入口：独立验证可转债基准对一日方向的增量。"""

import argparse
import json
import msvcrt
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import direction_1d_sprint as base  # noqa: E402
from app.services import direction_1d_sprint_convertible as convertible  # noqa: E402

from scripts import direction_1d_sprint_food as existing  # noqa: E402


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
    for name, action in (("existing_through_round46", existing.run), ("convertible", convertible.tick)):
        try:
            result[name] = action()
        except Exception as exc:
            errors[name] = failure(exc, "run")
    if errors:
        base.save(convertible.root() / "branch-errors.json", errors, replace=True)
        raise ValueError("INDEPENDENT_BRANCH_FAILED")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "train", "preflight", "tick", "run", "report"))
    action = parser.parse_args().action
    lock_path = base.ROOT / "run.lock" if action in ("run", "tick", "report") else base.ROOT / "round47-training.lock"
    with lock_path.open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print('{"status":"ALREADY_RUNNING"}')
            return
        try:
            result = run() if action == "run" else getattr(convertible, action)()
            print(json.dumps(result, ensure_ascii=True))
            if action in ("run", "tick"):
                base.save(
                    convertible.root() / "last-run.json",
                    {"at": base.now().isoformat(), "status": "SUCCEEDED"},
                    replace=True,
                )
        except Exception as exc:
            error = failure(exc, action)
            base.save(convertible.root() / "last-error.json", error, replace=True)
            print(json.dumps(error))
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
