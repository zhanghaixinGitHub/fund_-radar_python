"""保留期复核独立入口；本地历史分析使用独立进程锁，不占用真实预测任务的锁。"""

import argparse
import json
import msvcrt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import direction_1d_sprint as base  # noqa: E402
from app.services import direction_1d_sprint_reserved_audit as audit  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "run", "report"))
    action = parser.parse_args().action
    audit.root().mkdir(parents=True, exist_ok=True)
    with (audit.root() / "audit.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print('{"status":"ALREADY_RUNNING"}')
            return
        try:
            result = base.read(audit.root() / "result.json") if action == "report" else getattr(audit, action)()
            print(json.dumps(result, ensure_ascii=True))
        except Exception as exc:
            error = {"at": base.now().isoformat(), "action": action, "error": base.error_code(exc)}
            base.save(audit.root() / "last-error.json", error, replace=True)
            print(json.dumps(error))
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
