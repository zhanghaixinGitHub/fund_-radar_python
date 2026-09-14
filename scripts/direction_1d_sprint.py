"""三天研究命令；Windows文件锁随进程退出释放，防止手动与定时任务重复写入。"""

import argparse
import json
import msvcrt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import direction_1d_sprint as study  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("initialize", "snapshot", "train", "refresh", "tick", "report"))
    args = parser.parse_args()
    study.ROOT.mkdir(parents=True, exist_ok=True)
    with (study.ROOT / "run.lock").open("a+b") as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print('{"status":"ALREADY_RUNNING"}')
            return
        try:
            result = getattr(study, args.action)()
            if args.action == "refresh":
                result = {"at": result["at"], "funds": len(result["funds"]), "errors": result["errors"]}
            print(json.dumps(result, ensure_ascii=True))
        except Exception as exc:
            failure = {"at": study.now().isoformat(), "action": args.action, "error": study.error_code(exc)}
            study.save(study.ROOT / "last-error.json", failure, replace=True)
            print(json.dumps(failure))
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
