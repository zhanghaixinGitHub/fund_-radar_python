"""同一机器的来源请求共享预约时间，避免多个进程各自限流仍超额。"""

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data/source-budgets"


@contextmanager
def _locked(path: Path):
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def wait_for_tushare_slot(rate: int = 200, *, root: Path = ROOT) -> None:
    """只在真实外部请求前预约；文件不含账号/令牌，锁外等待，最长60秒。"""
    if not 1 <= rate <= 200:
        raise ValueError("来源请求预算不合法")
    root.mkdir(parents=True, exist_ok=True)
    state = root / "tushare.json"
    with _locked(root / "tushare.lock"):
        now = time.time()
        previous = json.loads(state.read_text(encoding="utf-8"))["next"] if state.exists() else now
        slot = max(now, previous)
        if slot - now > 60:
            raise RuntimeError("来源请求预算暂时不足，请稍后重试")
        temporary = root / f"tushare-{os.getpid()}.tmp"
        temporary.write_text(json.dumps({"next": slot + 60 / rate}), encoding="utf-8")
        os.replace(temporary, state)
    if slot > now:
        time.sleep(slot - now)
