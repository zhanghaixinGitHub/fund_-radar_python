"""唯一00:30任务使用的通用续办入口；每次推进同一期，不接受基金白名单或固定目标日。"""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

from app.services.fund_daily_prediction import DailyPredictionRun
from app.services.fund_daily_runtime import PROJECT, LocalDailyBackend
from app.services.fund_exposure_common import now
from app.services.material_content import material_read_cache


@contextmanager
def run_lock():
    """跨运行目录也只允许一个续办进程；非阻塞获取，锁忙不写成功或重复提交。"""
    import msvcrt

    path = PROJECT / ".local-runs/daily-prediction.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise ValueError("DAILY_PREDICTION_RUN_ALREADY_ACTIVE") from None
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


@material_read_cache
def main():
    # 本人读回及逐周期保存会复用同一批不可变原文，沿用64MB有界缓存，
    # 不在每份快照中反复从数据库加载同文；写后读的新版本仍独立核验。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="恢复原运行时指定；首次自动使用北京时间当日独立目录")
    parser.add_argument("--reviews-file", type=Path, help="当前聊天实际阅读后的逐基金综合用途审阅映射")
    parser.add_argument("--blocked-file", type=Path, help="按基金/周期/目标绑定的真实外部阻断证据")
    parser.add_argument("--resume-batch", help="首次接管已经提交的纯资料jobId；后续一律使用台账内原jobId")
    args = parser.parse_args()
    directory = args.run_dir or PROJECT / ".local-runs" / ("daily-prediction-" + now().strftime("%Y%m%d"))
    from app.services.fund_daily_services import ensure_services

    ensure_services(directory)
    backend = LocalDailyBackend(directory)
    if args.resume_batch and not (directory / "run.json").exists():
        # 接管也需真实核验纯资料模式与作业时间，不接受旧期作业冒充本次同步。
        from app.services.fund_materials_store import versioned_save

        at, owner, scope, current = backend.context()
        batch = {"job_id": args.resume_batch, "explicit_resume": True}
        status = backend.batch_status(batch)
        started = status["job"].get("started_at") or ""
        if started[:10] != current["target_nav_date"]:
            from datetime import datetime

            from app.services.direction_1d_protocol import ZONE

            if (
                not started
                or datetime.fromisoformat(started).astimezone(ZONE).date().isoformat() != current["target_nav_date"]
            ):
                raise ValueError("RESUME_BATCH_WRONG_PREDICTION_WINDOW")
        versioned_save(
            directory / "run.json",
            {
                "owner_user_id": owner,
                "window": current,
                "scope": scope,
                "created_at": at,
                "funds": {},
                "batch": batch,
                "first_failures": [],
            },
        )
    try:
        result = DailyPredictionRun(directory, backend).advance(
            reviews=json.loads(args.reviews_file.read_text(encoding="utf-8")) if args.reviews_file else None,
            blocked_file=args.blocked_file,
        )
    finally:
        backend.close()
    print(
        json.dumps(
            {
                "status": result["status"],
                "fund_count": len(result.get("scope", [])),
                "pending_funds": result.get("pending_funds", []),
                "run_dir": str(directory.resolve()),
                "work_items": str((directory / "work-items.json").resolve()),
                "all_predictions_generated": result.get("all_predictions_generated", False),
            },
            ensure_ascii=False,
        )
    )
    raise SystemExit(0 if result["status"] == "CLOSED" else 2)


if __name__ == "__main__":
    with run_lock():
        main()
