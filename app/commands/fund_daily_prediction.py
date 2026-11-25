"""唯一00:30任务使用的通用续办入口；每次推进同一期，不接受基金白名单或固定目标日。"""

import argparse
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from sys import argv
from time import monotonic, sleep

from app.services.fund_daily_prediction import TERMINAL, DailyPredictionRun, supplement_run_id
from app.services.fund_daily_runtime import PROJECT, LocalDailyBackend
from app.services.fund_exposure_common import now
from app.services.material_content import material_read_cache


@contextmanager
def run_lock(path=None, *, wait=False):
    """默认非阻塞获取共享业务锁；短提交可等待当前写入退出。"""
    import msvcrt

    path = path or PROJECT / ".local-runs/daily-prediction.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK if wait else msvcrt.LK_NBLCK, 1)
        except OSError:
            raise ValueError("DAILY_PREDICTION_RUN_ALREADY_ACTIVE") from None
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def fund_execution_lock(directory, code):
    if not code or len(code) != 6 or not code.isascii() or not code.isdecimal():
        raise ValueError("DAILY_FUND_CODE_INVALID")
    return run_lock(directory / "fund-business" / code / "execution.lock")


@material_read_cache
def main():
    # 本人读回及逐周期保存会复用同一批不可变原文，沿用64MB有界缓存，
    # 不在每份快照中反复从数据库加载同文；写后读的新版本仍独立核验。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="恢复原运行时指定；首次自动使用北京时间当日独立目录")
    parser.add_argument("--reviews-file", type=Path, help="当前聊天实际阅读后的逐基金综合用途审阅映射")
    parser.add_argument("--blocked-file", type=Path, help="按基金/周期/目标绑定的真实外部阻断证据")
    parser.add_argument("--resume-batch", help="首次接管已经提交的纯资料jobId；后续一律使用台账内原jobId")
    parser.add_argument("--resume-only", action="store_true", help="只允许人工续办已有运行和窗口")
    parser.add_argument("--publish-prepared", action="store_true",
                        help="实际阅读并写入审阅文件后续办；仍核当前资料版本，不重复采集已有候选")
    parser.add_argument("--refresh-checklist", metavar="基金代码", help="只用本轮台账和最新档案重画指定基金的补数清单")
    parser.add_argument("--supplements-only", action="store_true", help="在已有运行中按原窗口逐项核查和补件")
    parser.add_argument("--initialize-only", action="store_true", help="总指挥初始化真实持仓和今日清单后返回，不启动全量业务")
    args = parser.parse_args()
    from app.services.prediction_trial_v2 import storage_check

    instant = now()
    storage_check(instant.replace(hour=0, minute=0, second=0, microsecond=0).isoformat())
    directory = args.run_dir or PROJECT / ".local-runs" / ("daily-prediction-" + now().strftime("%Y%m%d"))
    if args.supplements_only:
        from app.services.fund_daily_services import ensure_services
        from app.services.fund_daily_supplements import record_supplement_summary
        from app.services.fund_exposure_common import read

        directory = directory.resolve()
        if directory.parent != (PROJECT / ".local-runs").resolve():
            parser.error("逐项补件只接受本项目 .local-runs 下的原运行目录")
        state = read(directory / "run.json")
        job = supplement_run_id(state)
        direct = bool(state.get("supplement_run_id"))
        if not direct:
            terminal = read(directory / "materials-status.json")
            if not terminal["terminal"] or terminal["job"]["job_id"] != job:
                raise ValueError("DAILY_SUPPLEMENT_ORIGINAL_JOB_NOT_TERMINAL")
        ensure_services(directory)
        backend = LocalDailyBackend(directory)
        try:
            at, owner, scope, live = backend.context()
            if (owner != state["owner_user_id"] or state["window"] != live
                    or set(scope) != set(state["scope"]) or at[:10] != state["created_at"][:10]):
                raise ValueError("DAILY_SUPPLEMENT_ORIGINAL_SCOPE_OR_WINDOW_CHANGED")
            current = {**state, "checked_at": at}
            options = {"recover_materials": backend.recover_materials} if direct else {}
            supplement = backend.supplements(current, backend.save_profiles(scope)).process(current, **options)
            record_supplement_summary(directory, current, supplement)
        finally:
            backend.close()
        print(json.dumps(supplement, ensure_ascii=False))
        raise SystemExit(0 if supplement["status"] == "SUCCEEDED" else 2)
    if args.refresh_checklist:
        from app.services.fund_daily_profiles import (
            VERIFIED_SOURCE_RESULTS,
            profiles,
            save_fund_checklist,
            save_pending,
        )
        from app.services.fund_exposure_common import read

        directory = directory.resolve()
        if directory.parent != (PROJECT / ".local-runs").resolve():
            parser.error("补数清单只从本项目 .local-runs 下的已有运行目录刷新")
        state = read(directory / "run.json")
        code = args.refresh_checklist
        if code not in state["scope"]:
            parser.error("基金不在本轮持仓范围")
        profile = read(directory / "fund-profiles.json")[code]
        profile.update({k: v for k, v in profiles().get(code, {}).items() if v is not None})
        ledger = directory / "supplement-funds" / (code + ".json")
        audit = read(ledger) if ledger.is_file() else {}
        if (audit and audit.get("job_id") == supplement_run_id(state)
                and audit.get("day") == instant.date().isoformat() and audit.get("window") == state["window"]):
            rows = audit["items"]
            verified = {row["field"]: row for row in rows if row["status"] in VERIFIED_SOURCE_RESULTS}
            tasks = [{**row, "source_review": row} for row in rows if row["status"] not in VERIFIED_SOURCE_RESULTS]
            save_fund_checklist(directory.parent / "fund-supplements", code, profile, state["window"],
                                audit["day"], tasks, set(verified), verified=verified)
        else:
            tasks = save_pending(directory, state, records={code: profile}, fund_code=code, aggregate=False)
        print(json.dumps({"fund_code": code, "pending_count": len(tasks),
                          "checklist": str(directory.parent / "fund-supplements" / f"{code}.md")}, ensure_ascii=False))
        return
    resume_receipt = None
    if args.resume_only:
        from app.services.fund_exposure_common import read
        from app.services.fund_materials_store import versioned_save

        state_path = directory / "run.json"
        original = read(state_path) if state_path.exists() else {}
        if not supplement_run_id(original):
            parser.exit(2, "DAILY_RESUME_JOB_MISSING\n")
        resume_receipt = {
            "started_at": instant.isoformat(),
            "window": original["window"],
            "job_id": supplement_run_id(original),
            "status_before": original.get("status"),
            "reviewed_before": original.get("reviewed_period_count"),
            "pending_before": original.get("pending_funds"),
        }
        versioned_save(directory / "morning-resume.json", resume_receipt)
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
        runner = DailyPredictionRun(directory, backend)
        inputs = {
            "reviews": json.loads(args.reviews_file.read_text(encoding="utf-8")) if args.reviews_file else None,
            "blocked_file": args.blocked_file, "publish_prepared": args.publish_prepared, "resume_only": args.resume_only,
        }
        if args.initialize_only:
            inputs["initialize_only"] = True
        result = runner.advance(**inputs)
        if result["status"] == "WAITING_MATERIALS":
            print(json.dumps({"status": result["status"], "job_id": result["batch"]["job_id"]}), flush=True)
        last_storage_check = monotonic()
        while result["status"] == "WAITING_MATERIALS":
            deadline = datetime.fromisoformat(result["window"]["deadline_at"])
            if now() >= deadline:
                result = runner.advance(**inputs)
                break
            # 只查原作业状态。等待期间不重复提交、回放原件或重新读完整预测档案。
            sleep(min(10, max(0, (deadline - now()).total_seconds())))
            if monotonic() - last_storage_check >= 25:
                storage_check(instant.replace(hour=0, minute=0, second=0, microsecond=0).isoformat())
                last_storage_check = monotonic()
            job = backend.get(backend.base + "/" + result["batch"]["job_id"])
            if job["status"] in TERMINAL:
                result = runner.advance(**inputs)
    finally:
        backend.close()
    if resume_receipt is not None:
        resume_receipt.update(
            finished_at=now().isoformat(),
            status_after=result["status"],
            reviewed_after=result.get("reviewed_period_count"),
            pending_after=result.get("pending_funds"),
        )
        versioned_save(directory / "morning-resume.json", resume_receipt)
    print(
        json.dumps(
            {
                "status": result["status"],
                "fund_count": len(result.get("scope", [])),
                "pending_funds": result.get("pending_funds", []),
                "run_dir": str(directory.resolve()),
                "work_items": str((directory / "work-items.json").resolve()),
                "fund_checklists": str((directory.resolve().parent / "fund-supplements")),
                "all_predictions_generated": result.get("all_predictions_generated", False),
                "all_reviews_saved": result.get("all_reviews_saved", False),
                "reviewed_period_count": result.get("reviewed_period_count", 0),
                "unreviewed_period_count": result.get("unreviewed_period_count", 0),
                "abstain_saved_count": result.get("abstain_saved_count", 0),
                "blocked_saved_count": result.get("blocked_saved_count", 0),
                "reassessment_pending_count": result.get("reassessment_pending_count", 0),
                "reassessment_check_failure_count": result.get("reassessment_check_failure_count", 0),
                "monthly_compilation_pending": result.get("monthly_compilation", {}).get("pending"),
                "advice": {key: value for key, value in result.get("advice", {}).items()
                           if key in {"status", "reports_read_back", "actionable_count", "counts"}},
                "supplements": {key: value for key, value in result.get("supplements", {}).items()
                                if key in {"status", "item_count", "pending_item_count", "counts"}},
                "one_day_outcomes": {
                    k: v for k, v in result.get("one_day_outcomes", {}).items() if k in {"status", "summary", "reason"}
                },
            },
            ensure_ascii=False,
        )
    )
    raise SystemExit(0 if result["status"] == "CLOSED" else 2)


if __name__ == "__main__":
    refresh = any(arg.split("=", 1)[0] == "--refresh-checklist" for arg in argv[1:])
    with run_lock(wait=refresh):
        main()
