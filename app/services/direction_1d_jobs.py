"""持久化作业状态与独立单线程；Java负责个人范围、先核对后训练和调度时点。"""

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from threading import BoundedSemaphore
from uuid import uuid4

from sqlalchemy import text

from app.db.session import get_engine
from app.repositories import direction_1d as repo
from app.services.direction_1d_data import inventory
from app.services.direction_1d_protocol import PROTOCOL, canonical, window

logger = logging.getLogger(__name__)
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="direction-1d")
slots = BoundedSemaphore(32)


def sync_missing(codes):
    """仅为适用基金补实际缺口；复用原限频及同步服务，单轮每基金一次，不购权。"""
    from app.services.tushare_fund_sync import TushareFundSyncService

    result = []
    coverage = inventory(codes)
    service = TushareFundSyncService()
    for f in coverage["items"]:
        if not f.get("group_id") or not f.get("missing_dates"):
            continue
        with get_engine().connect() as lock:
            if not lock.execute(
                text("SELECT pg_try_advisory_lock(721104,:code)"), {"code": int(f["fund_code"])}
            ).scalar():
                result.append({"fund_code": f["fund_code"], "status": "SYNC_IN_PROGRESS"})
                continue
            try:
                with get_engine().connect() as c:
                    ts_code = c.execute(
                        text("SELECT source_fund_code FROM fund_share_class WHERE fund_code=:code"),
                        {"code": f["fund_code"]},
                    ).scalar_one()
                outcome = service.sync_market_nav_history(
                    (ts_code,),
                    start_date=date.fromisoformat(min(f["missing_dates"])),
                    end_date=date.fromisoformat(max(f["missing_dates"])),
                )
                result.append(
                    {
                        "fund_code": f["fund_code"],
                        "status": "SUCCEEDED",
                        "run_id": str(outcome.sync_run_id),
                        "fetched": outcome.fetched_count,
                        "created": outcome.created_count,
                        "updated": outcome.updated_count,
                    }
                )
            except Exception as error:
                # 失败保留接口及类型，不把可能带Token/响应原文的异常字符串回传。
                result.append(
                    {
                        "fund_code": f["fund_code"],
                        "status": "DATA_PENDING",
                        "api": "fund_nav",
                        "error_type": type(error).__name__,
                    }
                )
            finally:
                lock.execute(text("SELECT pg_advisory_unlock(721104,:code)"), {"code": int(f["fund_code"])})
    return {"actions": result, "after": inventory(codes)}


def submit_forecast(code):
    w = window(repo.clock())
    if w["status"] != "OPEN":
        raise ValueError("MISSED_DEADLINE")
    key = f"{PROTOCOL}:{code}:{w['target_nav_date']}"
    return submit(key, "FORECAST", {"fund_code": code})


def submit(key, kind, payload):
    if not slots.acquire(blocking=False):
        raise ValueError("QUEUE_FULL")
    job_id = uuid4()
    try:
        with get_engine().begin() as c:
            c.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:key,721107))"), {"key": key})
            previous = (
                c.execute(text("SELECT * FROM direction_1d_job WHERE task_key=:key"), {"key": key}).mappings().first()
            )
            if previous:
                if previous["state"] == "SUCCEEDED" or (
                    previous["state"] in {"RUNNING", "QUEUED"}
                    and previous["lease_until"]
                    and previous["lease_until"] > repo.clock()
                ):
                    slots.release()
                    return {"job_id": str(previous["job_id"]), "state": previous["state"]}
                # 失败每30分钟重试，不追加已成功分支；训练每周不得重新拟合。
                if kind == "TRAINING" or (
                    previous["finished_at"] and repo.clock() - previous["finished_at"] < timedelta(minutes=30)
                ):
                    slots.release()
                    return {"job_id": str(previous["job_id"]), "state": previous["state"]}
                job_id = previous["job_id"]
                c.execute(
                    text(
                        "UPDATE direction_1d_job SET state='QUEUED',result=NULL,finished_at=NULL,"
                        "lease_until=clock_timestamp()+interval '30 minutes' WHERE job_id=:id"
                    ),
                    {"id": job_id},
                )
            else:
                c.execute(
                    text("""INSERT INTO direction_1d_job(job_id,task_key,kind,state,payload,lease_until)
                  VALUES(:id,:key,:kind,'QUEUED',CAST(:p AS jsonb),clock_timestamp()+interval '30 minutes')"""),
                    {"id": job_id, "key": key, "kind": kind, "p": canonical(payload)},
                )
        executor.submit(execute, job_id, kind, payload)
        return {"job_id": str(job_id), "state": "QUEUED"}
    except Exception:
        slots.release()
        raise


def execute(job_id, kind, payload):
    """持有作业数据库锁执行；租约过期后的重复worker不得同时计算。"""
    with get_engine().connect() as lock:
        locked = lock.execute(
            text("SELECT pg_try_advisory_lock(hashtextextended(:key,721113))"), {"key": str(job_id)}
        ).scalar()
        if not locked:
            slots.release()
            return
        try:
            _execute_locked(job_id, kind, payload)
        finally:
            lock.execute(text("SELECT pg_advisory_unlock(hashtextextended(:key,721113))"), {"key": str(job_id)})


def _execute_locked(job_id, kind, payload):
    from app.services.direction_1d_inference import infer

    try:
        with get_engine().begin() as c:
            # 租约恢复后排队的旧worker也可能晚到；拿到执行锁后再核对成功状态。
            state = c.execute(text("SELECT state FROM direction_1d_job WHERE job_id=:id"), {"id": job_id}).scalar_one()
            if state == "SUCCEEDED":
                return
            c.execute(text("UPDATE direction_1d_job SET state='RUNNING' WHERE job_id=:id"), {"id": job_id})
        if kind == "FORECAST":
            sync_missing([payload["fund_code"]])
            result = infer(payload["fund_code"])
        elif kind == "LABEL_SYNC":
            result = sync_answer(payload)
        else:
            result = weekly_train()
        state = "FAILED" if kind == "TRAINING" and result.get("status") == "FAILED" else "SUCCEEDED"
    except Exception as error:
        # 预期拒绝只返回有限原因码；栈放服务日志，不把SQL参数或来源响应外传。
        reason = (
            str(error)
            if isinstance(error, ValueError) and str(error).isupper() and len(str(error)) < 80
            else "INTERNAL_FAILURE"
        )
        result, state = {"reason": reason}, "FAILED"
        logger.error(
            "Direction1d.execute   >>>   jobId=%s kind=%s errorType=%s",
            job_id,
            kind,
            type(error).__name__,
            exc_info=(type(error), ValueError(reason), error.__traceback__),
        )
    finally:
        slots.release()
    with get_engine().begin() as c:
        c.execute(
            text(
                "UPDATE direction_1d_job SET state=:state,result=CAST(:result AS jsonb),"
                "finished_at=clock_timestamp() WHERE job_id=:id"
            ),
            {"id": job_id, "state": state, "result": canonical(result)},
        )


def weekly_train():
    """固定初始名单，只吸收Java已完成核对的公共原始样本；无新增时零拟合。"""
    from app.services import direction_1d_training as t

    with get_engine().connect() as c:
        runs = c.execute(text("SELECT * FROM direction_1d_training_run ORDER BY created_at LIMIT 1")).mappings().first()
        if not runs:
            return {"status": "MODEL_PENDING"}
        frozen = runs["spec"]
        directory = frozen["artifact_directory"]
        root = (t.RUN_ROOT / directory).resolve()
        if root.parent != t.RUN_ROOT.resolve() or not directory.startswith("direction-1d-"):
            raise ValueError("TRAINING_PATH_INVALID")
        original = t.read(root / "history.json")
        # 答案必须已有Java核对回执，且训练时已可见，暂停/取消关注不会丢历史。
        rows = (
            c.execute(
                text("""SELECT s.payload, s.task_key, s.content_hash, a.assessed_at, i.payload AS input
          FROM direction_1d_snapshot s JOIN direction_1d_assessment_ack a
            ON a.task_key=s.task_key AND a.label_hash=s.content_hash
          JOIN direction_1d_snapshot i ON i.snapshot_id=(s.payload->>'input_snapshot_id')::uuid
          WHERE s.kind='LABEL' AND s.expires_at>clock_timestamp() AND a.assessed_at<=clock_timestamp()
            AND s.payload->>'training_eligible'='true'
          ORDER BY s.as_of LIMIT 50001""")
            )
            .mappings()
            .all()
        )
        if len(rows) > 50000:
            raise ValueError("ASSESSED_SAMPLE_LIMIT_REACHED")
        if not rows:
            return {"status": "SKIPPED", "reason": "NO_NEW_ASSESSED_SAMPLES", "fits": 0}
        known = {f["fund_code"]: f for f in original["funds"]}
        samples = {}
        for r in rows:
            inp, answer = r["input"], r["payload"]
            code = inp["fund_code"]
            if code not in known:
                continue
            samples[r["task_key"]] = {
                "fund_code": code,
                "family": known[code]["product_family_id"],
                "group": known[code]["group_id"],
                "t": inp["base_nav_date"],
                "u": inp["target_nav_date"],
                "x": inp["features"],
                "y": answer["y"],
                "actual_direction": answer["actual_direction"],
                "mature_at": r["assessed_at"].isoformat(),
                "momentum": int(float(inp["values"][-1]["unit_nav"]) > float(inp["values"][-2]["unit_nav"])),
                "input_hash": t.digest(inp),
                "label_hash": r["content_hash"],
                "kind": "FORWARD_ORIGINAL",
            }
    original["forward_samples"] = list(samples.values())
    with get_engine().connect() as c:
        latest = c.execute(
            text("SELECT spec FROM direction_1d_training_run WHERE cohort_id=:cohort ORDER BY created_at DESC LIMIT 1"),
            {"cohort": frozen["cohort_id"]},
        ).scalar_one()
        if latest["data_hash"] == t.digest(original):
            return {"status": "SKIPPED", "reason": "NO_NEW_ASSESSED_SAMPLES", "fits": 0}
    new = t.RUN_ROOT / ("direction-1d-weekly-" + str(uuid4()))
    t.freeze(original, new, weekly_cohort=frozen["cohort_id"])
    t.build(new)
    t.train(new)
    t.replay(new, new.with_name(new.name + "-replay"))
    registered = t.register(new)
    return {
        "status": "SUCCEEDED" if registered else "FAILED",
        "models": registered,
        "reason": None if registered else "INSUFFICIENT_MATURE_504_SESSION_WINDOW",
        "run": new.name,
    }


def sync_answer(payload):
    """答案缺失时异步补T到U的公共净值；每基金30分钟一个幂等作业，沿用原数据锁。"""
    from app.services.tushare_fund_sync import TushareFundSyncService

    code = payload["fund_code"]
    with get_engine().connect() as c:
        if not c.execute(text("SELECT pg_try_advisory_lock(721104,:code)"), {"code": int(code)}).scalar():
            raise ValueError("SYNC_IN_PROGRESS")
        try:
            repo.source(c)
            ts_code = c.execute(
                text("SELECT source_fund_code FROM fund_share_class WHERE fund_code=:code"), {"code": code}
            ).scalar_one()
            result = TushareFundSyncService().sync_market_nav_history(
                (ts_code,),
                start_date=date.fromisoformat(payload["base_nav_date"]),
                end_date=date.fromisoformat(payload["target_nav_date"]),
            )
            return {"run_id": str(result.sync_run_id), "created": result.created_count, "updated": result.updated_count}
        finally:
            c.execute(text("SELECT pg_advisory_unlock(721104,:code)"), {"code": int(code)})
