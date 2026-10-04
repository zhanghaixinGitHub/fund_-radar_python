"""公共多周期生成作业：按期次幂等、批次冻结版本、逐项留失败与恢复检查点。"""

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, time
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import text

from app.db.session import get_engine
from app.repositories.prediction_store import encode, one, rows, save_prediction
from app.services.prediction_contract import (
    PredictionFailure,
    fingerprint,
    legacy_policy,
    nav_anchored,
    policy_for_record,
    policy_scope,
    prediction_policy,
    target_dates,
)
from app.services.prediction_direction import LABELS, classify, direction_fields, validate_identity
from app.services.prediction_features import build_features, cash_events, read_fund_data
from app.services.prediction_models import freeze_routes, infer_route, route_key

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-prediction")


def _request_window(instant):
    """保守地按上海自然日和15点分界判定活动任务可复用性。

    节假日可能对应同一估值期，仍允许结束后按真实期次键复用；活动任务不跨此边界
    复用，避免未知旧日历、旧策略或重试子集被误算为本次完成。
    """
    local = datetime.fromisoformat(instant) if isinstance(instant, str) else instant
    local = local.astimezone(ZoneInfo("Asia/Shanghai"))
    return [str(local.date()), "BEFORE_CLOSE" if local.time() < time(15) else "AFTER_CLOSE"]


def _creation_time(connection):
    """生成时间只取数据库；单独封装便于隔离库覆盖跨日/收盘边界。"""
    return connection.execute(text("SELECT clock_timestamp()")).scalar()


def create_task(codes, *, request_key=None, horizons=None, retry_items=None):
    """相同期次、策略和项目集合才能复用；冲突旧任务占用时不另起执行者。"""
    codes = sorted(set(codes))
    policy = prediction_policy()
    allowed = {h["horizon_id"] for h in policy["horizons"]}
    selected = sorted(set(horizons or allowed))
    if len(codes) > 500 or not set(selected) <= allowed:
        raise PredictionFailure(
            "BATCH_INPUT_INVALID", "VALIDATION", "批量最多500只基金且只能使用已开放周期", retryable=False
        )
    wanted = {
        (code, horizon)
        for code in codes
        for horizon in selected
        if retry_items is None or (code, horizon) in retry_items
    }
    policy_hash = fingerprint(policy)
    explicit_request = request_key is not None
    request_key = policy_hash + ":" + (request_key or str(uuid4()))
    # 短事务串行检查创建动作；真实执行另由任务会话锁保护，无关基金批次可正常排队。
    with get_engine().begin() as c:
        c.execute(text("SELECT pg_advisory_xact_lock(hashtext('prediction-create'))"))
        now = _creation_time(c)
        candidates = rows(
            c,
            """SELECT DISTINCT t.* FROM prediction_generation_task t
            JOIN prediction_generation_item i USING(task_id)
            WHERE t.mode='LIVE' AND t.status IN ('QUEUED','RUNNING','INTERRUPTED')
            AND i.fund_code=ANY(:codes) AND i.horizon_id=ANY(:horizons)
            ORDER BY t.created_at""",
            codes=codes,
            horizons=selected,
        )
        occupied = set()
        for active in candidates:
            active_items = rows(
                c, "SELECT fund_code,horizon_id FROM prediction_generation_item WHERE task_id=:id", id=active["task_id"]
            )
            actual = {(i["fund_code"], i["horizon_id"]) for i in active_items}
            if not wanted.intersection(actual):
                continue
            # Java 的私人任务归属表是一任务一所有者。其他显式请求只能收到忙碌提示，
            # 不能取得另一用户的任务编号/基金集合，也不能抢占原任务归属。
            if explicit_request and active["request_key"] != request_key:
                raise PredictionFailure("TASK_SCOPE_BUSY", "GENERATE", "该基金的预测正在处理中，请稍后重试")
            payload = active["payload"]
            compatible = (
                actual == wanted
                and payload.get("policyHash") == policy_hash
                and payload.get("generatedAt")
                and _request_window(payload["generatedAt"]) == _request_window(now)
            )
            if compatible and len(candidates) == 1:
                return {
                    **task_status(active["task_id"]),
                    "blockedByPreviousTask": False,
                    "requestedWindow": _request_window(now),
                }
            if explicit_request:
                raise PredictionFailure(
                    "REQUEST_KEY_WINDOW_OR_SCOPE_MISMATCH", "VALIDATION", "该请求对应其他日期或范围，请重新发起"
                )
            occupied.update(wanted.intersection(actual))
        blocked_items = [{"fundCode": code, "horizonId": horizon} for code, horizon in sorted(occupied)]
        # 占用是基金+周期的边界；同一批中的无关项目仍创建任务，不暴露旧任务的额外基金。
        wanted -= occupied
        if blocked_items and not wanted:
            return {
                "taskId": None,
                "status": "WAITING_FOR_ACTIVE_TASK",
                "items": [],
                "pendingItems": 0,
                "plannedItems": 0,
                "blockedItems": blocked_items,
                "blockedByPreviousTask": True,
                "requestedWindow": _request_window(now),
            }
        existing = one(c, "SELECT * FROM prediction_generation_task WHERE request_key=:key", key=request_key)
        if existing:
            payload = existing["payload"]
            if not payload.get("generatedAt") or _request_window(payload["generatedAt"]) != _request_window(now):
                raise PredictionFailure(
                    "REQUEST_KEY_WINDOW_MISMATCH", "VALIDATION", "该请求已属于其他预测日期，请重新发起"
                )
            previous = task_status(existing["task_id"])
            actual = {(i["fundCode"], i["horizonId"]) for i in previous["items"]}
            if actual != wanted:
                raise PredictionFailure(
                    "REQUEST_KEY_SCOPE_MISMATCH", "VALIDATION", "该请求对应其他预测范围，请重新发起"
                )
            return previous
        routes = freeze_routes()
        if not routes:
            from app.services.prediction_models import bootstrap_models

            bootstrap_models()
            routes = freeze_routes()
        task_id = uuid4()
        payload = {
            "fundCodes": sorted({code for code, _ in wanted}),
            "horizonIds": sorted({horizon for _, horizon in wanted}),
            "blockedItems": blocked_items,
            "routes": routes,
            "generatedAt": now.isoformat(),
            "policyVersion": policy["version"],
            "policyHash": policy_hash,
            "predictionPolicy": policy,
        }
        c.execute(
            text("""INSERT INTO prediction_generation_task(task_id,request_key,mode,status,payload)
            VALUES(:id,:key,'LIVE','QUEUED',CAST(:payload AS jsonb))"""),
            {"id": task_id, "key": request_key, "payload": encode(payload)},
        )
        for code, horizon in sorted(wanted):
            c.execute(
                text("""INSERT INTO prediction_generation_item(task_id,fund_code,horizon_id)
                VALUES(:id,:code,:horizon)"""),
                {"id": task_id, "code": code, "horizon": horizon},
            )
    _dispatch_task(task_id)
    return task_status(task_id)


def _dispatch_task(task_id):
    """数据库提交后派发失败须留下可重试终态；真实执行会话仍持锁时绝不改写状态。"""
    try:
        _executor.submit(run_task, task_id)
        return True
    except Exception:
        logger.exception("prediction_generation._dispatch_task >>> task=%s 后台派发失败", task_id)
        with get_engine().begin() as c:
            acquired = c.execute(
                text("SELECT pg_try_advisory_xact_lock(hashtext(:key))"), {"key": "prediction-task:" + str(task_id)}
            ).scalar()
            if not acquired:
                return False
            changed = c.execute(
                text("""UPDATE prediction_generation_task
                SET status='FAILED',lease_until=NULL,finished_at=clock_timestamp()
                WHERE task_id=:id AND status IN ('QUEUED','INTERRUPTED') RETURNING task_id"""),
                {"id": task_id},
            ).scalar()
            if changed:
                failure = encode(
                    {"error": {"code": "TASK_DISPATCH_FAILED", "summary": "预测未能开始，请重试", "retryable": True}}
                )
                c.execute(
                    text("""UPDATE prediction_generation_item SET status='FAILED',result=CAST(:r AS jsonb)
                    WHERE task_id=:id AND status IN ('PENDING','RUNNING')"""),
                    {"id": task_id, "r": failure},
                )
        return False


def task_status(task_id):
    with get_engine().connect() as c:
        task = one(c, "SELECT * FROM prediction_generation_task WHERE task_id=:id", id=task_id)
        if not task:
            raise PredictionFailure("TASK_NOT_FOUND", "READ", "未找到预测任务", retryable=False)
        items = rows(
            c, "SELECT * FROM prediction_generation_item WHERE task_id=:id ORDER BY fund_code,horizon_id", id=task_id
        )
    states = [i["status"] for i in items]
    counts = {
        "plannedItems": len(items),
        "fundCount": len({i["fund_code"] for i in items}),
        "createdItems": states.count("CREATED"),
        "reusedItems": states.count("REUSED"),
        "failedItems": states.count("FAILED"),
        "cancelledItems": states.count("CANCELLED"),
        "pendingItems": states.count("PENDING") + states.count("RUNNING"),
        "fallbackItems": sum(bool((i["result"] or {}).get("fallbackReason")) for i in items),
        "baselineItems": sum(bool((i["result"] or {}).get("baseline")) for i in items),
    }
    return {
        "taskId": str(task_id),
        "status": task["status"],
        "blockedItems": task["payload"].get("blockedItems", []),
        "createdAt": str(task["created_at"]),
        "finishedAt": str(task["finished_at"]) if task["finished_at"] else None,
        **counts,
        "items": [
            {
                "fundCode": i["fund_code"],
                "horizonId": i["horizon_id"],
                "status": i["status"],
                "attempts": i["attempts"],
                "result": i["result"],
            }
            for i in items
        ],
    }


def prediction_payload(data, features, horizon, now, route, *, mode="LIVE", task_id=None):
    dates = target_dates(data["calendar"], now, horizon["horizon_id"])
    if nav_anchored() and dates["baseNavDate"] != features["dataAsOf"]:
        raise PredictionFailure("BASE_NAV_MISMATCH", "GENERATE", "预测日期与实际净值基准不一致，停止生成")
    inference = infer_route(route, features["features"], horizon["horizon_id"])
    direction = LABELS[inference["direction"]]
    lookback = features["features"]["actualLookbackReturns"]
    return {
        "predictionId": str(uuid4()),
        "fundCode": data["fund"]["fund_code"],
        **dates,
        **inference,
        "predictionPolicySnapshot": prediction_policy(),
        "mode": mode,
        "generationStatus": "SUCCEEDED",
        "generatedAt": now.isoformat(),
        "knowledgeCutoff": features["knowledgeCutoff"],
        "dataAsOf": features["dataAsOf"],
        "featureSnapshot": features,
        "featureHash": features["featureHash"],
        "taskId": str(task_id),
        "reason": f"依据截至{features['dataAsOf']}的总回报历史（{lookback}段），本周期预计{direction}。"
        + ("使用基础实验方法。" if inference["baseline"] else "使用当前已登记实验模型。"),
        "limitations": features["missingOptionalFactors"] + [features["calendarAssumption"], "实验结果可能错误"],
        "role": "PRIMARY",
    }


def run_task(task_id):
    """会话锁覆盖真实执行期；租约仅用于恢复发现，不允许长计算期间抢跑第二个执行者。"""
    with get_engine().connect() as lock:
        key = "prediction-task:" + str(task_id)
        acquired = lock.execute(text("SELECT pg_try_advisory_lock(hashtext(:key))"), {"key": key}).scalar()
        if not acquired:
            return
        try:
            with get_engine().connect() as c:
                task = one(c, "SELECT payload FROM prediction_generation_task WHERE task_id=:id", id=task_id)
            if not task:
                return
            frozen = task["payload"].get("predictionPolicy") or legacy_policy()
            with policy_scope(frozen):
                return _run_task(task_id)
        except Exception:
            logger.exception("prediction_generation.run_task >>> task=%s 执行中断，保留成功项", task_id)
            # 真实线程退出才释放锁；未提交项明确失败，不能悬挂 RUNNING 或误计成功。
            with get_engine().begin() as c:
                failure = encode({"error": {"code": "TASK_EXECUTION_INTERRUPTED", "summary": "预测执行中断，请重试"}})
                c.execute(
                    text("""UPDATE prediction_generation_item SET status='FAILED',result=CAST(:r AS jsonb)
                    WHERE task_id=:id AND status IN ('PENDING','RUNNING')"""),
                    {"id": task_id, "r": failure},
                )
                c.execute(
                    text("""UPDATE prediction_generation_task SET status='FAILED',lease_until=NULL,
                    finished_at=clock_timestamp() WHERE task_id=:id AND status<>'CANCELLED'"""),
                    {"id": task_id},
                )
        finally:
            lock.execute(text("SELECT pg_advisory_unlock(hashtext(:key))"), {"key": key})


def prediction_period_key(result):
    """相同基金/观察起点/目标/规则只有首份主预测；采用新模型也不覆盖本期原文。"""
    base = f"LIVE:{result['fundCode']}:{result['horizonId']}:{result['startDate']}"
    if result.get("directionPolicyHash"):
        return base + f":TARGET:{result['targetDefinitionId']}:RULE:{result['directionPolicyHash']}"
    return base


def _run_task(task_id):
    """持久检查点按项提交；进程中断后只恢复未提交项，租约避免多服务同时运行。"""
    with get_engine().begin() as c:
        task = one(c, "SELECT * FROM prediction_generation_task WHERE task_id=:id FOR UPDATE", id=task_id)
        if not task or task["status"] in {"SUCCEEDED", "PARTIAL_SUCCESS", "FAILED", "CANCELLED"}:
            return
        if task["lease_until"] and task["lease_until"] > datetime.now(UTC):
            return
        c.execute(
            text("""UPDATE prediction_generation_task SET status='RUNNING',
          lease_until=clock_timestamp()+interval '5 minutes' WHERE task_id=:id"""),
            {"id": task_id},
        )
        items = rows(
            c,
            """SELECT * FROM prediction_generation_item WHERE task_id=:id
          AND status IN ('PENDING','RUNNING') ORDER BY fund_code,horizon_id""",
            id=task_id,
        )
    payload = task["payload"]
    now = datetime.fromisoformat(payload["generatedAt"])
    policy = {h["horizon_id"]: h for h in prediction_policy()["horizons"]}
    cache = {}
    for item in items:
        code, horizon_id = item["fund_code"], item["horizon_id"]
        with get_engine().begin() as c:
            if (
                one(c, "SELECT status FROM prediction_generation_task WHERE task_id=:id", id=task_id)["status"]
                == "CANCELLED"
            ):
                break
            c.execute(
                text("""UPDATE prediction_generation_task SET lease_until=clock_timestamp()+interval '5 minutes'
              WHERE task_id=:id"""),
                {"id": task_id},
            )
        try:
            if code not in cache:
                from app.services.prediction_task_inputs import task_input

                cache[code] = task_input("LIVE", task_id, code, lambda code=code: read_fund_data(code, now))
            data = cache[code]
            if "frozenFailure" in data:
                failure = data["frozenFailure"]
                raise PredictionFailure(
                    failure["code"],
                    failure["stage"],
                    failure["summary"],
                    details=failure.get("details"),
                    retryable=failure.get("retryable", True),
                )
            horizon = policy[horizon_id]
            identity = {
                "fundCode": code,
                **target_dates(data["calendar"], now, horizon_id),
                **direction_fields(horizon_id),
            }
            period_key = prediction_period_key(identity)
            with get_engine().connect() as c:
                existing = one(c, "SELECT payload FROM fund_prediction_record WHERE period_key=:key", key=period_key)
            if existing:
                result, status = existing["payload"], "REUSED"
            else:
                features = build_features(data, now, horizon["lookback_returns"])
                route = payload["routes"].get(route_key(horizon_id, data["fund"]["fund_type"])) or payload[
                    "routes"
                ].get(route_key(horizon_id))
                if not route:
                    raise PredictionFailure("MODEL_PACKAGE_MISSING", "ROUTING", "本周期尚未登记可运行模型")
                result = prediction_payload(data, features, horizon, now, route, task_id=task_id)
                period_key = prediction_period_key(result)
                with get_engine().begin() as c:
                    if nav_anchored():
                        completed = c.execute(text("SELECT clock_timestamp()")).scalar_one()
                        deadline = datetime.combine(
                            date.fromisoformat(result["startDate"]), time(15), ZoneInfo("Asia/Shanghai")
                        )
                        if completed >= deadline:
                            raise PredictionFailure(
                                "WINDOW_CHANGED", "GENERATE", "目标估值日已收盘，请重新检查最新净值后生成"
                            )
                        result["completedAt"] = completed.isoformat()
                    result, created = save_prediction(c, result, period_key)
                status = "CREATED" if created else "REUSED"
                # 候选以自身模型身份独立落档，不混入主预测统计与用户当前卡。
                for shadow in route.get("shadow_ids", [])[:3]:
                    try:
                        shadow_route = route | {
                            "model_id": shadow,
                            "previous_model_id": None,
                            "strictModel": True,
                            "release_id": None,
                        }
                        shadow_result = prediction_payload(data, features, horizon, now, shadow_route, task_id=task_id)
                        shadow_result["role"] = "SHADOW"
                        with get_engine().begin() as c:
                            save_prediction(c, shadow_result, period_key + ":SHADOW:" + shadow)
                    except Exception:
                        logger.exception(
                            "prediction_generation.run_task >>> shadow failed taskId=%s fund=%s horizon=%s",
                            task_id,
                            code,
                            horizon_id,
                        )
        except PredictionFailure as error:
            result = {
                "generationStatus": "FAILED",
                "decision": None,
                "error": error.payload | {"traceId": str(task_id)},
                "generatedAt": now.isoformat(),
            }
            status = "FAILED"
        except Exception:
            logger.exception(
                "prediction_generation.run_task >>> failed taskId=%s fund=%s horizon=%s", task_id, code, horizon_id
            )
            result = {
                "generationStatus": "FAILED",
                "decision": None,
                "generatedAt": now.isoformat(),
                "error": {
                    "code": "INFERENCE_ERROR",
                    "stage": "GENERATE",
                    "summary": "预测计算异常，请凭请求号排查",
                    "traceId": str(task_id),
                    "retryable": True,
                    "nextAction": "重试或查看后台日志",
                },
            }
            status = "FAILED"
        result.update(targetDefinitionId=prediction_policy()["target_definition_id"], **direction_fields(horizon_id))
        with get_engine().begin() as c:
            c.execute(
                text("""INSERT INTO prediction_attempt(attempt_id,task_id,fund_code,horizon_id,payload)
              VALUES(:id,:task,:code,:horizon,CAST(:result AS jsonb))"""),
                {"id": uuid4(), "task": task_id, "code": code, "horizon": horizon_id, "result": encode(result)},
            )
            c.execute(
                text("""UPDATE prediction_generation_item SET status=:status,result=CAST(:result AS jsonb),
              prediction_id=:prediction,attempts=attempts+1
              WHERE task_id=:task AND fund_code=:code AND horizon_id=:horizon"""),
                {
                    "status": status,
                    "result": encode(result),
                    "prediction": result.get("predictionId"),
                    "task": task_id,
                    "code": code,
                    "horizon": horizon_id,
                },
            )
    summary = task_status(task_id)
    status = (
        ("PARTIAL_SUCCESS" if summary["failedItems"] < summary["plannedItems"] else "FAILED")
        if summary["failedItems"]
        else "SUCCEEDED"
    )
    with get_engine().begin() as c:
        c.execute(
            text("""UPDATE prediction_generation_task SET status=:status,finished_at=clock_timestamp(),
          lease_until=NULL,result=CAST(:result AS jsonb) WHERE task_id=:id AND status<>'CANCELLED'"""),
            {"id": task_id, "status": status, "result": encode(summary)},
        )


def recover_tasks():
    """只有租约过期且真实执行会话已经退出的任务才允许恢复。"""
    pending = []
    with get_engine().begin() as c:
        candidates = rows(
            c,
            """SELECT task_id FROM prediction_generation_task
            WHERE status IN ('QUEUED','RUNNING','INTERRUPTED')
            AND (lease_until IS NULL OR lease_until<clock_timestamp()) ORDER BY created_at FOR UPDATE SKIP LOCKED""",
        )
        for task in candidates:
            acquired = c.execute(
                text("SELECT pg_try_advisory_xact_lock(hashtext(:key))"),
                {"key": "prediction-task:" + str(task["task_id"])},
            ).scalar()
            if not acquired:
                continue
            c.execute(
                text("UPDATE prediction_generation_task SET status='INTERRUPTED',lease_until=NULL WHERE task_id=:id"),
                {"id": task["task_id"]},
            )
            c.execute(
                text("""INSERT INTO prediction_task_event(event_id,task_id,kind,payload)
                VALUES(:id,:task,'RECOVERED','{"reason":"实际执行已退出，按原日期恢复未完成项"}')"""),
                {"id": uuid4(), "task": task["task_id"]},
            )
            pending.append(task)
    dispatched = 0
    for task in pending:
        try:
            dispatched += int(_dispatch_task(task["task_id"]))
        except Exception:
            # 数据库本身不可写时也不能中断后续恢复；原租约已释放，下一次维护仍可发现该项。
            logger.exception("prediction_generation.recover_tasks >>> task=%s 恢复收尾失败", task["task_id"])
    return dispatched


def retry_failed(task_id):
    previous = task_status(task_id)
    failed = {(i["fundCode"], i["horizonId"]) for i in previous["items"] if i["status"] == "FAILED"}
    with get_engine().connect() as c:
        original = one(c, "SELECT payload FROM prediction_generation_task WHERE task_id=:id", id=task_id)
    # 重试沿用原任务口径；新版本规则由正常的新建生成任务使用。
    with policy_scope(original["payload"].get("predictionPolicy") or legacy_policy()):
        return create_task([c for c, _ in failed], request_key=str(uuid4()), retry_items=failed)


def current_predictions(fund_code):
    target = prediction_policy()["target_definition_id"]
    rule = direction_fields("T5_V1").get("directionPolicyHash", "")
    with get_engine().connect() as c:
        latest = rows(
            c,
            """SELECT DISTINCT ON(horizon_id) payload FROM fund_prediction_record
          WHERE fund_code=:code AND mode='LIVE' AND payload->>'role'='PRIMARY'
          AND payload->>'targetDefinitionId'=:target AND COALESCE(payload->>'directionPolicyHash','')=:rule
          ORDER BY horizon_id,generated_at DESC""",
            code=fund_code,
            target=target,
            rule=rule,
        )
        attempts = rows(
            c,
            """SELECT DISTINCT ON(horizon_id) horizon_id,payload,created_at
          FROM prediction_attempt WHERE fund_code=:code
          AND payload->>'targetDefinitionId'=:target AND COALESCE(payload->>'directionPolicyHash','')=:rule
          ORDER BY horizon_id,created_at DESC""",
            code=fund_code,
            target=target,
            rule=rule,
        )
        resolutions = rows(
            c,
            """SELECT DISTINCT ON(r.prediction_id) r.prediction_id,r.payload
          FROM prediction_target_resolution r JOIN fund_prediction_record p USING(prediction_id)
          WHERE p.fund_code=:code ORDER BY r.prediction_id,r.created_at DESC LIMIT 100""",
            code=fund_code,
        )
    return {
        "fundCode": fund_code,
        "horizons": prediction_policy()["horizons"],
        "targetDefinitionId": target,
        "directionPolicyHash": rule,
        "predictions": [r["payload"] for r in latest],
        "latestAttempts": attempts,
        "targetResolutions": {str(r["prediction_id"]): r["payload"] for r in resolutions},
    }


def record_check(prediction_id, status, payload):
    """可变检查水位仅用于公平轮转；原预测、目标解析和每版到期答案仍为追加式证据。"""
    with get_engine().begin() as c:
        c.execute(
            text("""INSERT INTO prediction_check_state(prediction_id,status,payload)
          VALUES(:id,:status,CAST(:payload AS jsonb)) ON CONFLICT(prediction_id) DO UPDATE
          SET status=excluded.status,payload=excluded.payload,last_attempt_at=clock_timestamp()"""),
            {"id": prediction_id, "status": status, "payload": encode(payload)},
        )


def resolve_pending_targets(limit=100):
    """官方远期日历公布即追加日期解析，不必等到半年到期才展示具体终点。"""
    from app.services.prediction_features import public_calendar

    with get_engine().connect() as c:
        pending = rows(
            c,
            """SELECT p.prediction_id,p.payload,f.fund_code,f.fund_name,f.fund_type
          FROM fund_prediction_record p JOIN fund_share_class f USING(fund_code)
          LEFT JOIN prediction_check_state s USING(prediction_id)
          WHERE p.payload->>'endDate' IS NULL AND NOT EXISTS
          (SELECT 1 FROM prediction_target_resolution r WHERE r.prediction_id=p.prediction_id)
          ORDER BY s.last_attempt_at ASC NULLS FIRST,p.generated_at LIMIT :limit""",
            limit=min(limit, 500),
        )
    resolved_count = 0
    for item in pending:
        try:
            prediction = item["payload"]
            resolved = target_dates(
                public_calendar(item, legacy=not nav_anchored(policy_for_record(prediction))),
                datetime.fromisoformat(prediction["generatedAt"]),
                prediction["horizonId"],
                policy=policy_for_record(prediction),
            )
            if resolved["startDate"] != prediction["startDate"]:
                raise PredictionFailure("CALENDAR_START_REVISED", "CALENDAR", "新日历改变原起点，保留原目标等待核验")
            if resolved["endDate"]:
                with get_engine().begin() as c:
                    c.execute(
                        text("""INSERT INTO prediction_target_resolution(prediction_id,resolution_hash,payload)
                      VALUES(:id,:hash,CAST(:payload AS jsonb)) ON CONFLICT DO NOTHING"""),
                        {"id": item["prediction_id"], "hash": fingerprint(resolved), "payload": encode(resolved)},
                    )
                resolved_count += 1
            record_check(item["prediction_id"], "RESOLVED" if resolved["endDate"] else "PENDING_CALENDAR", resolved)
        except PredictionFailure as error:
            record_check(item["prediction_id"], "PENDING_CALENDAR", error.payload)
    return {"checked": len(pending), "resolved": resolved_count}


def verify_outcomes(limit=100):
    """到期标签追加版本；尚未有未来官方日历时等待，绝不把名义日期伪装成已解析日期。"""
    from app.services.prediction_contract import reinvested_series

    with get_engine().connect() as c:
        records = rows(
            c,
            """SELECT prediction_id,payload FROM fund_prediction_record
            WHERE mode='LIVE' AND coalesce(payload->>'endDate',payload->>'nominalEndDate')<CAST(CURRENT_DATE AS text)
            ORDER BY (SELECT s.last_attempt_at FROM prediction_check_state s
                      WHERE s.prediction_id=fund_prediction_record.prediction_id)
                      ASC NULLS FIRST,generated_at LIMIT :limit""",
            limit=min(limit, 500),
        )
    now, completed = datetime.now(UTC), 0
    for record in records:
        prediction = record["payload"]
        check_status, check_payload = "PENDING_DATA", {"summary": "到期净值或分红核验水位尚未齐备"}
        try:
            # 新收益从已取得的基准净值计算；旧预测仍使用原startDate，禁止重写旧成绩。
            start_day = date.fromisoformat(prediction.get("baseNavDate", prediction["startDate"]))
            data = read_fund_data(
                prediction["fundCode"],
                now,
                start=start_day,
                legacy_calendar=not nav_anchored(policy_for_record(prediction)),
            )
            end = prediction["endDate"]
            if not end:
                resolved = target_dates(
                    data["calendar"],
                    datetime.fromisoformat(prediction["generatedAt"]),
                    prediction["horizonId"],
                    policy=policy_for_record(prediction),
                )
                if not resolved["endDate"]:
                    continue
                if resolved["startDate"] != prediction["startDate"]:
                    raise PredictionFailure(
                        "CALENDAR_START_REVISED", "OUTCOME", "新日历改变原起点，需要单独审查，不能改写目标"
                    )
                with get_engine().begin() as c:
                    c.execute(
                        text("""INSERT INTO prediction_target_resolution(prediction_id,resolution_hash,payload)
                                      VALUES(:id,:hash,CAST(:payload AS jsonb)) ON CONFLICT DO NOTHING"""),
                        {"id": record["prediction_id"], "hash": fingerprint(resolved), "payload": encode(resolved)},
                    )
                end = resolved["endDate"]
            if end >= str(now.date()):
                continue
            end_day = date.fromisoformat(end)
            dates = [d for d in data["calendar"].sessions if start_day <= d <= end_day]
            if data["dividendWatermark"].date() < end_day:
                continue
            nav = {r["nav_date"]: r["unit_nav"] for r in data["navs"] if r["updated_at"] <= now}
            events = cash_events(data, dates, now)
            series = reinvested_series(dates, nav, events)
            value = series[-1] / series[0] - 1
            saved_policy = policy_for_record(prediction)
            validate_identity(prediction, prediction["horizonId"], saved_policy)
            actual = classify(value, prediction["horizonId"], saved_policy)
            outcome = {
                "endDate": end,
                "totalReturn": str(value),
                "actualDirection": actual,
                "correct": actual == prediction["direction"],
                **direction_fields(prediction["horizonId"], saved_policy),
                "exactZero": value == 0,
                "checkedAt": now.isoformat(),
                "flat": value == 0,
                "targetDefinitionId": prediction["targetDefinitionId"],
                "navHash": fingerprint({str(d): str(nav[d]) for d in dates}),
                "eventHash": fingerprint({str(d): str(v) for d, v in events.items()}),
            }
            # 核验时刻不参与内容身份；净值或分红修订才追加新版本，重复检查不增加独立成绩。
            identity = {k: v for k, v in outcome.items() if k != "checkedAt"}
            with get_engine().begin() as c:
                c.execute(
                    text("""INSERT INTO prediction_outcome(outcome_id,prediction_id,content_hash,payload)
                  VALUES(:id,:prediction,:hash,CAST(:payload AS jsonb)) ON CONFLICT DO NOTHING"""),
                    {
                        "id": uuid4(),
                        "prediction": record["prediction_id"],
                        "hash": fingerprint(identity),
                        "payload": encode(outcome),
                    },
                )
            completed += 1
            check_status, check_payload = "SUCCEEDED", {"checkedAt": now.isoformat(), "endDate": end}
        except PredictionFailure as error:
            check_payload = error.payload
            logger.info(
                "prediction_generation.verify_outcomes >>> outcome pending predictionId=%s code=%s",
                record["prediction_id"],
                error.payload["code"],
            )
        except Exception:
            check_status, check_payload = "FAILED", {"summary": "到期核验服务异常", "code": "OUTCOME_CHECK_FAILED"}
            logger.exception(
                "prediction_generation.verify_outcomes >>> predictionId=%s failed", record["prediction_id"]
            )
        finally:
            record_check(record["prediction_id"], check_status, check_payload)
    return {"checked": len(records), "completed": completed}
