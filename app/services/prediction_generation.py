"""公共多周期生成作业：按期次幂等、批次冻结版本、逐项留失败与恢复检查点。"""

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import text

from app.db.session import get_engine
from app.repositories.prediction_store import encode, one, rows, save_prediction
from app.services.prediction_contract import (
    PredictionFailure,
    fingerprint,
    legacy_policy,
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


def create_task(codes, *, request_key=None, horizons=None, retry_items=None):
    """浏览器不能指定过去时间或模型；生成时刻由数据库产生。"""
    codes = sorted(set(codes))
    allowed = {h["horizon_id"] for h in prediction_policy()["horizons"]}
    selected = horizons or sorted(allowed)
    if len(codes) > 500 or not set(selected) <= allowed:
        raise PredictionFailure(
            "BATCH_INPUT_INVALID", "VALIDATION", "批量最多500只基金且只能使用已开放周期", retryable=False
        )
    request_key = fingerprint(prediction_policy()) + ":" + (request_key or str(uuid4()))
    routes = freeze_routes()
    with get_engine().begin() as c:
        c.execute(text("SELECT pg_advisory_xact_lock(hashtext(:key))"), {"key": request_key})
        existing = one(c, "SELECT task_id FROM prediction_generation_task WHERE request_key=:key", key=request_key)
        if existing:
            return task_status(existing["task_id"])
        now = c.execute(text("SELECT clock_timestamp()")).scalar()
        task_id = uuid4()
        payload = {
            "fundCodes": codes,
            "horizonIds": selected,
            "routes": routes,
            "generatedAt": now.isoformat(),
            "policyVersion": prediction_policy()["version"],
            "policyHash": fingerprint(prediction_policy()),
            "predictionPolicy": prediction_policy(),
        }
        c.execute(
            text("""INSERT INTO prediction_generation_task(task_id,request_key,mode,status,payload)
          VALUES(:id,:key,'LIVE','QUEUED',CAST(:payload AS jsonb))"""),
            {"id": task_id, "key": request_key, "payload": encode(payload)},
        )
        for code in codes:
            for horizon in selected:
                if retry_items is not None and (code, horizon) not in retry_items:
                    continue
                c.execute(
                    text("""INSERT INTO prediction_generation_item(task_id,fund_code,horizon_id)
                  VALUES(:id,:code,:horizon)"""),
                    {"id": task_id, "code": code, "horizon": horizon},
                )
    _executor.submit(run_task, task_id)
    return task_status(task_id)


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
    with get_engine().connect() as c:
        task = one(c, "SELECT payload FROM prediction_generation_task WHERE task_id=:id", id=task_id)
    if not task:
        return
    frozen = task["payload"].get("predictionPolicy") or legacy_policy()
    with policy_scope(frozen):
        return _run_task(task_id)


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
            features = build_features(data, now, horizon["lookback_returns"])
            route = payload["routes"].get(route_key(horizon_id, data["fund"]["fund_type"])) or payload["routes"].get(
                route_key(horizon_id)
            )
            if not route:
                raise PredictionFailure("MODEL_PACKAGE_MISSING", "ROUTING", "本周期尚未登记可运行模型")
            result = prediction_payload(data, features, horizon, now, route, task_id=task_id)
            period_key = prediction_period_key(result)
            with get_engine().begin() as c:
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
    with get_engine().begin() as c:
        pending = rows(
            c,
            """UPDATE prediction_generation_task SET status='INTERRUPTED',lease_until=NULL
          WHERE status IN ('QUEUED','RUNNING','INTERRUPTED') AND (lease_until IS NULL OR lease_until<clock_timestamp())
          RETURNING task_id""",
        )
        for task in pending:
            c.execute(
                text("""INSERT INTO prediction_task_event(event_id,task_id,kind,payload)
                              VALUES(:id,:task,'RECOVERED','{"reason":"进程中断或租约超时，复用原时间和版本恢复检查点"}')"""),
                {"id": uuid4(), "task": task["task_id"]},
            )
    for task in pending:
        _executor.submit(run_task, task["task_id"])
    return len(pending)


def retry_failed(task_id):
    previous = task_status(task_id)
    failed = {(i["fundCode"], i["horizonId"]) for i in previous["items"] if i["status"] == "FAILED"}
    with get_engine().connect() as c:
        original = one(c, "SELECT payload FROM prediction_generation_task WHERE task_id=:id", id=task_id)
    # 重试沿用原任务口径；新版本规则由正常的新建生成任务使用。
    with policy_scope(original["payload"].get("predictionPolicy") or legacy_policy()):
        return create_task([c for c, _ in failed], retry_items=failed)


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
                public_calendar(item),
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
            from datetime import date

            start_day = date.fromisoformat(prediction["startDate"])
            data = read_fund_data(prediction["fundCode"], now, start=start_day)
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
