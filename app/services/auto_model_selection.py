"""后台自动周期：既有维护入口驱动，共享研究worker，扣费账本仍由Java执行。"""

import logging
from datetime import UTC, date, datetime, timedelta
from uuid import uuid4

from sqlalchemy import text

from app.db.session import get_engine
from app.repositories.prediction_store import encode, one, rows
from app.services.auto_model_contract import auto_policy, rolling_window, weekly_slot
from app.services.auto_model_store import cycle_state, event, frozen_input, publish_bundle, transition
from app.services.prediction_contract import PredictionFailure, fingerprint, prediction_policy
from app.services.prediction_direction import direction_fields, validate_identity
from app.services.prediction_models import freeze_routes, load_model, route_key
from app.services.prediction_research import _executor, create_research, research_status, run_research
from app.services.prediction_selection import direction_not_worse

logger = logging.getLogger(__name__)
ACTIVE = ("QUEUED", "TRAINING", "EVALUATING", "REPLAYING", "DECIDING", "INTERRUPTED")


def data_version(codes):
    """以来源实际更新时间及行数检测净值/分红修订；成熟答案内容水位独立保存。"""
    with get_engine().connect() as c:
        nav = rows(
            c,
            """SELECT fund_code,count(*) n,min(nav_date) first_date,max(nav_date) last_date,
          max(updated_at) updated FROM nav_daily WHERE fund_code=ANY(:codes) GROUP BY fund_code ORDER BY fund_code""",
            codes=codes,
        )
        dividends = rows(
            c,
            """SELECT fund_code,count(*) n,max(updated_at) updated FROM fund_dividend
          WHERE fund_code=ANY(:codes) GROUP BY fund_code ORDER BY fund_code""",
            codes=codes,
        )
        events = rows(
            c,
            """SELECT fund_code,dividends_verified_at FROM simulation_market_refresh
          WHERE fund_code=ANY(:codes) ORDER BY fund_code""",
            codes=codes,
        )
        outcomes = one(
            c,
            """SELECT count(*) n,max(o.checked_at) updated FROM prediction_outcome o
          JOIN fund_prediction_record p USING(prediction_id) WHERE p.fund_code=ANY(:codes)""",
            codes=codes,
        )
    return {"nav": nav, "dividends": dividends, "events": events, "outcomeWatermark": fingerprint(outcomes)}


def check_auto(codes, source="MAINTENANCE", now=None):
    """检查仅排队，不在HTTP线程训练；同范围活跃周期唯一，重复心跳返回原任务。"""
    codes, now = sorted(set(codes)), now or datetime.now(UTC)
    if not codes:
        return {"status": "WAITING_DATA", "reason": "尚无关注基金", "fundCount": 0}
    policy = auto_policy()
    scope_hash, version = fingerprint(codes), data_version(codes)
    data_hash, policy_hash = fingerprint(version), fingerprint(policy)
    slot = weekly_slot(now, policy)
    with get_engine().begin() as c:
        c.execute(text("SELECT pg_advisory_xact_lock(721109,5)"))
        active = one(
            c,
            """SELECT * FROM prediction_auto_cycle WHERE scope_hash=:scope AND status=ANY(:states)
            ORDER BY created_at DESC LIMIT 1""",
            scope=scope_hash,
            states=list(ACTIVE),
        )
        if (
            active
            and active["protocol_hash"] != policy_hash
            and (active["lease_until"] is None or active["lease_until"] < now)
            and c.execute(text("SELECT pg_try_advisory_xact_lock(721109,4)")).scalar()
        ):
            # 仅淘汰已空闲的旧实现周期，不抢其他进程正在执行的租约。
            c.execute(
                text("""UPDATE prediction_auto_cycle SET status='FAILED',
              error='{"code":"AUTO_IMPLEMENTATION_CHANGED","summary":"实现版本改变，按新协议重新评价"}'::jsonb,
              updated_at=clock_timestamp(),finished_at=clock_timestamp() WHERE cycle_id=:id"""),
                {"id": active["cycle_id"]},
            )
            event(c, active["cycle_id"], "FAILED", {"code": "AUTO_IMPLEMENTATION_CHANGED"})
            active = None
        if active:
            recover_auto()
            return summary(active)
        latest = one(
            c,
            "SELECT * FROM prediction_auto_cycle WHERE scope_hash=:scope ORDER BY created_at DESC LIMIT 1",
            scope=scope_hash,
        )
        same_protocol = latest and latest["protocol_hash"] == policy_hash
        same_week = same_protocol and latest["spec"].get("weeklySlot") == slot
        new_answers = latest and latest["spec"].get("outcomeWatermark") != version["outcomeWatermark"]
        retry_data = latest and latest["status"] in {"WAITING_DATA", "FAILED"} and latest["data_hash"] != data_hash
        if same_week and not new_answers and not retry_data:
            return summary(latest) | {"check": "NO_NEW_EVALUATION_OR_WEEK"}
        trigger = "MATURED_OR_REVISED" if same_week and new_answers else "INITIAL" if not latest else "WEEKLY"
        window = rolling_window(now, policy)
        spec = {
            "fundCodes": codes,
            "shards": [codes[i : i + policy["shardSize"]] for i in range(0, len(codes), policy["shardSize"])],
            "horizonIds": [h["horizon_id"] for h in policy["predictionPolicy"]["horizons"]],
            **window,
            "stride": policy["stride"],
            "evidenceLevel": "DEVELOPMENT_ONLY",
            "policy": policy,
            "routes": freeze_routes(),
            "weeklySlot": slot,
            "outcomeWatermark": version["outcomeWatermark"],
            "dataVersion": version,
            "source": source,
            "maximumWorkerSeconds": policy["maximumWorkerSeconds"],
        }
        if (
            trigger == "MATURED_OR_REVISED"
            and same_protocol
            and set(latest["spec"]["routes"]) == set(spec["routes"])
            and latest["checkpoint"].get("runId")
            and all(
                (old["model_id"], old["revision"]) == (spec["routes"][key]["model_id"], spec["routes"][key]["revision"])
                for key, old in latest["spec"]["routes"].items()
            )
        ):
            spec["reuseRunId"] = latest["checkpoint"]["runId"]
        identity = {
            "scope": scope_hash,
            "data": data_hash,
            "protocol": policy_hash,
            "slot": slot,
            "window": window,
            "routes": spec["routes"],
            "trigger": trigger,
        }
        key = fingerprint(identity)
        previous = one(c, "SELECT * FROM prediction_auto_cycle WHERE request_key=:key", key=key)
        if previous:
            return summary(previous)
        cycle_id = uuid4()
        ready = any(r["n"] >= 61 for r in version["nav"])
        status = "QUEUED" if ready else "WAITING_DATA"
        c.execute(
            text("""INSERT INTO prediction_auto_cycle
              (cycle_id,request_key,scope_hash,data_hash,protocol_hash,status,trigger_reason,spec,error)
          VALUES(:id,:key,:scope,:data,:policy,:status,:trigger,CAST(:spec AS jsonb),CAST(:error AS jsonb))"""),
            {
                "id": cycle_id,
                "key": key,
                "scope": scope_hash,
                "data": data_hash,
                "policy": policy_hash,
                "status": status,
                "trigger": trigger,
                "spec": encode(spec),
                "error": None
                if ready
                else encode({"code": "AUTO_DATA_INSUFFICIENT", "summary": "必要历史净值不足，现有预测继续运行"}),
            },
        )
        event(c, cycle_id, status, {"source": source, "fundCount": len(codes), "trigger": trigger})
    if ready:
        _executor.submit(run_auto, cycle_id)
    return summary(cycle_state(cycle_id))


def summary(value):
    return {
        "cycleId": str(value["cycle_id"]),
        "status": value["status"],
        "trigger": value["trigger_reason"],
        "fundCount": len(value["spec"]["fundCodes"]),
        "createdAt": value["created_at"].isoformat(),
        "updatedAt": value["updated_at"].isoformat(),
        "finishedAt": str(value["finished_at"]) if value["finished_at"] else None,
        "error": value["error"],
        "decision": (value["result"] or {}).get("decision"),
        "retries": value["retries"],
        "nextAttemptAt": str(value["next_attempt_at"]) if value["next_attempt_at"] else None,
    }


def recover_auto():
    with get_engine().connect() as c:
        pending = rows(
            c,
            """SELECT cycle_id FROM prediction_auto_cycle WHERE
          status IN ('QUEUED','TRAINING','EVALUATING','INTERRUPTED')
          AND (lease_until IS NULL OR lease_until<clock_timestamp())
          AND (next_attempt_at IS NULL OR next_attempt_at<=clock_timestamp()) ORDER BY created_at LIMIT 1""",
        )
    for item in pending:
        _executor.submit(run_auto, item["cycle_id"])
    return len(pending)


def run_auto(cycle_id):
    """会话锁跨线程/进程共享；恢复固定输入和已完成周期，不抢正在工作的旧进程。"""
    with get_engine().connect() as lock:
        if not lock.execute(text("SELECT pg_try_advisory_lock(721109,4)")).scalar():
            return
        owner = uuid4()
        try:
            with get_engine().begin() as c:
                claimed = c.execute(
                    text("""UPDATE prediction_auto_cycle SET lease_owner=:owner,
                  lease_until=clock_timestamp()+make_interval(secs=>(spec->'policy'->>'leaseSeconds')::integer),
                  heartbeat_at=clock_timestamp()
                  WHERE cycle_id=:id AND status IN ('QUEUED','TRAINING','EVALUATING','INTERRUPTED')
                  AND (lease_until IS NULL OR lease_until<clock_timestamp()) RETURNING cycle_id"""),
                    {"id": cycle_id, "owner": owner},
                ).scalar()
            if not claimed:
                return
            state = cycle_state(cycle_id)
            spec, point = state["spec"], state["checkpoint"]
            if state["protocol_hash"] != fingerprint(auto_policy()):
                raise PredictionFailure(
                    "AUTO_IMPLEMENTATION_CHANGED", "RECOVERY", "实现或方向规则改变，原周期不可混版恢复", retryable=False
                )
            if state["cancel_requested"]:
                transition(cycle_id, owner, "CANCELLED")
                return
            if not point.get("runId"):
                # 排队后来源变动不能静默替换冻结版本；交给下一数据版本重新排队。
                if not spec.get("reuseRunId") and fingerprint(data_version(spec["fundCodes"])) != state["data_hash"]:
                    raise PredictionFailure(
                        "AUTO_INPUT_VERSION_CHANGED",
                        "FREEZE",
                        "排队后数据已变动，等待按新版本重建周期",
                        retryable=False,
                    )
                run = (
                    research_status(spec["reuseRunId"])
                    if spec.get("reuseRunId")
                    else create_research(spec | {"autoCycleId": str(cycle_id)}, submit=False)
                )
                point["runId"] = str(run["run_id"])
            transition(cycle_id, owner, "TRAINING", checkpoint=point)
            run = research_status(point["runId"])
            if run["status"] != "SUCCEEDED":
                if run["status"] in {"FAILED", "CANCELLED", "INTERRUPTED"}:
                    with get_engine().begin() as c:
                        c.execute(
                            text("""UPDATE prediction_research_run SET status='QUEUED',lease_until=NULL,
                          cancel_requested=false WHERE run_id=:id"""),
                            {"id": run["run_id"]},
                        )
                run_research(run["run_id"], shared_lock=True)
                run = research_status(run["run_id"])
            if run["status"] != "SUCCEEDED":
                error = (run["result"] or {}).get("error", {"code": "RESEARCH_INCOMPLETE", "summary": "研究未完成"})
                if run["status"] == "CANCELLED":
                    transition(cycle_id, owner, "CANCELLED", checkpoint=point)
                    return
                if error.get("summary") in {"OperationalError", "TimeoutError", "ConnectError"}:
                    retry_transient(cycle_id, owner, point, error)
                    return
                transition(
                    cycle_id,
                    owner,
                    "WAITING_DATA"
                    if error["code"] in {"TRAINING_SAMPLE_INSUFFICIENT", "MODEL_NOT_AVAILABLE_AT_TIME"}
                    else "FAILED",
                    error=error,
                    checkpoint=point,
                )
                return
            transition(cycle_id, owner, "EVALUATING", checkpoint=point)
            bundles = build_bundles(spec, run["result"])
            live = mature_evaluation(spec["fundCodes"], bundles)
            for bundle in bundles[1:]:
                evidence = bundle["predictionEvidence"]
                mature = live["comparisons"].get(evidence["modelId"])
                metric, baseline = evidence["metrics"], bundle["currentPredictionMetrics"]
                if mature and mature["comparable"]:
                    bundle["predictionEligible"] = direction_not_worse(
                        mature["candidate"], mature["current"], spec["policy"]["tolerance"]
                    )
                    bundle["predictionEvidenceLevel"] = "LIVE_MATURED"
                else:
                    bundle["predictionEligible"] = direction_not_worse(metric, baseline, spec["policy"]["tolerance"])
                    bundle["predictionEvidenceLevel"] = "DEVELOPMENT_ONLY"
            point.update(
                bundles=bundles,
                matureEvaluation=live,
                replayStart=str(date.fromisoformat(spec["validationEnd"]) + timedelta(days=1)),
                replayEnd=spec["selectionEnd"],
            )
            transition(cycle_id, owner, "REPLAYING", checkpoint=point)
            # Java研究worker领取后才续租；不占用日常预测worker等待跨服务回放。
            with get_engine().begin() as c:
                c.execute(
                    text("UPDATE prediction_auto_cycle SET lease_until=NULL WHERE cycle_id=:id AND lease_owner=:owner"),
                    {"id": cycle_id, "owner": owner},
                )
            notify_replay_ready(cycle_id)
        except Exception as error:
            logger.exception("auto_model_selection.run_auto >>> cycleId=%s failed", cycle_id)
            detail = (
                error.payload
                if isinstance(error, PredictionFailure)
                else {"code": "AUTO_RESEARCH_ERROR", "summary": type(error).__name__}
            )
            if type(error).__name__ in {"OperationalError", "TimeoutError", "ConnectError"}:
                retry_transient(cycle_id, owner, cycle_state(cycle_id)["checkpoint"], detail)
            else:
                transition(cycle_id, owner, "FAILED", error=detail)
        finally:
            lock.execute(text("SELECT pg_advisory_unlock(721109,4)"))


def notify_replay_ready(cycle_id):
    """完成检查点后通知唯一Java协调器；通知失败由原维护日程补偿，不创建第二计时器。"""
    import httpx

    from app.core.config import get_settings

    settings = get_settings()
    try:
        response = httpx.post(
            settings.core_service_base_url.rstrip("/") + "/internal/v1/multi-predictions/sync/replay-ready",
            headers={"X-Service-Token": settings.ai_service_token.get_secret_value()},
            json={"cycleId": str(cycle_id)},
            timeout=5,
        )
        response.raise_for_status()
    except Exception:
        logger.warning("auto_model_selection.notify_replay_ready >>> cycleId=%s waiting maintenance", cycle_id)
        with get_engine().begin() as c:
            event(c, cycle_id, "WAITING_MAINTENANCE", {"reason": "回放通知未送达，等待既有维护补偿"})


def retry_transient(cycle_id, owner, checkpoint, error):
    """只有连接/数据库瞬态故障有限退避，包/契约/数据错误等待新版本。"""
    state = cycle_state(cycle_id)
    schedule = state["spec"]["policy"]["retryMinutes"]
    if state["retries"] >= len(schedule):
        transition(cycle_id, owner, "FAILED", checkpoint=checkpoint, error=error)
        return
    transition(cycle_id, owner, "INTERRUPTED", checkpoint=checkpoint, error=error)
    with get_engine().begin() as c:
        c.execute(
            text("""UPDATE prediction_auto_cycle SET retries=retries+1,lease_until=NULL,
          next_attempt_at=clock_timestamp()+make_interval(mins=>:minutes)
          WHERE cycle_id=:id AND lease_owner=:owner"""),
            {"id": cycle_id, "owner": owner, "minutes": schedule[state["retries"]]},
        )


def build_bundles(spec, result):
    """当前组合与按固定周期顺序单项替换的候选，最多三组；不按回放收益穷举。"""
    current, candidates = [], []
    for horizon in spec["horizonIds"]:
        route = spec["routes"][route_key(horizon)]
        model = load_model(route["model_id"])
        validate_identity(model["manifest"], horizon)
        current.append(
            {
                "modelId": model["modelId"],
                "modelHash": model["modelHash"],
                "horizonId": horizon,
                "activationRevision": route["revision"],
            }
        )
    bundles = [
        {"id": "MODEL_" + fingerprint(current)[:16], "role": "CURRENT", "label": "当前完整组合", "modelRefs": current}
    ]
    for horizon in spec["horizonIds"]:
        evaluated = result["horizons"][horizon]["models"]
        base = evaluated[0]["metrics"]
        for entry in sorted(evaluated[1:], key=lambda item: item["modelId"]):
            if not entry.get("runnable") or not entry.get("timeValid"):
                continue
            model = load_model(entry["modelId"])
            validate_identity(model["manifest"], horizon)
            refs = [
                ref | {"modelId": model["modelId"], "modelHash": model["modelHash"]}
                if ref["horizonId"] == horizon
                else dict(ref)
                for ref in current
            ]
            candidates.append(
                {
                    "id": "MODEL_" + fingerprint(refs)[:16],
                    "role": "CANDIDATE",
                    "label": "候选完整组合",
                    "modelRefs": refs,
                    "predictionEvidence": entry,
                    "currentPredictionMetrics": base,
                    "changedHorizon": horizon,
                }
            )
    return bundles + candidates[: spec["policy"]["maximumBundles"] - 1]


def mature_evaluation(codes, bundles):
    """只比较已真实发出的同题候选；最新修订替代统计答案，历史开发分数不混入。"""
    from app.services.prediction_selection import evaluate_answers

    with get_engine().connect() as c:
        records = rows(
            c,
            """SELECT p.payload,o.payload answer FROM fund_prediction_record p
          JOIN LATERAL (SELECT payload FROM prediction_outcome WHERE prediction_id=p.prediction_id
            ORDER BY checked_at DESC,outcome_id DESC LIMIT 1) o ON true
          WHERE p.mode='LIVE' AND p.fund_code=ANY(:codes) ORDER BY p.generated_at,p.prediction_id""",
            codes=codes,
        )
    planned, answers = {}, {}
    for row in records:
        p, outcome = row["payload"], row["answer"]
        # 必须在形成题池分母之前隔离；旧规则题目不能降低新三分类模型覆盖率。
        if p.get("targetDefinitionId") != prediction_policy()["target_definition_id"] or p.get(
            "directionPolicyHash", ""
        ) != direction_fields(p["horizonId"]).get("directionPolicyHash", ""):
            continue
        # 原始输入指纹不同的真实预测不混作同一道题；不给缺候选的题补造答案。
        key = f"{p['fundCode']}:{p['horizonId']}:{p['startDate']}:{p.get('featureHash')}:{p['targetDefinitionId']}"
        planned.setdefault(p["horizonId"], {})[key] = {
            "key": key,
            "fund": p["fundCode"],
            "family": p.get("featureSnapshot", {}).get("family", p["fundCode"]),
            "date": p["startDate"],
            "actual": outcome["actualDirection"],
        }
        answers.setdefault(p["modelId"], {})[key] = {"direction": p["direction"]}
    comparisons = {}
    current_refs = {r["horizonId"]: r["modelId"] for r in bundles[0]["modelRefs"]}
    for bundle in bundles[1:]:
        horizon = bundle["changedHorizon"]
        model = bundle["predictionEvidence"]["modelId"]
        exam = list(planned.get(horizon, {}).values())
        current = evaluate_answers(exam, answers.get(current_refs[horizon], {}))
        candidate = evaluate_answers(exam, answers.get(model, {}))
        comparisons[model] = {
            "current": current,
            "candidate": candidate,
            "comparable": bool(exam) and current["coverage"] == 1 and candidate["coverage"] == 1,
        }
    return {
        "evidenceLevel": "LIVE_MATURED",
        "questionHash": fingerprint(planned),
        "plannedItems": sum(len(items) for items in planned.values()),
        "comparisons": comparisons,
        "note": "真实同题覆盖齐全后优先使用；未到期不计错误",
    }


def claim_replay():
    with get_engine().begin() as c:
        item = one(
            c,
            """SELECT * FROM prediction_auto_cycle WHERE status IN ('REPLAYING','DECIDING')
          AND (lease_until IS NULL OR lease_until<clock_timestamp())
          ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1""",
        )
        if not item:
            return None
        if item["protocol_hash"] != fingerprint(auto_policy()):
            c.execute(
                text("""UPDATE prediction_auto_cycle SET status='FAILED',error=CAST(:error AS jsonb),
              updated_at=clock_timestamp(),finished_at=clock_timestamp() WHERE cycle_id=:id"""),
                {
                    "id": item["cycle_id"],
                    "error": encode(
                        {
                            "code": "AUTO_IMPLEMENTATION_CHANGED",
                            "summary": "执行实现与冻结协议不同，保留旧证据并按新协议重新评价",
                        }
                    ),
                },
            )
            event(c, item["cycle_id"], "FAILED", {"code": "AUTO_IMPLEMENTATION_CHANGED"})
            return None
        owner = uuid4()
        c.execute(
            text("""UPDATE prediction_auto_cycle SET lease_owner=:owner,status='REPLAYING',
              lease_until=clock_timestamp()+make_interval(secs=>(spec->'policy'->>'leaseSeconds')::integer)
          WHERE cycle_id=:id"""),
            {"owner": owner, "id": item["cycle_id"]},
        )
        completed = {
            row["input_key"][7:]
            for row in rows(
                c,
                "SELECT input_key FROM prediction_auto_input WHERE cycle_id=:id AND input_key LIKE 'result:%'",
                id=item["cycle_id"],
            )
        }
        pending = [code for code in item["spec"]["fundCodes"] if code not in completed]
        # 最后一条回执已提交、决策尚未提交的崩溃点，只重送一条原回执完成决策。
        if not pending:
            pending = item["spec"]["fundCodes"][:1]
    return {
        "cycleId": str(item["cycle_id"]),
        "leaseOwner": str(owner),
        "fundCodes": pending,
        "executionPolicy": item["spec"]["policy"]["executionPolicy"],
        "maximumWorkerSeconds": item["spec"]["policy"]["maximumWorkerSeconds"],
    }


def replay_input(cycle_id, owner, code):
    from app.services.prediction_replay_inputs import replay_inputs

    state = check_replay_owner(cycle_id, owner)
    if code not in state["spec"]["fundCodes"]:
        raise PredictionFailure("AUTO_SCOPE_MISMATCH", "REPLAY", "基金不属于冻结范围", retryable=False)
    point = state["checkpoint"]
    return frozen_input(
        cycle_id,
        "replay:" + code,
        lambda: replay_inputs(
            code,
            date.fromisoformat(point["replayStart"]),
            date.fromisoformat(point["replayEnd"]),
            frozen_bundles=point["bundles"],
            auto_cycle_id=cycle_id,
        ),
    )


def check_replay_owner(cycle_id, owner):
    with get_engine().begin() as c:
        value = one(
            c,
            """UPDATE prediction_auto_cycle SET heartbeat_at=clock_timestamp(),
              lease_until=clock_timestamp()+make_interval(secs=>(spec->'policy'->>'leaseSeconds')::integer)
          WHERE cycle_id=:id AND lease_owner=:owner AND status='REPLAYING' AND NOT cancel_requested RETURNING *""",
            id=cycle_id,
            owner=owner,
        )
    if not value:
        raise PredictionFailure("AUTO_LEASE_LOST", "REPLAY", "回放租约不属于本任务或已取消", retryable=False)
    if value["protocol_hash"] != fingerprint(auto_policy()):
        raise PredictionFailure(
            "AUTO_PROTOCOL_CHANGED", "REPLAY", "运行协议已改变，不能续用旧周期回放", retryable=False
        )
    return value


def save_replay(cycle_id, owner, code, result):
    state = check_replay_owner(cycle_id, owner)
    if code not in state["spec"]["fundCodes"]:
        raise PredictionFailure("AUTO_SCOPE_MISMATCH", "REPLAY", "基金不属于冻结范围", retryable=False)
    if result.get("status") == "SUCCEEDED":
        with get_engine().connect() as c:
            source = one(
                c,
                "SELECT payload FROM prediction_auto_input WHERE cycle_id=:id AND input_key=:key",
                id=cycle_id,
                key="replay:" + code,
            )
        if not source or result.get("inputHash") != source["payload"].get("inputHash"):
            raise PredictionFailure("AUTO_REPLAY_INPUT_MISMATCH", "REPLAY", "回放回执未对应冻结输入", retryable=False)
        engine_hash = result.get("engineHash", "")
        if len(engine_hash) != 64 or any(ch not in "0123456789abcdef" for ch in engine_hash):
            raise PredictionFailure("AUTO_ENGINE_HASH_MISSING", "REPLAY", "账本执行版本缺失", retryable=False)
        frozen_engine = frozen_input(cycle_id, "java-engine", lambda: {"hash": engine_hash})
        if frozen_engine["hash"] != engine_hash:
            raise PredictionFailure("AUTO_ENGINE_CHANGED", "REPLAY", "同一周期不可混用不同账本实现", retryable=False)
    frozen_input(cycle_id, "result:" + code, lambda: result)
    with get_engine().connect() as c:
        saved = rows(
            c,
            "SELECT input_key,payload FROM prediction_auto_input WHERE cycle_id=:id AND input_key LIKE 'result:%'",
            id=cycle_id,
        )
    if len(saved) == len(state["spec"]["fundCodes"]):
        from app.services.auto_model_contract import thaw_value

        transition(cycle_id, owner, "DECIDING")
        result = decide_bundle(state, {r["input_key"][7:]: thaw_value(r["payload"]) for r in saved})
        # 新目标首次建组需要发布完整基线以核验真实采用；这不是候选胜出，也不绕过后续选优门槛。
        initial = (
            len(state["spec"]["routes"]) == 3
            and all(not route.get("release_id") for route in state["spec"]["routes"].values())
            and result["metrics"][result["winner"]]["fundCount"] > 0
        )
        if result["decision"] == "ACTIVATE" or initial:
            if initial and result["decision"] != "ACTIVATE":
                result["releaseKind"] = "INITIALIZE_CURRENT_BASELINES"
            winner = next(b for b in state["checkpoint"]["bundles"] if b["id"] == result["winner"])
            try:
                release = publish_bundle(
                    cycle_id,
                    winner,
                    state["spec"]["routes"],
                    result
                    | {
                        "policyHash": state["protocol_hash"],
                        "executionPolicy": state["spec"]["policy"]["executionPolicy"],
                        "predictionPolicy": state["spec"]["policy"]["predictionPolicy"],
                    },
                )
                result["releaseId"] = str(release["release_id"])
                transition(cycle_id, owner, "VERIFYING_ADOPTION", result=result)
            except PredictionFailure as error:
                transition(cycle_id, owner, "FAILED", result=result, error=error.payload)
        else:
            transition(cycle_id, owner, "COMPLETED", result=result)
    return summary(cycle_state(cycle_id))


def decide_bundle(state, results):
    """基金家族等权：每基金同本金，缺失或失败仍保留分母且不得靠挑基金胜出。"""
    spec, bundles = state["spec"], state["checkpoint"]["bundles"]
    policy, metrics, failures = spec["policy"], {}, {}
    for bundle in bundles:
        values = []
        for code in spec["fundCodes"]:
            report = results[code]
            value = report.get("comparisons", {}).get(bundle["id"])
            if report.get("status") == "SUCCEEDED" and value:
                values.append((report.get("family", code), value))
            else:
                failures.setdefault(bundle["id"], []).append(
                    {"fundCode": code, "error": report.get("error", "组合不完整或数据不足")}
                )
        families = {}
        for family, value in values:
            families.setdefault(family, []).append(value)

        def mean(field, families=families):
            return (
                sum(sum(float(v[field]) for v in items) / len(items) for items in families.values()) / len(families)
                if families
                else None
            )

        metrics[bundle["id"]] = {
            "fundCount": len(values),
            "plannedFunds": len(spec["fundCodes"]),
            "netReturn": mean("netReturn"),
            "maxDrawdown": mean("maxDrawdown"),
            "fees": mean("fees"),
            "turnover": mean("turnover"),
            "cashOnlySessions": mean("cashOnlySessions"),
        }
    current = bundles[0]["id"]
    winner, reasons = current, []
    for bundle in bundles[1:]:
        if not bundle.get("predictionEligible", True):
            reasons.append(
                {"bundle": bundle["id"], "decision": "KEEP_SHADOW", "reason": "同题预测主指标不满足不劣条件"}
            )
            continue
        key, score, base = bundle["id"], metrics[bundle["id"]], metrics[current]
        # 所有基金的失败集合必须一致；任何缺失不能通过降低分母增加候选胜率。
        same_funds = {r["fundCode"] for r in failures.get(key, [])} == {
            r["fundCode"] for r in failures.get(current, [])
        }
        if not same_funds or score["netReturn"] is None or base["netReturn"] is None:
            reasons.append({"bundle": key, "decision": "BLOCKED_TECHNICAL", "reason": "无共同可比账本或覆盖变化"})
            continue
        better = score["netReturn"] > metrics[winner]["netReturn"] + policy["tolerance"]
        safe = score["maxDrawdown"] <= base["maxDrawdown"] + policy["maxDrawdownDeterioration"]
        if better and safe:
            winner = key
        reasons.append(
            {
                "bundle": key,
                "decision": "ELIGIBLE" if safe else "KEEP_SHADOW",
                "reason": "收益严格改善且回撤未超过协议上限" if better and safe else "无严格改善或回撤恶化",
            }
        )
    return {
        "decision": "ACTIVATE" if winner != current else "KEEP_CURRENT",
        "winner": winner,
        "metrics": metrics,
        "failures": failures,
        "reasons": reasons,
        "inputHash": fingerprint(results),
        "evidenceLevel": "DEVELOPMENT_ONLY",
        "matureEvaluation": state["checkpoint"].get("matureEvaluation"),
        "benchmark": {code: result.get("comparisons", {}).get("BUY_HOLD") for code, result in results.items()},
    }
