"""正式第二版的新窗口请求；缺净值或缺实际审阅时只登记等待，不猜方向。"""

import copy
import json
from datetime import date, datetime, time
from uuid import UUID, uuid4

from sqlalchemy import text

from app.db.session import get_engine
from app.repositories import direction_1d as repo
from app.services.direction_1d_protocol import ZONE, digest, window
from app.services.fund_exposure_common import now
from app.services.prediction_contract import PredictionFailure
from app.services.prediction_review_v2 import policy_v2
from app.services.prediction_trial_v2 import current_holdings

HORIZONS = ("T1", "T5_V1", "T20_V1", "M6_V1")


def enabled_v2():
    from app.services.prediction_review_v2_publication import adoption_status

    return adoption_status() is not None


def _readiness(codes, instant):
    """请求中的基准只用于安排补查，正式预测仍须核本基金估值条款与资料。"""
    planning = window(instant)
    base = date.fromisoformat(planning["base_nav_date"])
    found = {}
    with get_engine().connect() as connection:
        source = repo.source(connection)
        for code in codes:
            available = repo.navs(connection, code, source["source_id"], base, base)
            found[code] = any(
                row["nav_date"] == base
                and row.get("unit_nav") is not None
                and row["unit_nav"] > 0
                and (row.get("ann_date") is None or row["ann_date"] <= instant.date())
                for row in available
            )
    return planning, found


def request_batch(codes, *, horizons=None, request_key=None, retry_items=None):
    """保存真实待办，不复用旧版本生成器，也不把已发布旧分析当新窗口完成。"""
    if not enabled_v2():
        raise PredictionFailure("V2_NOT_ADOPTED", "GENERATE", "第二版尚未正式启用")
    codes = sorted(set(codes))
    selected = sorted(set(horizons or HORIZONS))
    if not codes or len(codes) > 500 or not set(selected) <= set(HORIZONS):
        raise PredictionFailure("BATCH_INPUT_INVALID", "VALIDATION", "请选择有效基金和预测周期", retryable=False)
    scope = current_holdings()
    if not set(codes) <= set(scope["scope"]):
        raise PredictionFailure("V2_FUND_NOT_HELD", "VALIDATION", "本轮仅更新本人当前持仓基金", retryable=False)
    instant = now()
    planning, ready = _readiness(codes, instant)
    items = [
        {
            "fundCode": code,
            "horizonId": horizon,
            "status": "PENDING",
            "attempts": 0,
            "result": {"waitingFor": "ANALYSIS" if ready[code] else "NAV"},
        }
        for code in codes
        for horizon in selected
        if retry_items is None or (code, horizon) in retry_items
    ]
    if not items:
        raise PredictionFailure("BATCH_INPUT_INVALID", "VALIDATION", "没有需要更新的周期", retryable=False)
    status = "PENDING_ANALYSIS" if all(ready.values()) else "WAITING_FOR_NAV"
    summary = (
        "已登记按新规则更新，等待完成本期资料审阅；页面保留上次分析及日期。"
        if status == "PENDING_ANALYSIS"
        else "已登记更新，等待新基准净值及资料审阅；页面保留上次分析及日期。"
    )
    identity = digest(
        {
            "codes": codes,
            "items": [(i["fundCode"], i["horizonId"]) for i in items],
            "window": planning,
            "policy": digest(policy_v2()),
            "request": str(request_key) if request_key else None,
        }
    )
    payload = {
        "taskId": str(uuid4()),
        "status": status,
        "summary": summary,
        "requestedAt": instant.isoformat(),
        "createdAt": instant.isoformat(),
        "generatedAt": None,
        "analysisPolicyVersion": "FUND_COMPREHENSIVE_REVIEW_V2",
        "analysisPolicyHash": digest(policy_v2()),
        "requiredBaseNavDate": planning["base_nav_date"],
        "planningWindow": planning,
        "fundCount": len(codes),
        "plannedItems": len(items),
        "pendingItems": len(items),
        "createdItems": 0,
        "reusedItems": 0,
        "failedItems": 0,
        "cancelledItems": 0,
        "fallbackItems": 0,
        "baselineItems": 0,
        "items": items,
        "note": "等待新的实际审阅；此请求不含方向，不是已完成的预测。",
        "_ownerHash": scope["owner_hash"],
    }
    with get_engine().begin() as connection:
        connection.execute(
            text("""INSERT INTO prediction_review_v2_request(task_id,idempotency_key,payload)
            VALUES(:id,:key,CAST(:payload AS jsonb)) ON CONFLICT(idempotency_key) DO NOTHING"""),
            {"id": payload["taskId"], "key": identity, "payload": json.dumps(payload, ensure_ascii=False)},
        )
        result = connection.execute(
            text("SELECT payload FROM prediction_review_v2_request WHERE idempotency_key=:key"), {"key": identity}
        ).scalar_one()
        if result["status"] in {"WAITING_FOR_NAV", "PENDING_ANALYSIS"} and result["status"] != status:
            result = {**result, "status": status, "summary": summary}
            connection.execute(
                text(
                    "UPDATE prediction_review_v2_request SET payload=CAST(:payload AS jsonb) WHERE idempotency_key=:key"
                ),
                {"key": identity, "payload": json.dumps(result, ensure_ascii=False)},
            )
    return _public_task(result)


def request_review(fund_code):
    result = request_batch([fund_code])
    return {
        "status": "AVAILABLE"
        if result["status"] == "SUCCEEDED"
        else "WAITING_FOR_NAV"
        if result["status"] == "WAITING_FOR_NAV"
        else "PENDING_ANALYSIS",
        "summary": result["summary"],
        "requestedAt": result["requestedAt"],
    }


def _public_task(payload):
    return {key: value for key, value in payload.items() if not key.startswith("_")}


def find_task(task_id):
    """未迁移时无新任务；数据库故障继续报错，不能伪装成旧版或空结果。"""
    with get_engine().connect() as connection:
        exists = connection.execute(text("SELECT to_regclass('prediction_review_v2_request')")).scalar_one()
        if not exists:
            return None
        result = connection.execute(
            text("SELECT payload FROM prediction_review_v2_request WHERE task_id=:id"), {"id": UUID(str(task_id))}
        ).scalar_one_or_none()
    return _public_task(result) if result is not None else None


def _completed_payload(payload, prepared, release_id, adopted_at):
    """只接纳同账号、同规则、同窗口的严格审阅；旧36项研究没有这类完成凭据。"""
    if payload.get("analysisPolicyHash") != prepared["policyHash"]:
        return None
    result = copy.deepcopy(payload)
    found = False
    planning = payload.get("planningWindow") or {}
    cases = prepared["archive"].get("cases", [])
    for item in result["items"]:
        if item["status"] != "PENDING":
            continue
        for case in cases:
            record, identity = case.get("analysis", {}), case.get("workflow_identity", {})
            source = case.get("case", {})
            if (
                source.get("fund_code") != item["fundCode"]
                or source.get("horizon") != item["horizonId"]
                or not payload.get("_ownerHash")
                or identity.get("owner_hash") != payload["_ownerHash"]
                or identity.get("base_date") != planning.get("base_nav_date")
                or identity.get("target_date") != planning.get("target_nav_date")
                or identity.get("snapshot_verified") is not True
                or not identity.get("materials_input_hash")
            ):
                continue
            evaluated = record.get("evaluated_at")
            if not evaluated or not identity.get("target_date"):
                continue
            deadline = datetime.combine(date.fromisoformat(identity["target_date"]), time(15), ZONE)
            if datetime.fromisoformat(evaluated) >= deadline:
                continue
            decision = source.get("decision")
            conditional = item["horizonId"] == "M6_V1" and not identity.get("end_date")
            if conditional:
                state = "CONDITIONAL_SAVED"
            elif decision in {"ABSTAIN", "BLOCKED"}:
                state = "REVIEWED_ABSTAIN"
            elif identity.get("ex_ante_eligible") is True and decision in {"UP", "DOWN"}:
                state = "CREATED"
            else:
                continue
            item.update(
                status=state,
                result={
                    "releaseId": str(release_id),
                    "reviewedAt": evaluated,
                    "decision": decision,
                    "excludedFromAccuracy": True,
                },
            )
            found = True
            break
    if not found:
        return None
    states = [item["status"] for item in result["items"]]
    pending = states.count("PENDING")
    result.update(
        status="SUCCEEDED" if not pending else "PARTIAL_REVIEW",
        pendingItems=pending,
        createdItems=states.count("CREATED"),
        reviewedItems=len(states) - pending,
        conditionalItems=states.count("CONDITIONAL_SAVED"),
        abstainItems=states.count("REVIEWED_ABSTAIN"),
        finishedAt=adopted_at.isoformat() if not pending else None,
        summary="本期审阅已保存；半年条件意见和暂不判断单列，不代表每个周期都有事前预测。"
        if not pending
        else "部分周期的新审阅已保存，其余周期仍待处理。",
    )
    return result


def complete_from_release(prepared, release_id, adopted_at):
    """正式发布提交后更新本新增队列进度，不写旧预测、订单或准确率。"""
    bases = sorted(
        {
            item.get("workflow_identity", {}).get("base_date")
            for item in prepared["archive"].get("cases", [])
            if item.get("workflow_identity", {}).get("base_date")
        }
    )
    if not bases:
        return 0
    updated = 0
    with get_engine().begin() as connection:
        if connection.scalar(text("SELECT to_regclass('prediction_review_v2_request')")) is None:
            return 0
        rows = connection.execute(
            text("""SELECT task_id,payload FROM prediction_review_v2_request
            WHERE payload->>'analysisPolicyHash'=:policy
              AND payload->'planningWindow'->>'base_nav_date'=ANY(:bases)
              AND payload->>'status' IN ('WAITING_FOR_NAV','PENDING_ANALYSIS','PARTIAL_REVIEW')
            ORDER BY created_at DESC LIMIT 500 FOR UPDATE"""),
            {"policy": prepared["policyHash"], "bases": bases},
        ).mappings()
        for row in rows:
            value = _completed_payload(row["payload"], prepared, release_id, adopted_at)
            if value is not None:
                connection.execute(
                    text("UPDATE prediction_review_v2_request SET payload=CAST(:payload AS jsonb) WHERE task_id=:id"),
                    {"id": row["task_id"], "payload": json.dumps(value, ensure_ascii=False)},
                )
                updated += 1
    return updated
