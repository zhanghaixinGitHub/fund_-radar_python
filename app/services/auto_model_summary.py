"""日常业务摘要只返回限定公共基金的聚合，不泄漏模型包、全局题集或研究参数。"""

from app.db.session import get_engine
from app.repositories.prediction_store import one, rows
from app.services.auto_model_selection import summary
from app.services.prediction_contract import prediction_policy
from app.services.prediction_direction import direction_fields


def operating_summary(codes):
    with get_engine().connect() as c:
        cycle = one(c, "SELECT * FROM prediction_auto_cycle ORDER BY created_at DESC LIMIT 1")
        attempts = rows(
            c,
            """SELECT DISTINCT ON(fund_code,horizon_id) fund_code,horizon_id,payload,created_at
          FROM prediction_attempt WHERE fund_code=ANY(:codes)
          AND payload->>'targetDefinitionId'=:target AND COALESCE(payload->>'directionPolicyHash','')=:rule
          ORDER BY fund_code,horizon_id,created_at DESC""",
            codes=codes,
            target=prediction_policy()["target_definition_id"],
            rule=direction_fields("T5_V1").get("directionPolicyHash", ""),
        )
        latest = one(
            c,
            """SELECT max(generated_at) generated FROM fund_prediction_record
          WHERE mode='LIVE' AND payload->>'role'='PRIMARY' AND fund_code=ANY(:codes)
          AND payload->>'targetDefinitionId'=:target AND COALESCE(payload->>'directionPolicyHash','')=:rule""",
            codes=codes,
            target=prediction_policy()["target_definition_id"],
            rule=direction_fields("T5_V1").get("directionPolicyHash", ""),
        )
    statuses = {}
    for item in attempts:
        statuses.setdefault(item["fund_code"], []).append(item["payload"].get("generationStatus") != "FAILED")
    return {
        "fundCount": len(set(codes)),
        "successfulFunds": sum(len(v) == 3 and all(v) for v in statuses.values()),
        "partialFailureFunds": sum(any(v) and (not all(v) or len(v) != 3) for v in statuses.values()),
        "failedFunds": sum(not any(v) for v in statuses.values()),
        "pendingFunds": len(set(codes) - statuses.keys()),
        "periodItems": len(attempts),
        "failedItems": sum(v["payload"].get("generationStatus") == "FAILED" for v in attempts),
        "lastGeneratedAt": str(latest["generated"]) if latest["generated"] else None,
        "improvement": {
            "status": cycle["status"],
            "updatedAt": str(cycle["updated_at"]),
            "error": {"summary": cycle["error"].get("summary")} if cycle["error"] else None,
        }
        if cycle
        else None,
    }


def effect_summary(codes):
    """每条预测只用最新一版答案；失败尝试按基金/周期/生成期次去重，不算作正确。"""
    with get_engine().connect() as c:
        counts = rows(
            c,
            """SELECT p.horizon_id,p.payload->>'targetDefinitionId' target_definition_id,
          COALESCE(p.payload->>'directionPolicyHash','') direction_policy_hash,
          count(*) records,count(DISTINCT p.fund_code) funds,
          count(*) FILTER(WHERE o.payload->>'actualDirection'='UP') up_actual,
          count(*) FILTER(WHERE o.payload->>'actualDirection'='FLAT') flat_actual,
          count(*) FILTER(WHERE o.payload->>'actualDirection'='DOWN') down_actual,
          count(*) FILTER(WHERE o.payload->>'actualDirection'='UP' AND (o.payload->>'correct')::boolean) up_correct,
          count(*) FILTER(WHERE o.payload->>'actualDirection'='FLAT' AND (o.payload->>'correct')::boolean) flat_correct,
          count(*) FILTER(WHERE o.payload->>'actualDirection'='DOWN' AND (o.payload->>'correct')::boolean) down_correct,
          count(o.payload) matured,
          count(*) FILTER(WHERE o.payload IS NULL AND COALESCE(s.status,'')<>'FAILED' AND
            COALESCE(t.payload->>'endDate',p.payload->>'endDate',p.payload->>'nominalEndDate')>=
            ((clock_timestamp() AT TIME ZONE 'Asia/Shanghai')::date)::text) unmatured,
          count(*) FILTER(WHERE o.payload IS NULL AND COALESCE(s.status,'')<>'FAILED' AND
            (COALESCE(t.payload->>'endDate',p.payload->>'endDate',p.payload->>'nominalEndDate')<
            ((clock_timestamp() AT TIME ZONE 'Asia/Shanghai')::date)::text OR
            COALESCE(t.payload->>'endDate',p.payload->>'endDate',p.payload->>'nominalEndDate') IS NULL))
            pending_answers,
          count(*) FILTER(WHERE o.payload IS NULL AND s.status='FAILED') check_failed,
          count(*) FILTER(WHERE (o.payload->>'correct')::boolean) correct,
          count(DISTINCT p.fund_code) FILTER(WHERE o.payload IS NOT NULL) matured_funds
          FROM fund_prediction_record p LEFT JOIN LATERAL
          (SELECT payload FROM prediction_outcome WHERE prediction_id=p.prediction_id
            ORDER BY checked_at DESC,outcome_id DESC LIMIT 1) o ON true
          LEFT JOIN prediction_check_state s ON s.prediction_id=p.prediction_id
          LEFT JOIN LATERAL (SELECT payload FROM prediction_target_resolution WHERE prediction_id=p.prediction_id
            ORDER BY created_at DESC,resolution_hash DESC LIMIT 1) t ON true
          WHERE p.mode='LIVE' AND p.payload->>'role'='PRIMARY' AND p.fund_code=ANY(:codes)
          GROUP BY p.horizon_id,p.payload->>'targetDefinitionId',COALESCE(p.payload->>'directionPolicyHash','')
          ORDER BY p.horizon_id,target_definition_id,direction_policy_hash""",
            codes=codes,
        )
        distinct = one(
            c,
            """SELECT count(DISTINCT fund_code) funds FROM fund_prediction_record
          WHERE mode='LIVE' AND payload->>'role'='PRIMARY' AND fund_code=ANY(:codes)""",
            codes=codes,
        )
        failure = one(
            c,
            """SELECT count(*) n FROM
          (SELECT DISTINCT fund_code,horizon_id,
            ((payload->>'generatedAt')::timestamptz AT TIME ZONE 'Asia/Shanghai')::date period
           FROM prediction_attempt WHERE fund_code=ANY(:codes) AND payload->>'generationStatus'='FAILED') a""",
            codes=codes,
        )
    return {
        "mode": "LIVE_MATURED",
        "fundCount": distinct["funds"],
        "followedFundCount": len(set(codes)),
        "horizons": counts,
        "failedPeriods": failure["n"],
        "note": "只统计当时已发出的主预测；最新答案修订不增加预测数；未到期、待答案及核验失败分别列出，不算错误",
    }


def technical_cycles(limit=20, before=None, before_id=None):
    with get_engine().connect() as c:
        values = rows(
            c,
            """SELECT * FROM prediction_auto_cycle WHERE
          (CAST(:before AS timestamptz) IS NULL OR created_at<:before
            OR (created_at=:before AND cycle_id<CAST(:before_id AS uuid)))
          ORDER BY created_at DESC,cycle_id DESC LIMIT :limit""",
            before=before,
            before_id=before_id,
            limit=limit,
        )
    return {
        "items": [
            summary(v)
            | {
                "protocolHash": v["protocol_hash"],
                "checkpoint": {k: x for k, x in v["checkpoint"].items() if k not in {"bundles", "matureEvaluation"}},
            }
            for v in values
        ]
    }
