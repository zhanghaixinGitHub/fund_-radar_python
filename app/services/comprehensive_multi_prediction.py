"""多周期综合审阅的冻结、导入及只读消费；没有统计模型或付费回退路径。"""

import logging
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import text

from app.db.session import get_engine
from app.repositories import direction_1d as repo
from app.repositories.prediction_store import encode, one, save_prediction, snapshot
from app.schemas.fund_information_analysis import validate_analysis
from app.services.comprehensive_prediction_policy import identity, policy, validate_review
from app.services.direction_1d_protocol import digest, window
from app.services.fund_codex_analysis import SOURCE, catalog
from app.services.fund_exposure_common import read, save
from app.services.fund_information_analysis import narrative
from app.services.fund_information_contracts import time_context
from app.services.fund_information_snapshot import prepare
from app.services.prediction_contract import PredictionFailure, prediction_policy, target_dates
from app.services.prediction_direction import direction_fields

logger = logging.getLogger(__name__)


def freeze_packet(directory: Path, code: str, horizon: str, materials_path=None) -> dict:
    """分别冻结周期和日期；不能将已完成的一日判断复制为其他周期。"""
    from app.services.prediction_features import read_fund_data

    if horizon not in {h["horizon_id"] for h in prediction_policy()["horizons"]}:
        raise ValueError("COMPREHENSIVE_HORIZON_INVALID")
    now = repo.clock()
    data = prepare(now, code, materials_path)
    calendar_data = read_fund_data(code, now)
    contents = {
        "data": data,
        "dates": target_dates(calendar_data["calendar"], now, horizon),
        "horizon_id": horizon,
        "prediction_policy": prediction_policy(),
        "analysis_policy": policy(),
    }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    packet = {"format": "CODEX_COMPREHENSIVE_MULTI_V1", "input_hash": digest(contents), "input": contents}
    save(directory / "input.json", packet)
    return {
        "fund_code": code,
        "horizon_id": horizon,
        "target_date": contents["dates"]["startDate"],
        "input_hash": packet["input_hash"],
        "directory": str(directory.resolve()),
    }


def import_review(packet_path: Path, review_path: Path) -> dict:
    """只导入本机完成的资料审阅；发布对象和时间由冻结包与数据库决定。"""
    packet, review = read(packet_path), read(review_path)
    contents = packet["input"]
    if (
        packet.get("format") != "CODEX_COMPREHENSIVE_MULTI_V1"
        or digest(contents) != packet["input_hash"]
        or review.get("input_hash") != packet["input_hash"]
        or review.get("generation_source") != SOURCE
        or contents["analysis_policy"] != policy()
        or contents["prediction_policy"] != prediction_policy()
    ):
        raise ValueError("COMPREHENSIVE_REVIEW_IDENTITY_INVALID")
    data, horizon = contents["data"], contents["horizon_id"]
    if horizon not in {h["horizon_id"] for h in prediction_policy()["horizons"]}:
        raise ValueError("COMPREHENSIVE_HORIZON_INVALID")
    now = repo.clock()
    cutoff = datetime.fromisoformat(data["as_of"])
    if (
        cutoff > now
        or window(now)["target_nav_date"] != contents["dates"]["startDate"]
        or now >= datetime.fromisoformat(data["window"]["deadline_at"])
    ):
        raise ValueError("MISSED_DEADLINE")
    code = data["fund_code"]
    from app.services.prediction_features import read_fund_data

    calendar_data = read_fund_data(code, cutoff)
    if target_dates(calendar_data["calendar"], cutoff, horizon) != contents["dates"]:
        raise ValueError("COMPREHENSIVE_DATE_CONTRACT_MISMATCH")
    restored = prepare(cutoff, code, data.get("materials_path"))
    if digest(restored) != digest(data):
        raise ValueError("MANUAL_FROZEN_SOURCE_CHANGED")
    if review.get("horizon_id") != horizon:
        raise ValueError("COMPREHENSIVE_HORIZON_REVIEW_REQUIRED")
    facts, _, coverage = catalog(data, review)
    validate_review(data, review, facts)
    analysis = validate_analysis(review["analysis"], facts, time_context(data))
    if review.get("audit") != {"valid": True, "issues": []}:
        raise ValueError("MANUAL_REVIEW_FAILED")
    review_hash = digest(review)
    key = f"LIVE:{code}:{horizon}:{contents['dates']['startDate']}:COMPREHENSIVE:{digest(policy())}:{review_hash}"
    manifest = {
        "recipeVersion": policy()["version"],
        "adapter": "COMPREHENSIVE_REVIEW_V1",
        "horizonId": horizon,
        "features": [],
        "trainedAt": None,
        "labelEndMax": None,
        "evidenceLevel": "UNVALIDATED",
        "analysisPolicyHash": digest(policy()),
        "description": "综合资料审阅规则的身份，不是训练模型或准确率验证",
    }
    model_id = f"COMPREHENSIVE_V1:{horizon}"
    with get_engine().begin() as c:
        # 同一基金同一周期的相同审阅复用原记录和原时间；不覆盖首次结果。
        c.execute(text("SELECT pg_advisory_xact_lock(hashtext(:key))"), {"key": key})
        previous = one(c, "SELECT payload,content_hash FROM fund_prediction_record WHERE period_key=:key", key=key)
        if previous:
            if digest(previous["payload"]) != previous["content_hash"]:
                raise ValueError("PREDICTION_EVIDENCE_HASH_MISMATCH")
            return {"prediction_id": previous["payload"]["predictionId"], "reused": True, "calls": 0}
        generated = c.execute(text("SELECT clock_timestamp()")).scalar_one()
        if generated >= datetime.fromisoformat(data["window"]["deadline_at"]):
            raise ValueError("MISSED_DEADLINE")
        c.execute(
            text("""INSERT INTO model_artifact(model_id,content_hash,horizon_id,target_definition_id,
            asset_group,adapter,manifest)
            VALUES(:id,:hash,:horizon,:target,'ALL','COMPREHENSIVE_REVIEW_V1',CAST(:m AS jsonb))
            ON CONFLICT(model_id) DO NOTHING"""),
            {
                "id": model_id,
                "hash": digest(manifest),
                "horizon": horizon,
                "target": prediction_policy()["target_definition_id"],
                "m": encode(manifest),
            },
        )
        registered = one(c, "SELECT content_hash FROM model_artifact WHERE model_id=:id", id=model_id)
        if registered["content_hash"] != digest(manifest):
            raise ValueError("COMPREHENSIVE_RULE_IDENTITY_CONFLICT")
        source = {"data": data, "review": review, "facts": facts, "coverage": coverage}
        snapshot_id = snapshot(c, code, cutoff, source)
        display = narrative(
            analysis, facts, data["inventory"], ["部分资料仅审阅片段或尚未完成核实。"] if coverage else []
        )
        result = {
            "predictionId": str(uuid4()),
            "fundCode": code,
            "horizonId": horizon,
            **contents["dates"],
            **direction_fields(horizon),
            **identity(),
            "generationSource": SOURCE,
            "analysis": analysis,
            "narrative": display,
            "reviewHash": review_hash,
            "mode": "LIVE",
            "role": "PRIMARY",
            "generationStatus": "SUCCEEDED",
            "generatedAt": generated.isoformat(),
            "knowledgeCutoff": cutoff.isoformat(),
            "dataAsOf": data["latest_nav_date"] or data["window"]["base_nav_date"],
            "reason": analysis["summary"],
            "limitations": display["limitations"],
            "modelId": model_id,
            "modelHash": digest(manifest),
            "modelManifest": manifest,
            "activationRevision": 1,
            "baseline": False,
            "fallbackReason": None,
            "direction": analysis["direction"],
            "confidence": "LOW",
            "upProbability": None,
            "validationStatus": "UNVALIDATED",
            "predictionPolicySnapshot": prediction_policy(),
            "featureSnapshot": {
                "fundCode": code,
                "dataAsOf": data["latest_nav_date"],
                "features": {},
                "inputSnapshotId": str(snapshot_id),
            },
            "featureHash": digest(source),
        }
        if len(encode(result).encode("utf-8")) > 1_048_576:
            raise ValueError("ANALYSIS_RESULT_LIMIT")
        stored, created = save_prediction(c, result, key)
    return {"prediction_id": stored["predictionId"], "reused": not created, "calls": 0}


def ready_result(code: str, horizon: str, now) -> dict:
    """网页及后台只消费当前期次的综合原文；无结果时明确等待审阅。"""
    target = window(now)["target_nav_date"]
    with get_engine().connect() as c:
        row = one(
            c,
            """SELECT payload,content_hash FROM fund_prediction_record WHERE fund_code=:code
            AND horizon_id=:horizon AND mode='LIVE' AND payload->>'role'='PRIMARY'
            AND payload->>'startDate'=:target AND payload->>'generationMethod'='COMPREHENSIVE'
            AND payload->>'analysisPolicyHash'=:hash ORDER BY generated_at DESC,prediction_id DESC LIMIT 1""",
            code=code,
            horizon=horizon,
            target=target,
            hash=digest(policy()),
        )
    if not row:
        raise PredictionFailure(
            "MANUAL_ANALYSIS_REQUIRED",
            "COMPREHENSIVE_REVIEW",
            "本期资料分析尚未完成，暂未生成预测",
            next_action="完成本基金本周期资料分析后重试",
        )
    value = row["payload"]
    if (
        digest(value) != row["content_hash"]
        or value.get("fundCode") != code
        or value.get("horizonId") != horizon
        or any(value.get(k) != v for k, v in identity().items())
    ):
        raise PredictionFailure("COMPREHENSIVE_RESULT_INVALID", "COMPREHENSIVE_REVIEW", "综合判断与原始资料未通过核对")
    return value


def run_task(task_id):
    """逐项消费已导入结果；旧排队任务也不得恢复旧净值模型。

    沿用既有任务表、执行会话锁和逐项事务；失败只影响本基金本周期。开始时刻及
    每项完成时刻都核对目标窗口，跨过截止后不擅自换成下一期。
    """
    from datetime import timedelta

    from app.repositories.prediction_store import rows

    with get_engine().begin() as c:
        task = one(c, "SELECT * FROM prediction_generation_task WHERE task_id=:id FOR UPDATE", id=task_id)
        now = c.execute(text("SELECT clock_timestamp()")).scalar_one()
        if not task or task["status"] in {"SUCCEEDED", "PARTIAL_SUCCESS", "FAILED", "CANCELLED"}:
            return
        if task["lease_until"] and task["lease_until"] > now:
            return
        c.execute(
            text("UPDATE prediction_generation_task SET status='RUNNING',lease_until=:lease WHERE task_id=:id"),
            {"id": task_id, "lease": now + timedelta(minutes=5)},
        )
        items = rows(
            c,
            "SELECT fund_code,horizon_id FROM prediction_generation_item WHERE task_id=:id "
            "AND status IN ('PENDING','RUNNING') ORDER BY fund_code,horizon_id",
            id=task_id,
        )
    requested = window(datetime.fromisoformat(task["payload"]["generatedAt"]))["target_nav_date"]
    for item in items:
        code, horizon = item["fund_code"], item["horizon_id"]
        try:
            now = repo.clock()
            if task["payload"].get("generationMethod") != "COMPREHENSIVE":
                raise PredictionFailure(
                    "NAV_ONLY_PREDICTION_DISABLED", "COMPREHENSIVE_REVIEW", "旧预测任务已停用，请重新发起综合资料分析"
                )
            if window(now)["target_nav_date"] != requested:
                raise PredictionFailure("WINDOW_CHANGED", "COMPREHENSIVE_REVIEW", "预测日期已切换，请重新发起")
            result = ready_result(code, horizon, now)
            state = "REUSED"
        except PredictionFailure as error:
            result = {
                "generationStatus": "FAILED",
                "error": error.payload,
                "generatedAt": now.isoformat(),
                **identity(),
            }
            state = "FAILED"
        except Exception:
            logger.exception(
                "comprehensive_multi_prediction.run_task   >>> task=%s fund=%s horizon=%s", task_id, code, horizon
            )
            # 不回传完整SQL、来源正文或凭据，已保存的成功结果不受其他项目失败影响。
            result = {
                "generationStatus": "FAILED",
                "generatedAt": now.isoformat(),
                **identity(),
                "error": {
                    "code": "COMPREHENSIVE_READ_FAILED",
                    "stage": "COMPREHENSIVE_REVIEW",
                    "summary": "综合判断暂时无法核对，请稍后重试",
                    "retryable": True,
                },
            }
            state = "FAILED"
        result = {
            **result,
            "targetDefinitionId": prediction_policy()["target_definition_id"],
            **direction_fields(horizon),
        }
        with get_engine().begin() as c:
            state_now = one(c, "SELECT status FROM prediction_generation_task WHERE task_id=:id FOR UPDATE", id=task_id)
            if state_now["status"] == "CANCELLED":
                return
            completed = c.execute(text("SELECT clock_timestamp()")).scalar_one()
            if state == "REUSED" and window(completed)["target_nav_date"] != requested:
                state = "FAILED"
                result = {
                    "generationStatus": "FAILED",
                    "generatedAt": completed.isoformat(),
                    **identity(),
                    "targetDefinitionId": prediction_policy()["target_definition_id"],
                    **direction_fields(horizon),
                    "error": {
                        "code": "WINDOW_CHANGED",
                        "stage": "COMPREHENSIVE_REVIEW",
                        "summary": "预测日期已切换，未保存为本期成功",
                        "retryable": True,
                    },
                }
            c.execute(
                text(
                    "INSERT INTO prediction_attempt(attempt_id,task_id,fund_code,horizon_id,payload) "
                    "VALUES(:id,:task,:code,:horizon,CAST(:payload AS jsonb))"
                ),
                {"id": uuid4(), "task": task_id, "code": code, "horizon": horizon, "payload": encode(result)},
            )
            c.execute(
                text(
                    "UPDATE prediction_generation_item SET status=:status,attempts=attempts+1,"
                    "prediction_id=:prediction,result=CAST(:payload AS jsonb) WHERE task_id=:task "
                    "AND fund_code=:code AND horizon_id=:horizon"
                ),
                {
                    "status": state,
                    "prediction": result.get("predictionId") if state == "REUSED" else None,
                    "payload": encode(result),
                    "task": task_id,
                    "code": code,
                    "horizon": horizon,
                },
            )
            c.execute(
                text(
                    "UPDATE prediction_generation_task SET lease_until=clock_timestamp()+interval '5 minutes' "
                    "WHERE task_id=:task"
                ),
                {"task": task_id},
            )
    with get_engine().begin() as c:
        state_now = one(c, "SELECT status FROM prediction_generation_task WHERE task_id=:id FOR UPDATE", id=task_id)
        if state_now["status"] == "CANCELLED":
            return
        counts = one(
            c,
            "SELECT count(*) FILTER(WHERE status='FAILED') AS failed,"
            "count(*) FILTER(WHERE status IN ('CREATED','REUSED')) AS succeeded "
            "FROM prediction_generation_item WHERE task_id=:id",
            id=task_id,
        )
        status = (
            "PARTIAL_SUCCESS"
            if counts["failed"] and counts["succeeded"]
            else "FAILED"
            if counts["failed"]
            else "SUCCEEDED"
        )
        c.execute(
            text(
                "UPDATE prediction_generation_task SET status=:status,finished_at=clock_timestamp(),"
                "lease_until=NULL WHERE task_id=:id"
            ),
            {"status": status, "id": task_id},
        )
