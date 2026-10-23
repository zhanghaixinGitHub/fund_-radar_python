"""多周期综合审阅的冻结、导入及只读消费；没有统计模型或付费回退路径。"""

import logging
from datetime import date, datetime, timedelta
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


def review_calendar(data: dict, code: str, at: datetime):
    """复核资料包已证明的估值政策，并限定多周期可证实的日历覆盖。

    仅为完整披露股票均属境内、且有本基金估值条款的已核实政策增加衔接。
    再核对原文、全部证券、基准行情及项目日历摘要，不接受包内单个就绪标志。
    港股通及人民币股票QDII仅使用境内外正常交易日完全一致的未来区间；
    第一个休市差异、半日市或来源年度边界即停止覆盖，不推断差异日估值。
    已逐项核实的多市场上市原油产品另查自身开放日条款；仍只延续日历一致区间。
    黄金及其他上市基金组合继续使用原路径，不从股票条款泛化其他资产。
    """
    from app.services.fund_listed_product_contract import POLICY as multi_listed_policy
    from app.services.prediction_features import read_fund_data

    mode = (data.get("calendar_evidence") or {}).get("policy")
    if mode not in {
        "CN_NAV_DISCLOSED_DOMESTIC_ASSETS_V1",
        "CN_NAV_HK_CONNECT_WINDOW_V1",
        "QDII_CNY_VERIFIED_SESSION_V1",
        multi_listed_policy,
    }:
        return read_fund_data(code, at)["calendar"]
    from app.services.comprehensive_calendar import verified_domestic_window
    from app.services.prediction_contract import ValuationCalendar
    from app.services.trading_calendar import load_calendar, load_current_calendar

    with get_engine().connect() as connection:
        profiles = repo.profiles(connection, [code])
    documents = {document["id"]: document for document in data["documents"]}
    past, current = load_calendar(), load_current_calendar()
    calendar_hash = digest([past.content_hash, current.content_hash])
    from app.services.qdii_valuation_contract import verified_qdii_window

    verified = (
        verified_qdii_window
        if mode in {"QDII_CNY_VERIFIED_SESSION_V1", multi_listed_policy}
        else verified_domestic_window
    )
    if (
        data.get("fund_code") != code
        or len(profiles) != 1
        or not data.get("time_context", {}).get("baseline_complete")
        or data["window"]["calendar_version"] != calendar_hash
        or not verified(profiles[0], data, documents, data["window"])
    ):
        raise PredictionFailure("CALENDAR_POLICY_MISSING", "CALENDAR", "本基金实际投向或估值条款仍待核验")
    sessions, end, source_hash = past.sessions + current.sessions, current.definition.coverage_end, calendar_hash
    if mode != "CN_NAV_DISCLOSED_DOMESTIC_ASSETS_V1":
        sessions, end, source_hash = _matched_future_sessions(data, documents, sessions, end, calendar_hash)
    return ValuationCalendar(
        mode,
        sessions,
        past.definition.coverage_start,
        end,
        source_hash,
    )


def _matched_future_sessions(data, documents, sessions, coverage_end, calendar_hash):
    """仅延续官方日历一致的有限区间；基准日净值和本期休市参考已在上游复核。

    未来中外交易日不同不等于基金必然不估值，本函数对此没有作出判断，直接
    截断可用区间。因此不能将日历交集当作基金的永久估值政策，也不能跨年猜测。
    """
    import json
    import re
    from urllib.parse import urlparse

    from app.services.fund_listed_product_contract import POLICY as multi_listed_policy
    from app.services.fund_listed_product_contract import european_calendar
    from app.services.fund_valuation_sessions import hk_closed_dates

    evidence = data["calendar_evidence"]
    own = documents.get(evidence.get("prospectus_ref"), {})
    body = re.sub(r"\s+", "", own.get("body", ""))
    listed = evidence["policy"] == multi_listed_policy
    open_quote = evidence.get("joint_open_days_quote", "")
    listed_clause = (
        listed
        and open_quote in own.get("body", "")
        and "共同交易日" in open_quote
        and all(market in open_quote for market in ("纽约", "伦敦", "东京"))
        and "估值日为本基金的开放日" in body
    )
    if (
        data["fund_code"] not in own.get("codes", [])
        or own.get("kind") != "PROSPECTUS"
        # 部分基金原文明确写“共同交易日”。下方本已只延续中外交易日完全一致
        # 的区间，识别这个更严格的真实条款不意味着放开任一市场单独休市的日期。
        or not (
            listed_clause
            or re.search(r"估值日为本基金相关的证券(?:[/／、]期货)?交易场所的(?:共同)?交易日", body)
        )
    ):
        raise PredictionFailure("CALENDAR_POLICY_MISSING", "CALENDAR", "本基金未来估值日条款仍待核验")
    target = date.fromisoformat(data["window"]["target_nav_date"])
    if evidence["policy"] == "CN_NAV_HK_CONNECT_WINDOW_V1":
        markets, references = {"HK"}, {"HK": evidence.get("calendar_ref")}
        if any(not c["code"].endswith((".HK", ".SH", ".SZ", ".BJ")) for c in data["companies"]):
            raise PredictionFailure("CALENDAR_POLICY_MISSING", "CALENDAR", "实际投向包含尚未核验的交易市场")
    else:
        markets = {
            q.get("valuation_market")
            for q in [*(c.get("quote", {}) for c in data["companies"]), *data["market"].values()]
        } - {"CN"}
        references = evidence.get("calendar_refs", {})
    if not markets or not markets <= ({"HK", "US", "JP", "UK", "CH"} if listed else {"HK", "US", "JP"}):
        raise PredictionFailure("CALENDAR_POLICY_MISSING", "CALENDAR", "未来交易市场日历范围仍待核验")
    calendars, hashes = [], [calendar_hash, own["body_hash"]]
    end = min(coverage_end, date(target.year, 12, 31))
    for market in sorted(markets):
        source = documents[references[market]]
        host = urlparse(source.get("url", "")).hostname
        if listed and market in {"UK", "CH"}:
            try:
                closed, start, stop = european_calendar(market, source, target.year)
            except (KeyError, TypeError, ValueError) as error:
                raise PredictionFailure("CALENDAR_POLICY_MISSING", "CALENDAR", "上市产品交易日历仍待核验") from error
            if target < start or target > stop:
                raise PredictionFailure("CALENDAR_POLICY_MISSING", "CALENDAR", "当前目标超出已核实日历覆盖")
            end = min(end, stop)
        elif market == "HK":
            # 完整港交所通告同时列出半日市，本接口保守视为未支持的日期。
            closed = hk_closed_dates(source, target.year)
            if not closed:
                raise PredictionFailure("CALENDAR_POLICY_MISSING", "CALENDAR", "香港年度日历尚未核验")
        else:
            parsed = json.loads(source["body"])
            if (
                source.get("kind") != "EXCHANGE_CALENDAR"
                or parsed["year"] != target.year
                or host != {"US": "www.nyse.com", "JP": "www.jpx.co.jp"}[market]
            ):
                raise PredictionFailure("CALENDAR_POLICY_MISSING", "CALENDAR", "境外年度日历尚未核验")
            closed = {date.fromisoformat(d) for d in parsed["closed_dates"] + parsed.get("early_close_dates", [])}
        calendars.append(closed)
        hashes.append(source["body_hash"])
    domestic = set(sessions)
    day = target
    while day <= end:
        if any((day in domestic) != (day.weekday() < 5 and day not in closed) for closed in calendars):
            end = day - timedelta(days=1)
            break
        day += timedelta(days=1)
    return tuple(d for d in sessions if d <= end), end, digest(hashes)


def freeze_packet(directory: Path, code: str, horizon: str, materials_path=None) -> dict:
    """分别冻结周期和日期；不能将已完成的一日判断复制为其他周期。"""
    if horizon not in {h["horizon_id"] for h in prediction_policy()["horizons"]}:
        raise ValueError("COMPREHENSIVE_HORIZON_INVALID")
    now = repo.clock()
    data = prepare(now, code, materials_path)
    calendar = review_calendar(data, code, now)
    contents = {
        "data": data,
        "dates": target_dates(calendar, now, horizon),
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
    calendar = review_calendar(data, code, cutoff)
    if target_dates(calendar, cutoff, horizon) != contents["dates"]:
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
