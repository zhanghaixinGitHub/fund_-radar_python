"""原预测解释的唯一生成入口：校验原文、恢复事实、有界调用、检查后保存。"""

import hashlib
import json
import logging
import time
from datetime import UTC, datetime

from app.core.config import get_settings
from app.db.session import get_engine
from app.integrations.prediction_narrative import generate
from app.repositories import direction_1d
from app.repositories import prediction_narrative as store
from app.repositories.prediction_store import one
from app.services.direction_1d_explanation import read_explanation
from app.services.prediction_contract import fingerprint
from app.services.prediction_narrative_facts import STYLE_VERSION, daily_facts, multi_facts, safe_narrative

logger = logging.getLogger(__name__)


def source(kind, source_id, fund_code):
    """只读取实际保存的预测；浏览器只能传记录标识，不能提供待润色的数据或提示词。"""
    if kind == "daily":
        job = direction_1d.get_job(source_id)
        if not job or job["kind"] != "FORECAST" or job["state"] != "SUCCEEDED":
            raise ValueError("FORECAST_NOT_FOUND")
        raw = job["result"]["payload_json"]
        original_hash = hashlib.sha256(raw.encode()).hexdigest()
        if original_hash != job["result"]["content_hash"]:
            raise ValueError("FORECAST_HASH_MISMATCH")
        body = json.loads(raw)
        actual_code = body["fund_code"]
    elif kind == "multi":
        with get_engine().connect() as c:
            row = one(
                c,
                "SELECT payload,content_hash FROM fund_prediction_record "
                "WHERE prediction_id=:id AND fund_code=:code AND mode='LIVE' "
                "AND payload->>'role'='PRIMARY'",
                id=source_id,
                code=fund_code,
            )
        if not row:
            raise ValueError("FORECAST_NOT_FOUND")
        body, original_hash = row["payload"], row["content_hash"]
        if fingerprint(body) != original_hash or body["predictionId"] != str(source_id):
            raise ValueError("FORECAST_HASH_MISMATCH")
        actual_code = body["fundCode"]
    else:
        raise ValueError("SOURCE_KIND_INVALID")
    if actual_code != fund_code:
        raise ValueError("FORECAST_FUND_MISMATCH")
    return body, original_hash


def cached(row, original_hash, fund_code):
    """逐版核对保存正文；模型配置变化不触发重写，内容升级也不覆盖旧版原文。"""
    if not row or row["state"] != "READY":
        return None
    if (
        row["source_hash"] != original_hash
        or row["fund_code"] != fund_code
        or fingerprint(row["payload"]) != row["content_hash"]
    ):
        raise ValueError("SAVED_NARRATIVE_MISMATCH")
    return row["payload"]


def ensure(kind, source_id, fund_code, trace_id=""):
    """显式 POST 整理单条解释；失败只降级解释，绝不重跑预测或改写其方向、日期、结果。"""
    started = time.monotonic()
    body, original_hash = source(kind, source_id, fund_code)
    envelope = {"fundCode": fund_code, "sourceId": str(source_id), "contentHash": original_hash}
    from app.schemas.fund_information_analysis import PROTOCOL as analysis_protocol

    if kind == "daily" and body.get("protocol") == analysis_protocol:
        # 结论与说明已经一起形成；展开依据不再调用外部模型，也不吸收后来资料。
        from app.services.fund_information_analysis import display_narrative

        return {**envelope, "state": "READY", "narrative": display_narrative(body)}
    owner = None
    safe = None

    def fallback(state="FALLBACK"):
        # 旧正文保留在数据库中，但旧版系数因果话术不能再作为显示回退。
        # 安全文本只来自此次已核验的原事实；FALLBACK/PENDING 不伪装为已生成成功。
        return {**envelope, "state": state, **({"narrative": safe} if safe else {})}

    try:
        from app.services import direction_1d_information as information

        if kind == "daily" and body.get("input", {}).get("feature_version") in information.SUPPORTED_VERSIONS:
            from app.services.direction_1d_business_explanation import narrative as business_narrative

            restored = read_explanation(source_id)
            if body["input"]["feature_version"] == information.event_model.VERSION:
                from app.services.direction_1d_event_narrative import narrative as event_narrative
                return {**envelope, "state": "READY", "narrative": event_narrative(body, restored)}
            return {**envelope, "state": "READY", "narrative": business_narrative(body, restored)}
        row = store.read(kind, source_id, STYLE_VERSION)
        saved = cached(row, original_hash, fund_code)
        if saved:
            return {**envelope, "state": "READY", "narrative": saved}
        facts = (
            daily_facts(
                read_explanation(source_id, include_reasoning=True), ternary=body["protocol"] == "DIRECTION_1D_V2"
            )
            if kind == "daily"
            else multi_facts(body)
        )
        safe = safe_narrative(facts)
        settings = get_settings()
        if not settings.deepseek_api_key.get_secret_value() or not settings.deepseek_model:
            return fallback()
        # 校验和资料恢复失败时不发外部请求；失败预算、冷却和并发规则仍按原版本机制执行。
        if row and (
            row["attempts"] >= 3
            or (row["retry_after"] and row["retry_after"] > datetime.now(UTC) and row["style_version"] == STYLE_VERSION)
        ):
            return fallback()
        owner = store.claim(
            kind, source_id, fund_code, original_hash, fingerprint(facts), STYLE_VERSION, settings.deepseek_model
        )
        if not owner:
            row = store.read(kind, source_id, STYLE_VERSION)
            saved = cached(row, original_hash, fund_code)
            if saved:
                return {**envelope, "state": "READY", "narrative": saved}
            pending = not row or (row["state"] == "RUNNING" and row["lease_until"] > datetime.now(UTC))
            return fallback("PENDING" if pending else "FALLBACK")
        narrative = generate(facts, settings)
        if not store.finish(kind, source_id, owner, payload=narrative):
            return fallback()
        logger.info(
            "prediction_narrative.ensure   >>> traceId=%s source=%s/%s state=READY elapsedMs=%d",
            trace_id,
            kind,
            source_id,
            (time.monotonic() - started) * 1000,
        )
        return {**envelope, "state": "READY", "narrative": narrative}
    except Exception as error:
        # 仅白名单业务码可入库；第三方响应及异常文本可能含请求内容，不直接写入日志。
        code = (
            str(error)
            if isinstance(error, ValueError) and str(error).isupper() and len(str(error)) <= 64
            else type(error).__name__
        )
        if owner:
            try:
                store.finish(kind, source_id, owner, failure=code)
            except Exception:
                logger.exception(
                    "prediction_narrative.ensure   >>> traceId=%s source=%s/%s saveFailure", trace_id, kind, source_id
                )
        logger.warning(
            "prediction_narrative.ensure   >>> traceId=%s source=%s/%s state=FALLBACK reason=%s elapsedMs=%d",
            trace_id,
            kind,
            source_id,
            code,
            (time.monotonic() - started) * 1000,
        )
        return fallback()
