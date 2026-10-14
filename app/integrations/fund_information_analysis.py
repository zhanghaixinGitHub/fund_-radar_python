"""有界调用已有服务：事务外请求、不可变缓存、失败无自动重发。"""

import json
import logging
import re
import time
from datetime import datetime
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from sqlalchemy import text

from app.core.deepseek_policy import PrivateDeepSeekSettings, provider_settings, require_scope
from app.db.session import get_engine
from app.services.deepseek_usage import UsageLedger
from app.services.direction_1d_protocol import canonical, digest

logger = logging.getLogger(__name__)


class Budget:
    def __init__(self, deadline: datetime):
        self.until = min(time.monotonic() + 900, time.monotonic() + deadline.timestamp() - time.time())
        self.calls = 0
        self.max_calls = PrivateDeepSeekSettings().max_calls_per_prediction
        self.usage = []

    def check(self):
        if time.monotonic() >= self.until:
            raise ValueError("ANALYSIS_TIME_LIMIT")


class ResponseFormatError(ValueError):
    """保留第一次原始响应供核对，不付费修复或从不完整JSON猜造事实。"""

    def __init__(self, raw: str):
        super().__init__("ANALYSIS_RESPONSE_JSON_INVALID")
        self.raw = raw


def decode_response(raw: str) -> dict:
    """只允许完整JSON对象或完整外层代码围栏；拒绝重复键、NaN与截断补全。"""
    text_value = raw.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n([\s\S]*)\n```", text_value)
    if fence:
        text_value = fence[1]

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("DUPLICATE_KEY")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("NON_FINITE_NUMBER")

    try:
        result = json.loads(text_value, object_pairs_hook=pairs, parse_constant=constant)
        if not isinstance(result, dict):
            raise ValueError("OBJECT_REQUIRED")
        return result
    except (ValueError, TypeError):
        raise ResponseFormatError(raw) from None


def request(prompt: str, value: dict, stage: str, budget: Budget) -> dict:
    scope = require_scope(stage, value)
    if stage == "EVENT":
        from app.services.fund_material_parsing import require_new_material

        require_new_material(value)
    budget.check()
    settings = provider_settings(stage, value)
    endpoint = settings.base_url.rstrip("/")
    url = urlparse(endpoint)
    if url.scheme != "https" or url.hostname != "api.deepseek.com" or url.username or url.query:
        raise ValueError("ANALYSIS_ENDPOINT_INVALID")
    if not settings.api_key.get_secret_value():
        raise ValueError("ANALYSIS_NOT_CONFIGURED")
    key = digest({"prompt": prompt, "input": value, "model": settings.model, "endpoint": endpoint})
    owner = uuid4()
    with get_engine().begin() as c:
        c.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:key,721136))"), {"key": key})
        row = (
            c.execute(
                text(
                    "SELECT *,lease_until>clock_timestamp() active FROM fund_analysis_event_parse WHERE cache_key=:key"
                ),
                {"key": key},
            )
            .mappings()
            .first()
        )
        if row and row["state"] == "READY":
            if digest(row["payload"]) != row["content_hash"]:
                raise ValueError("ANALYSIS_CACHE_MISMATCH")
            return row["payload"]
        # 同一请求一旦发出便不自动重发，超时/失败也不能通过再次点击消耗余额。
        if row and not (row["state"] == "FAILED" and row.get("failure_code") == "DEEPSEEK_DAILY_BUDGET"):
            raise ValueError("ANALYSIS_STAGE_PENDING" if row["state"] == "RUNNING" else "ANALYSIS_STAGE_FAILED")
        if budget.calls >= budget.max_calls:
            raise ValueError("ANALYSIS_CALL_LIMIT")
        c.execute(
            text("""INSERT INTO fund_analysis_event_parse(cache_key,stage,state,owner_token,lease_until,attempts)
          VALUES(:key,:stage,'RUNNING',:owner,clock_timestamp()+interval '120 seconds',1)
          ON CONFLICT(cache_key) DO UPDATE SET state='RUNNING',owner_token=:owner,
          lease_until=clock_timestamp()+interval '120 seconds',attempts=fund_analysis_event_parse.attempts+1
          WHERE fund_analysis_event_parse.state<>'READY'"""),
            {"key": key, "stage": stage, "owner": owner},
        )
    output_limit = {"EVENT": 2048, "SYNTHESIS": 3000, "AUDIT": 800}[stage]
    ledger = UsageLedger()
    usage, response_id, reserved = None, None, False
    raw_record = None
    try:
        ledger.reserve(
            key, scope, settings, stage, len((prompt + canonical(value)).encode("utf-8")) + 1024, output_limit
        )
        reserved = True
        budget.calls += 1
        # 正文明确作为数据，不给模型任何工具或抓取能力。相同响应原样保留，解析错误也不重发。
        with httpx.Client(
            timeout=httpx.Timeout(min(90, max(1, budget.until - time.monotonic())), connect=5), follow_redirects=False
        ) as client:
            response = client.post(
                endpoint + "/chat/completions",
                headers={"Authorization": "Bearer " + settings.api_key.get_secret_value()},
                json={
                    "model": settings.model,
                    "stream": False,
                    "thinking": {"type": "disabled"},
                    "temperature": 0,
                    "max_tokens": output_limit,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": canonical(value)}],
                },
            )
        if response.status_code != 200 or len(response.content) > 250000:
            raise ValueError("ANALYSIS_PROVIDER_FAILED")
        raw_record = {"http_text": response.text}
        envelope = response.json()
        usage, response_id = envelope.get("usage"), envelope.get("id")
        choice = envelope["choices"][0]
        raw_record = {
            "content": choice["message"]["content"],
            "finish_reason": choice.get("finish_reason"),
            "usage": envelope.get("usage"),
        }
        if choice.get("finish_reason") != "stop":
            raise ValueError("ANALYSIS_OUTPUT_INCOMPLETE")
        result = decode_response(choice["message"]["content"])
        with get_engine().begin() as c:
            updated = c.execute(
                text("""UPDATE fund_analysis_event_parse SET state='READY',payload=CAST(:value AS jsonb),
              content_hash=:hash,finished_at=clock_timestamp() WHERE cache_key=:key AND owner_token=:owner
              AND state='RUNNING' RETURNING cache_key"""),
                {"value": canonical(result), "hash": digest(result), "key": key, "owner": owner},
            ).first()
            if not updated:
                raise ValueError("ANALYSIS_LEASE_LOST")
        budget.check()
        receipt = ledger.finish(key, usage, response_id, "SUCCEEDED")
        budget.usage.append({"stage": stage, **receipt})
        logger.info(
            "fund_information_analysis.request   >>> fund=002112 horizon=1 target=%s stage=%s "
            "calls=%d input=%s output=%s estimatedPeakCny=%s",
            scope.target_nav_date,
            stage,
            budget.calls,
            receipt["prompt_tokens"],
            receipt["completion_tokens"],
            receipt["estimated_peak_cny"],
        )
        return result
    except Exception as error:
        code = str(error) if isinstance(error, ValueError) and str(error).isupper() else "ANALYSIS_PROVIDER_FAILED"
        with get_engine().begin() as c:
            c.execute(
                text("""UPDATE fund_analysis_event_parse SET state='FAILED',failure_code=:code,
              payload=CAST(:raw AS jsonb),
              finished_at=clock_timestamp() WHERE cache_key=:key AND owner_token=:owner AND state='RUNNING'"""),
                {"code": code[:64], "key": key, "owner": owner, "raw": canonical(raw_record)},
            )
        if reserved:
            ledger.finish(key, usage, response_id, "FAILED" if usage else "UNKNOWN")
        if isinstance(error, ResponseFormatError):
            raise
        raise ValueError(code) from None
