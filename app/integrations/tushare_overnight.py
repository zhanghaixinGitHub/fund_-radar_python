"""SPX个人研究只读客户端：有界单次查询，凭据不落盘，不进入业务市场同步接口。"""

import json
from datetime import date
from time import monotonic
from urllib.parse import urlsplit

import httpx

from app.core.config import get_settings

FIELDS = ["ts_code", "trade_date", "close", "pre_close", "pct_chg"]


def fetch_spx(start: date, end: date) -> bytes:
    """查询最多31天SPX日线并返回原始成功响应；不重试、不重定向，错误原文不保存。"""
    if not 0 <= (end - start).days <= 31:
        raise ValueError("OVERNIGHT_QUERY_RANGE_INVALID")
    settings = get_settings()
    endpoint = urlsplit(settings.tushare_api_url)
    if (endpoint.scheme, endpoint.hostname, endpoint.path) != ("https", "api.tushare.pro", "") or any(
        (endpoint.query, endpoint.username, endpoint.password, endpoint.fragment, endpoint.port)
    ):
        raise ValueError("OVERNIGHT_OFFICIAL_ENDPOINT_REQUIRED")
    token = settings.tushare_token.get_secret_value()
    if not token:
        raise ValueError("OVERNIGHT_TOKEN_MISSING")
    began = monotonic()
    with httpx.Client(timeout=httpx.Timeout(20, connect=5), follow_redirects=False) as client:
        with client.stream(
            "POST",
            settings.tushare_api_url,
            json={
                "api_name": "index_global",
                "token": token,
                "params": {
                    "ts_code": "SPX",
                    "start_date": start.strftime("%Y%m%d"),
                    "end_date": end.strftime("%Y%m%d"),
                },
                "fields": ",".join(FIELDS),
            },
        ) as response:
            if response.status_code != 200:
                raise ValueError("OVERNIGHT_PROVIDER_HTTP_FAILED")
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > 65536 or monotonic() - began > 30:
                    raise ValueError("OVERNIGHT_RESPONSE_LIMIT")
    if token.encode() in body:
        raise ValueError("OVERNIGHT_RESPONSE_CONTAINS_CREDENTIAL")
    payload = json.loads(body)
    if not isinstance(payload, dict) or type(payload.get("code")) is not int or payload["code"] != 0:
        raise ValueError("OVERNIGHT_PROVIDER_BUSINESS_FAILED")
    return bytes(body)
