"""三天研究的两融汇总只读客户端，固定官方接口、字段和请求上限。

金额原值为人民币元；不购买权限，不修改业务来源登记。调用方登记时槽和到达证据。
"""

import json
from datetime import date
from time import monotonic
from urllib.parse import urlsplit

import httpx

from app.core.config import get_settings

FIELDS = ["trade_date", "exchange_id", "rzye", "rzmre", "rzche", "rqye", "rzrqye"]
MAX_BYTES = 524288
URL = "https://api.tushare.pro"


def fetch_margin(start: date, end: date) -> bytes:
    """最多读取一个自然年的两融金额；无重定向、无自动重试，不输出凭据或原始异常响应。"""
    if not 0 <= (end - start).days <= 366:
        raise ValueError("MARGIN_QUERY_RANGE_INVALID")
    settings = get_settings()
    endpoint = urlsplit(settings.tushare_api_url)
    if (endpoint.scheme, endpoint.hostname, endpoint.path) != ("https", "api.tushare.pro", "") or any(
        (endpoint.query, endpoint.username, endpoint.password, endpoint.fragment, endpoint.port)
    ):
        raise ValueError("MARGIN_OFFICIAL_ENDPOINT_REQUIRED")
    token = settings.tushare_token.get_secret_value()
    if not token:
        raise ValueError("MARGIN_TOKEN_MISSING")
    began = monotonic()
    with httpx.Client(timeout=httpx.Timeout(20, connect=5), follow_redirects=False) as client:
        with client.stream(
            "POST",
            settings.tushare_api_url,
            json={
                "api_name": "margin",
                "token": token,
                "params": {"start_date": start.strftime("%Y%m%d"), "end_date": end.strftime("%Y%m%d")},
                "fields": ",".join(FIELDS),
            },
        ) as response:
            if response.status_code != 200:
                raise ValueError("MARGIN_PROVIDER_HTTP_FAILED")
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > MAX_BYTES or monotonic() - began > 30:
                    raise ValueError("MARGIN_RESPONSE_LIMIT")
    if token.encode() in body:
        raise ValueError("MARGIN_RESPONSE_CONTAINS_CREDENTIAL")
    value = json.loads(body)
    if not isinstance(value, dict) or type(value.get("code")) is not int or value["code"] != 0:
        raise ValueError("MARGIN_PROVIDER_BUSINESS_FAILED")
    data = value.get("data") or {}
    if (
        data.get("has_more", False)
        or data.get("fields") != FIELDS
        or not isinstance(data.get("items"), list)
        or len(data["items"]) > 1100
    ):
        raise ValueError("MARGIN_RESPONSE_FIELDS_OR_ROWS_INVALID")
    return bytes(body)
