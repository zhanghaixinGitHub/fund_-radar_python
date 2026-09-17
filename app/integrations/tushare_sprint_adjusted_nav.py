"""三天研究的复权单位净值只读客户端，独立于已冻结的单位净值采集。

固定官方地址、基金代码格式及字段；最多130天160行，单次请求无自动重试。调用者负责
先登记请求预算和实际到达时间。本模块不购买权限、不更新业务来源登记。
"""

import json
import re
from datetime import date
from time import monotonic
from urllib.parse import urlsplit

import httpx

from app.core.config import get_settings

FIELDS = ["ts_code", "ann_date", "nav_date", "unit_nav", "adj_nav"]


def fetch_adjusted_nav(code: str, start: date, end: date) -> bytes:
    if not re.fullmatch(r"[0-9]{6}\.(OF|SH|SZ)", code) or not 0 <= (end - start).days <= 130:
        raise ValueError("ADJUSTED_QUERY_RANGE_INVALID")
    settings = get_settings()
    endpoint = urlsplit(settings.tushare_api_url)
    if (endpoint.scheme, endpoint.hostname, endpoint.path) != ("https", "api.tushare.pro", "") or any(
        (endpoint.query, endpoint.username, endpoint.password, endpoint.fragment, endpoint.port)
    ):
        raise ValueError("ADJUSTED_OFFICIAL_ENDPOINT_REQUIRED")
    token = settings.tushare_token.get_secret_value()
    if not token:
        raise ValueError("ADJUSTED_TOKEN_MISSING")
    began = monotonic()
    with httpx.Client(timeout=httpx.Timeout(20, connect=5), follow_redirects=False) as client:
        with client.stream(
            "POST",
            settings.tushare_api_url,
            json={
                "api_name": "fund_nav",
                "token": token,
                "params": {
                    "ts_code": code,
                    "start_date": start.strftime("%Y%m%d"),
                    "end_date": end.strftime("%Y%m%d"),
                },
                "fields": ",".join(FIELDS),
            },
        ) as response:
            if response.status_code != 200:
                raise ValueError("ADJUSTED_PROVIDER_HTTP_FAILED")
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > 262144 or monotonic() - began > 30:
                    raise ValueError("ADJUSTED_RESPONSE_LIMIT")
    if token.encode() in body:
        raise ValueError("ADJUSTED_RESPONSE_CONTAINS_CREDENTIAL")
    value = json.loads(body)
    if not isinstance(value, dict) or type(value.get("code")) is not int or value["code"] != 0:
        raise ValueError("ADJUSTED_PROVIDER_BUSINESS_FAILED")
    data = value.get("data") or {}
    if data.get("fields") != FIELDS or not isinstance(data.get("items"), list) or len(data["items"]) > 160:
        raise ValueError("ADJUSTED_RESPONSE_FIELDS_OR_ROWS_INVALID")
    return bytes(body)
