"""五个预先选定行业指数的只读客户端；调用方登记预算、请求与实际接收时间。

只连接Tushare官方HTTPS接口，不新增权限，不输出凭据，不自动重试。
"""

import json
from datetime import date
from time import monotonic
from urllib.parse import urlsplit

import httpx

from app.core.config import get_settings

CODES = ("399998.SZ", "399986.SZ", "980017.SZ", "000819.SH", "980030.SZ")
FIELDS = ["ts_code", "trade_date", "close", "pre_close"]
MAX_BYTES = 1048576
URL = "https://api.tushare.pro"


def fetch_sector(code: str, start: date, end: date) -> bytes:
    """只读取固定五指数内的一只，窗口最多31个自然日；无重定向、无自动重试，不输出凭据或原始异常响应。"""
    if code not in CODES or not 0 <= (end - start).days <= 31:
        raise ValueError("SECTOR_QUERY_RANGE_INVALID")
    settings = get_settings()
    endpoint = urlsplit(settings.tushare_api_url)
    if (endpoint.scheme, endpoint.hostname, endpoint.path) != ("https", "api.tushare.pro", "") or any(
        (endpoint.query, endpoint.username, endpoint.password, endpoint.fragment, endpoint.port)
    ):
        raise ValueError("SECTOR_OFFICIAL_ENDPOINT_REQUIRED")
    token = settings.tushare_token.get_secret_value()
    if not token:
        raise ValueError("SECTOR_TOKEN_MISSING")
    began = monotonic()
    with httpx.Client(timeout=httpx.Timeout(20, connect=5), follow_redirects=False) as client:
        with client.stream(
            "POST",
            settings.tushare_api_url,
            json={
                "api_name": "index_daily",
                "token": token,
                "params": {"ts_code": code, "start_date": start.strftime("%Y%m%d"), "end_date": end.strftime("%Y%m%d")},
                "fields": ",".join(FIELDS),
            },
        ) as response:
            if response.status_code != 200:
                raise ValueError("SECTOR_PROVIDER_HTTP_FAILED")
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > MAX_BYTES or monotonic() - began > 30:
                    raise ValueError("SECTOR_RESPONSE_LIMIT")
    if token.encode() in body:
        raise ValueError("SECTOR_RESPONSE_CONTAINS_CREDENTIAL")
    value = json.loads(body)
    if not isinstance(value, dict) or type(value.get("code")) is not int or value["code"] != 0:
        raise ValueError("SECTOR_PROVIDER_BUSINESS_FAILED")
    data = value.get("data") or {}
    if data.get("fields") != FIELDS or not isinstance(data.get("items"), list) or len(data["items"]) > 366:
        raise ValueError("SECTOR_RESPONSE_FIELDS_OR_ROWS_INVALID")
    return bytes(body)
