"""一日研究用IF原始主力日线/映射解析：价格为指数点，不使用复权或跨合约涨幅。"""

import json
import math
import re
from datetime import datetime

FIELDS = {
    "fut_daily": ["ts_code", "trade_date", "pre_close", "open", "high", "low", "close", "settle", "vol", "oi"],
    "fut_mapping": ["ts_code", "trade_date", "mapping_ts_code"],
    "index_daily": ["ts_code", "trade_date", "close"],
}


def parse(raw: bytes, api: str, start: str, end: str) -> dict:
    """每次限一年、最多300行；日期、价格和合约标识异常直接拒绝，不插值补空。"""
    if api not in FIELDS or not raw or len(raw) > 1_048_576:
        raise ValueError("FUTURES_RESPONSE_SCOPE_INVALID")
    value = json.loads(raw)
    if type(value.get("code")) is not int or value["code"] != 0:
        raise ValueError("FUTURES_PROVIDER_REJECTED")
    data = value.get("data") or {}
    if (
        data.get("fields") != FIELDS[api]
        or not isinstance(data.get("items"), list)
        or not 1 <= len(data["items"]) <= 300
    ):
        raise ValueError("FUTURES_SCHEMA_OR_COUNT_INVALID")
    result = {}
    for item in data["items"]:
        if not isinstance(item, list) or len(item) != len(FIELDS[api]):
            raise ValueError("FUTURES_ROW_SHAPE_INVALID")
        row = dict(zip(FIELDS[api], item, strict=True))
        day = row["trade_date"]
        if not isinstance(day, str) or not re.fullmatch(r"\d{8}", day):
            raise ValueError("FUTURES_DATE_INVALID")
        parsed = datetime.strptime(day, "%Y%m%d").date()
        if not start <= day <= end or parsed.weekday() >= 5 or str(parsed) in result:
            raise ValueError("FUTURES_DATE_SCOPE_OR_DUPLICATE")
        if row["ts_code"] != ("000300.SH" if api == "index_daily" else "IF.CFX"):
            raise ValueError("FUTURES_IDENTITY_INVALID")
        if api == "fut_mapping":
            if not isinstance(row["mapping_ts_code"], str) or not re.fullmatch(r"IF\d{4}\.CFX", row["mapping_ts_code"]):
                raise ValueError("FUTURES_CONTRACT_MAPPING_INVALID")
        else:
            prices = ["close"] if api == "index_daily" else ["pre_close", "open", "high", "low", "close", "settle"]
            if any(type(row[k]) not in (int, float) or not math.isfinite(row[k]) or row[k] <= 0 for k in prices):
                raise ValueError("FUTURES_PRICE_INVALID")
            if api == "fut_daily":
                if not row["low"] <= min(row["open"], row["close"]) <= max(row["open"], row["close"]) <= row["high"]:
                    raise ValueError("FUTURES_OHLC_INVALID")
                if any(
                    type(row[k]) not in (int, float) or not math.isfinite(row[k]) or row[k] <= 0 for k in ("vol", "oi")
                ):
                    raise ValueError("FUTURES_LIQUIDITY_INVALID")
        result[str(parsed)] = row
    return dict(sorted(result.items()))
