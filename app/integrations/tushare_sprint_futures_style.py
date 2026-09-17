"""IH/IC原始主力日行情与具体合约映射，日期/品种/交易单位分别验证。"""

import json
import math
import re
from datetime import datetime

FIELDS = {
    "fut_daily": ["ts_code", "trade_date", "pre_close", "open", "high", "low", "close", "settle", "vol", "oi"],
    "fut_mapping": ["ts_code", "trade_date", "mapping_ts_code"],
}


def require(condition, reason):
    if not condition:
        raise ValueError("FUTURES_STYLE_" + reason)


def parse(raw, api, family, expected_dates):
    """每次一个品种、一个自然年、最多300行；缺失不能插值，连续别名不能替代合约映射。"""
    require(api in FIELDS and family in ("IH", "IC") and raw and len(raw) <= 1048576, "REQUEST_SCOPE_INVALID")
    require(
        1 <= len(expected_dates) <= 300 and len(set(expected_dates)) == len(expected_dates), "EXPECTED_DATES_INVALID"
    )
    value = json.loads(raw)
    require(type(value.get("code")) is int and value["code"] == 0, "PROVIDER_REJECTED")
    data = value.get("data") or {}
    require(
        data.get("fields") == FIELDS[api] and isinstance(data.get("items"), list) and 1 <= len(data["items"]) <= 300,
        "SCHEMA_INVALID",
    )
    rows = {}
    for item in data["items"]:
        require(isinstance(item, list) and len(item) == len(FIELDS[api]), "ROW_INVALID")
        row = dict(zip(FIELDS[api], item, strict=True))
        require(
            row["ts_code"] == family + ".CFX"
            and isinstance(row["trade_date"], str)
            and re.fullmatch(r"\d{8}", row["trade_date"]) is not None,
            "ROW_SCOPE_INVALID",
        )
        day = str(datetime.strptime(row["trade_date"], "%Y%m%d").date())
        require(day in expected_dates and day not in rows, "DATE_SCOPE_OR_DUPLICATE")
        if api == "fut_mapping":
            require(
                isinstance(row["mapping_ts_code"], str)
                and re.fullmatch(family + r"\d{4}\.CFX", row["mapping_ts_code"]) is not None,
                "MAPPING_INVALID",
            )
        else:
            require(
                all(
                    type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] > 0
                    for k in ("pre_close", "open", "high", "low", "close", "settle", "vol", "oi")
                ),
                "PRICE_OR_LIQUIDITY_INVALID",
            )
            require(
                row["low"] <= min(row["open"], row["close"]) <= max(row["open"], row["close"]) <= row["high"],
                "OHLC_INVALID",
            )
        rows[day] = row
    require(set(rows) == set(expected_dates), "MISSING_DATES")
    return dict(sorted(rows.items()))
