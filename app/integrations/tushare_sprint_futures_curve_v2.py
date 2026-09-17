"""一日研究的具体IF合约价格与期限价差；到期日允许零持仓但仍须真实成交和有效价格。"""

import json
import math
import re
from datetime import date, datetime

FIELDS = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "oi"]


def require(condition, reason):
    if not condition:
        raise ValueError("FUTURES_CURVE_" + reason)


def parse(raw, contract, expected_dates, expiry_date):
    """单个真实合约、最多80个明确交易日；缺日/重复/连续合约替代均拒绝。"""
    require(re.fullmatch(r"IF\d{4}\.CFX", contract) is not None, "CONTRACT_INVALID")
    require(raw and len(raw) <= 524288 and 1 <= len(expected_dates) <= 80, "REQUEST_BOUND_INVALID")
    expected = set(expected_dates)
    require(len(expected) == len(expected_dates), "EXPECTED_DATES_DUPLICATED")
    for day in expected:
        require(str(date.fromisoformat(day)) == day, "DATE_INVALID")
    value = json.loads(raw)
    require(type(value.get("code")) is int and value["code"] == 0, "PROVIDER_REJECTED")
    data = value.get("data") or {}
    require(
        data.get("fields") == FIELDS and isinstance(data.get("items"), list) and 1 <= len(data["items"]) <= 80,
        "SCHEMA_OR_COUNT_INVALID",
    )
    rows = {}
    for item in data["items"]:
        require(isinstance(item, list) and len(item) == len(FIELDS), "ROW_INVALID")
        row = dict(zip(FIELDS, item, strict=True))
        require(
            row["ts_code"] == contract
            and isinstance(row["trade_date"], str)
            and re.fullmatch(r"\d{8}", row["trade_date"]) is not None,
            "ROW_SCOPE_INVALID",
        )
        day = str(datetime.strptime(row["trade_date"], "%Y%m%d").date())
        require(day in expected and day not in rows, "DATE_SCOPE_OR_DUPLICATE")
        require(
            all(
                type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] > 0
                for k in ("open", "high", "low", "close", "vol")
            ),
            "PRICE_OR_LIQUIDITY_INVALID",
        )
        require(
            row["low"] <= min(row["open"], row["close"]) <= max(row["open"], row["close"]) <= row["high"],
            "OHLC_INVALID",
        )
        # 到期清算后持仓可以归零；非到期日仍要求正持仓，None和负数永远无效。
        require(
            type(row["oi"]) in (int, float)
            and math.isfinite(row["oi"])
            and (row["oi"] > 0 or row["oi"] == 0 and day == expiry_date),
            "OI_INVALID",
        )
        rows[day] = row
    require(set(rows) == expected, "MISSING_DATES")
    return dict(sorted(rows.items()))


def point(day, near_meta, far_meta, near_row, far_row):
    """按实际到期日间隔年化近远月价差；单位为百分点，固定截断±50，保留原值。"""
    dt = date.fromisoformat(day)
    expiries = []
    for meta, row in ((near_meta, near_row), (far_meta, far_row)):
        require(
            row["ts_code"] == meta["ts_code"] and row["trade_date"] == day.replace("-", ""), "PAIR_IDENTITY_INVALID"
        )
        require(
            meta["exchange"] == "CFFEX"
            and meta["fut_code"] == "IF"
            and meta["multiplier"] == 300
            and meta["quote_unit"] == "指数点",
            "CONTRACT_UNIT_INVALID",
        )
        listed = datetime.strptime(meta["list_date"], "%Y%m%d").date()
        expiry = datetime.strptime(meta["delist_date"], "%Y%m%d").date()
        require(listed <= dt <= expiry, "CONTRACT_NOT_ACTIVE")
        expiries.append(expiry)
    gap = (expiries[1] - expiries[0]).days
    require(gap > 0, "EXPIRY_ORDER_INVALID")
    raw = 100 * (far_row["close"] / near_row["close"] - 1) * 365 / gap
    require(math.isfinite(raw), "CURVE_NONFINITE")
    return {
        "date": day,
        "near_contract": near_meta["ts_code"],
        "far_contract": far_meta["ts_code"],
        "near_close": near_row["close"],
        "far_close": far_row["close"],
        "expiry_gap_days": gap,
        "raw_annualized_spread_pct": raw,
        "annualized_spread_pct": float(max(-50, min(50, raw))),
        "near_expiry_days": (expiries[0] - dt).days,
    }
