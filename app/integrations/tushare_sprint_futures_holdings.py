"""Tushare 公开期货会员排名：按真实具体合约与交易日解析，不把榜外空值补零。"""

import json
import math
from datetime import datetime

FIELDS = (
    "trade_date",
    "symbol",
    "broker",
    "vol",
    "vol_chg",
    "long_hld",
    "long_chg",
    "short_hld",
    "short_chg",
    "exchange",
)


def require(condition, reason):
    if not condition:
        raise ValueError("FUTURES_HOLDINGS_" + reason)


def parse(raw, symbol, dates):
    """返回每个预定日的完整排名及两个无量纲特征；任何日期/字段/排名不完整则整批拒绝。

    持仓和变化单位均为手。多头与空头各自独立排名，榜外None保持缺失；求和仅涵盖
    各自20个已公布排名值，不代表全市场净额或期货公司自营仓位。变化直接使用同一
    具体合约的供应商变化值，避免自行跨主力换月做差。
    """
    value = json.loads(raw)
    require(type(value.get("code")) is int and value["code"] == 0, "PROVIDER_REJECTED")
    data = value.get("data") or {}
    require(data.get("fields") == list(FIELDS) and isinstance(data.get("items"), list), "FIELDS_INVALID")
    require(0 < len(data["items"]) < 2000, "EMPTY_OR_POSSIBLY_TRUNCATED")
    require(
        isinstance(symbol, str) and len(symbol) == 6 and symbol.startswith("IF") and symbol[2:].isdigit(),
        "SYMBOL_INVALID",
    )
    require(bool(dates) and len(dates) == len(set(dates)), "DATES_INVALID")
    days = {day: [] for day in dates}
    seen = set()
    for item in data["items"]:
        require(isinstance(item, list) and len(item) == len(FIELDS), "ROW_WIDTH")
        row = dict(zip(FIELDS, item, strict=True))
        raw_day = row["trade_date"]
        require(isinstance(raw_day, str) and len(raw_day) == 8, "DATE_INVALID")
        day = str(datetime.strptime(raw_day, "%Y%m%d").date())
        require(day in days and row["symbol"] == symbol and row["exchange"] == "CFFEX", "SOURCE_IDENTITY")
        broker = row["broker"]
        require(isinstance(broker, str) and broker.strip() == broker and bool(broker), "BROKER_INVALID")
        require(not any(word in broker for word in ("合计", "总计")), "AGGREGATE_ROW")
        key = day, broker
        require(key not in seen, "DUPLICATE_MEMBER")
        seen.add(key)
        for name in FIELDS[3:9]:
            number = row[name]
            if number is None:
                continue
            require(type(number) in (int, float) and math.isfinite(number) and number == int(number), "NUMBER_INVALID")
            if name in ("vol", "long_hld", "short_hld"):
                require(number >= 0, "NEGATIVE_POSITION_OR_VOLUME")
        days[day].append(row)
    result = {}
    for day, rows in days.items():
        longs = [row for row in rows if row["long_hld"] is not None]
        shorts = [row for row in rows if row["short_hld"] is not None]
        require(len(longs) == len(shorts) == 20, "RANKING_COVERAGE")
        require(all(row["long_chg"] is not None for row in longs), "LONG_CHANGE_MISSING")
        require(all(row["short_chg"] is not None for row in shorts), "SHORT_CHANGE_MISSING")
        long_total = sum(row["long_hld"] for row in longs)
        short_total = sum(row["short_hld"] for row in shorts)
        total = long_total + short_total
        require(total > 0, "ZERO_TOTAL")
        long_change = sum(row["long_chg"] for row in longs)
        short_change = sum(row["short_chg"] for row in shorts)
        raw_change = (long_change - short_change) / total
        result[day] = {
            "main_contract": symbol + ".CFX",
            "ranked_long_contracts": long_total,
            "ranked_short_contracts": short_total,
            "ranked_long_change": long_change,
            "ranked_short_change": short_change,
            "level_imbalance": (long_total - short_total) / total,
            "change_imbalance": max(-1.0, min(1.0, raw_change)),
            "raw_change_imbalance": raw_change,
            "change_clipped": abs(raw_change) > 1.0,
            "long_members": 20,
            "short_members": 20,
            "rows": sorted(rows, key=lambda row: row["broker"]),
        }
    return result
