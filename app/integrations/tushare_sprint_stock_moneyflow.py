"""Tushare订单分类金额解析；保留供应商净额定义，不把买卖总和差冒充净流入。"""

import hashlib
import json
import math
import re
from datetime import datetime

FIELDS = (
    ["ts_code", "trade_date"]
    + [
        side + "_" + size + "_" + unit
        for size in ["sm", "md", "lg", "elg"]
        for side in ["buy", "sell"]
        for unit in ["vol", "amount"]
    ]
    + ["net_mf_vol", "net_mf_amount"]
)
MAX_BYTES, ROW_LIMIT = 4_194_304, 6000


def parse(raw, day, minimum_included=3000):
    """金额单位万元、量单位手；范围固定当日返回SH/SZ，BJ保留原文并排除。

    大单不平衡为大/特大单买卖差除双边金额和。净额比例的分母是各类买卖金额
    和的一半，仅为供应商分类金额尺度，不声称等于交易所成交额或真实资金净流入。
    全零分母显式不可用；部分股票零金额仍保留，净额不按买卖分类重新计算。
    """
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("MONEYFLOW_RAW_LIMIT")
    datetime.strptime(day, "%Y-%m-%d")
    value = json.loads(raw)
    if type(value.get("code")) is not int or value["code"] != 0:
        raise ValueError("MONEYFLOW_PROVIDER_REJECTED")
    data = value.get("data") or {}
    rows = data.get("items")
    if data.get("fields") != FIELDS or not isinstance(rows, list) or not 0 < len(rows) < ROW_LIMIT:
        raise ValueError("MONEYFLOW_SCHEMA_EMPTY_OR_TRUNCATED")
    seen, included, excluded = set(), [], []
    for item in rows:
        if not isinstance(item, list) or len(item) != len(FIELDS):
            raise ValueError("MONEYFLOW_ROW_SHAPE")
        row = dict(zip(FIELDS, item, strict=True))
        code = row["ts_code"]
        if not isinstance(code, str) or not re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", code) or code in seen:
            raise ValueError("MONEYFLOW_CODE_OR_DUPLICATE")
        if row["trade_date"] != day.replace("-", ""):
            raise ValueError("MONEYFLOW_DATE_CHANGED")
        seen.add(code)
        if code.endswith(".BJ"):
            excluded.append(code)
            continue
        if any(type(row[k]) not in (int, float) or not math.isfinite(row[k]) for k in FIELDS[2:]):
            raise ValueError("MONEYFLOW_NUMERIC_INVALID")
        if any(row[k] < 0 for k in FIELDS[2:-2]):
            raise ValueError("MONEYFLOW_NEGATIVE_CLASSIFIED_AMOUNT_OR_VOLUME")
        included.append(row)
    if len(included) < minimum_included:
        raise ValueError("MONEYFLOW_INSUFFICIENT_RETURNED_UNIVERSE")
    included.sort(key=lambda row: row["ts_code"])
    totals = {k: math.fsum(r[k] for r in included) for k in FIELDS[2:]}
    if not all(math.isfinite(x) for x in totals.values()):
        raise ValueError("MONEYFLOW_AGGREGATE_NONFINITE")
    large_buy = totals["buy_lg_amount"] + totals["buy_elg_amount"]
    large_sell = totals["sell_lg_amount"] + totals["sell_elg_amount"]
    gross = math.fsum(totals[k] for k in FIELDS[2:-2] if k.endswith("amount")) / 2
    large_gross = large_buy + large_sell
    available = gross > 0 and large_gross > 0
    net_fraction = totals["net_mf_amount"] / gross if gross > 0 else None
    if not all(math.isfinite(x) for x in (gross, large_gross)) or (
        net_fraction is not None and not math.isfinite(net_fraction)
    ):
        raise ValueError("MONEYFLOW_DERIVED_NONFINITE")
    return {
        "date": day,
        "available": available,
        "unavailable_reason": None if available else "ZERO_CLASSIFIED_DENOMINATOR",
        "scope": "PROVIDER_RETURNED_SH_SZ_MONEYFLOW",
        "amount_unit": "TEN_THOUSAND_CNY",
        "volume_unit": "LOT_100_SHARES",
        "returned_rows": len(rows),
        "included_rows": len(included),
        "excluded_bj_rows": len(excluded),
        "included_set_sha256": hashlib.sha256("\n".join(r["ts_code"] for r in included).encode()).hexdigest(),
        "totals": totals,
        "classified_gross_amount": gross,
        "large_classified_gross_amount": large_gross,
        "large_imbalance": (large_buy - large_sell) / large_gross if available else None,
        "raw_net_fraction": net_fraction,
        "net_fraction": max(-1.0, min(1.0, net_fraction)) if available else None,
        "net_fraction_clipped": available and abs(net_fraction) > 1,
        "full_registry_verified": False,
        "historical_first_publication_verified": False,
    }
