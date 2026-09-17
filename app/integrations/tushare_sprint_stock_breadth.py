"""沪深日线返回集合的涨跌分布；固定交易所范围，保留原文全量计数和排除原因。"""

import hashlib
import json
import math
import re
from datetime import datetime

import numpy as np

FIELDS = ["ts_code", "trade_date", "close", "pre_close", "pct_chg", "vol", "amount"]
MAX_BYTES, ROW_LIMIT = 4_194_304, 6000


def parse(raw, day, minimum_included=3000):
    """严格核验同一天返回数据，再计算固定SH/SZ范围内的广度、中位涨幅与四分位距。

    pct_chg沿用供应商除权参考昨收口径，不跨日自行拼接未复权close；每行以
    close/pre_close交叉核算，容差0.00011个百分点覆盖四位小数舍入。BJ原文
    保存并计数，但其历史身份口径未核验，因此在任何日期一律不进入研究范围。
    少于行数上限不证明全市场完整；停牌无记录，分母仅为当日实际返回的沪深股票。
    """
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("STOCK_BREADTH_RAW_LIMIT")
    datetime.strptime(day, "%Y-%m-%d")
    value = json.loads(raw)
    if type(value.get("code")) is not int or value["code"] != 0:
        raise ValueError("STOCK_BREADTH_PROVIDER_REJECTED")
    data = value.get("data") or {}
    rows = data.get("items")
    if data.get("fields") != FIELDS or not isinstance(rows, list) or not 0 < len(rows) < ROW_LIMIT:
        raise ValueError("STOCK_BREADTH_SCHEMA_EMPTY_OR_TRUNCATED")
    seen, included, excluded = set(), [], []
    maximum_error = 0.0
    for item in rows:
        if not isinstance(item, list) or len(item) != len(FIELDS):
            raise ValueError("STOCK_BREADTH_ROW_SHAPE")
        row = dict(zip(FIELDS, item, strict=True))
        code = row["ts_code"]
        if not isinstance(code, str) or not re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", code) or code in seen:
            raise ValueError("STOCK_BREADTH_CODE_OR_DUPLICATE")
        if row["trade_date"] != day.replace("-", ""):
            raise ValueError("STOCK_BREADTH_DATE_CHANGED")
        seen.add(code)
        if any(type(row[k]) not in (int, float) or not math.isfinite(row[k]) for k in FIELDS[2:]):
            raise ValueError("STOCK_BREADTH_NUMERIC_INVALID")
        if row["close"] <= 0 or row["pre_close"] <= 0 or row["vol"] < 0 or row["amount"] < 0:
            raise ValueError("STOCK_BREADTH_PRICE_OR_QUANTITY_INVALID")
        error = abs(100 * (row["close"] / row["pre_close"] - 1) - row["pct_chg"])
        if error > 0.00011:
            raise ValueError("STOCK_BREADTH_PCT_FORMULA_CHANGED")
        maximum_error = max(maximum_error, error)
        (excluded if code.endswith(".BJ") else included).append(row)
    if len(included) < minimum_included:
        raise ValueError("STOCK_BREADTH_INSUFFICIENT_RETURNED_UNIVERSE")
    returns = np.asarray([r["pct_chg"] for r in included])
    up, down, flat = (int(np.sum(returns > 0)), int(np.sum(returns < 0)), int(np.sum(returns == 0)))
    q25, median, q75 = np.quantile(returns, [0.25, 0.5, 0.75], method="linear")
    return {
        "date": day,
        "available": True,
        "returned_rows": len(rows),
        "included_rows": len(included),
        "excluded_bj_rows": len(excluded),
        "up": up,
        "down": down,
        "flat": flat,
        "breadth": (up - down) / len(included),
        "median_pct": float(np.clip(median, -20, 20)),
        "iqr_pct": float(np.clip(q75 - q25, 0, 40)),
        "raw_median_pct": float(median),
        "raw_iqr_pct": float(q75 - q25),
        "max_pct_formula_error": maximum_error,
        "included_set_sha256": hashlib.sha256("\n".join(sorted(r["ts_code"] for r in included)).encode()).hexdigest(),
        "excluded_set_sha256": hashlib.sha256("\n".join(sorted(r["ts_code"] for r in excluded)).encode()).hexdigest(),
        "scope": "PROVIDER_RETURNED_SH_SZ_DAILY_QUOTES",
        "all_listed_stocks_independently_verified": False,
        "historical_first_publication_verified": False,
    }
