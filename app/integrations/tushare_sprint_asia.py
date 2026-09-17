"""一日研究的日经历史开盘数据校验；日线存在不代表盘中已公布。"""

import json
import math
from datetime import datetime

FIELDS = ["ts_code", "trade_date", "open", "close", "pre_close"]


def parse_nikkei_history(raw, start, end):
    """只接受一年内不超过300行的N225原始价格；空、重复、越界和无效数值均拒绝。

    不推断缺少的日期一定休市，也不填写缺失值。返回的价格单位为指数点，
    opening_gap_pct为百分比；first_publication_verified恒为False。
    """
    if not isinstance(raw, bytes) or not 0 < len(raw) <= 500000:
        raise ValueError("NIKKEI_HISTORY_BYTES_INVALID")
    value = json.loads(raw)
    if type(value.get("code")) is not int or value["code"] != 0:
        raise ValueError("NIKKEI_HISTORY_PROVIDER_REJECTED")
    data = value.get("data") or {}
    if data.get("fields") != FIELDS or not isinstance(data.get("items"), list) or not 0 < len(data["items"]) <= 300:
        raise ValueError("NIKKEI_HISTORY_FIELDS_OR_COUNT_INVALID")
    rows = {}
    for row in data["items"]:
        if not isinstance(row, list) or len(row) != len(FIELDS):
            raise ValueError("NIKKEI_HISTORY_ROW_INVALID")
        code, stamp, opened, closed, previous = row
        if code != "N225" or not isinstance(stamp, str) or len(stamp) != 8 or not start <= stamp <= end:
            raise ValueError("NIKKEI_HISTORY_IDENTITY_OR_RANGE_INVALID")
        day = datetime.strptime(stamp, "%Y%m%d").date()
        if day.weekday() >= 5 or str(day) in rows:
            raise ValueError("NIKKEI_HISTORY_DUPLICATE_OR_WEEKEND")
        if not all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in (opened, closed, previous)):
            raise ValueError("NIKKEI_HISTORY_PRICE_INVALID")
        rows[str(day)] = {
            "open": opened,
            "close": closed,
            "pre_close": previous,
            "opening_gap_pct": 100 * (opened / previous - 1),
            "first_publication_verified": False,
        }
    return dict(sorted(rows.items()))
