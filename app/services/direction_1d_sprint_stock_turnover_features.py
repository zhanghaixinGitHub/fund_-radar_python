"""同一日返回股票的成交额权重方向与涨幅；只派生特征，不读取基金标签或请求供应商。"""

import json
import math

from app.integrations import tushare_sprint_stock_breadth_v2 as source
from app.services import direction_1d_sprint as base

KEYS = ("turnover_breadth", "turnover_return_pct")


def parse(raw, day, minimum_included=3000):
    """权重为当日amount/总amount；金额千元归一后抵消，不解释为资金流向。

    先复用冻结解析器核验全部SH/SZ原始记录；固定排除BJ，平盘金额仍在分母，
    已返回的零成交额行不删除。总额为零时明确缺失，不产生虚假的中性信号。
    对代码排序后用fsum，避免供应商改变记录顺序造成同一来源的摘要漂移。
    """
    validated = source.parse(raw, day, minimum_included=minimum_included)
    value = json.loads(raw)
    rows = sorted(
        (dict(zip(source.FIELDS, r, strict=True)) for r in value["data"]["items"] if r[0].endswith((".SH", ".SZ"))),
        key=lambda r: r["ts_code"],
    )
    result = {
        "date": day,
        "scope": "PROVIDER_RETURNED_SH_SZ_DAILY_QUOTES",
        "source_distribution_hash": base.digest(validated),
        "included_rows": len(rows),
        "included_set_sha256": validated["included_set_sha256"],
        "amount_unit": "THOUSAND_CNY",
        "zero_amount_rows": sum(r["amount"] == 0 for r in rows),
        "all_listed_stocks_independently_verified": False,
        "historical_first_publication_verified": False,
    }
    try:
        total = math.fsum(r["amount"] for r in rows)
        up = math.fsum(r["amount"] for r in rows if r["pct_chg"] > 0)
        down = math.fsum(r["amount"] for r in rows if r["pct_chg"] < 0)
        flat = math.fsum(r["amount"] for r in rows if r["pct_chg"] == 0)
    except OverflowError:
        raise ValueError("STOCK_TURNOVER_AMOUNT_OVERFLOW") from None
    if not all(math.isfinite(x) for x in (total, up, down, flat)):
        raise ValueError("STOCK_TURNOVER_AMOUNT_NONFINITE")
    result.update(total_amount=total, up_amount=up, down_amount=down, flat_amount=flat)
    if total == 0:
        return result | {
            "available": False,
            "unavailable_reason": "ZERO_TOTAL_RETURNED_AMOUNT",
            "turnover_breadth": None,
            "turnover_return_pct": None,
            "raw_turnover_return_pct": None,
        }
    # 先归一化再乘涨幅，避免两个大数先相乘溢出；仅最终均值截断，不删极端涨幅股票。
    breadth = (up - down) / total
    raw_return = math.fsum(r["amount"] / total * r["pct_chg"] for r in rows)
    if not math.isfinite(raw_return) or not -1 <= breadth <= 1:
        raise ValueError("STOCK_TURNOVER_FEATURE_NONFINITE_OR_RANGE")
    return result | {
        "available": True,
        "unavailable_reason": None,
        "turnover_breadth": breadth,
        "turnover_return_pct": max(-20.0, min(20.0, raw_return)),
        "raw_turnover_return_pct": raw_return,
    }
