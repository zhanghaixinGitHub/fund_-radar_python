"""第三版同比核验：保留原引文的数值边界，规范化文本仅用于原文定位。"""

import re
from decimal import Decimal

from app.services.fund_information_history_v1 import normalize


def review_yoy(spec, pages):
    """审核原文已经给出的同比百分比，不从负基数推算，不混为环比或百分点。

    reported_values 保存文字原值；增长/下降单独保存。只有明确增长/下降且
    原值非负，才给出带方向值；裸数或负基数时不推断经济含义。
    """
    if spec["comparison"] != "YEAR_ON_YEAR" or spec["unit"] != "PERCENT":
        raise ValueError("NOT_YEAR_ON_YEAR_PERCENT")
    if not spec.get("metric") or not spec.get("basis") or not spec.get("period_start") or not spec.get("period_end"):
        raise ValueError("YOY_BASIS_MISSING")
    if spec.get("direction_word") not in {"增长", "下降", "原文有符号数值"}:
        raise ValueError("YOY_DIRECTION_UNKNOWN")
    if spec.get("base_state") not in {"POSITIVE", "NEGATIVE", "ZERO", "UNKNOWN"}:
        raise ValueError("YOY_BASE_STATE_MISSING")
    page = pages[spec["page"] - 1]
    quote = normalize(spec["quote"])
    if not quote or normalize(page).count(quote) != 1 or "%" not in quote:
        raise ValueError("YOY_ANCHOR_INVALID")
    if "百分点" in quote:
        raise ValueError("PERCENTAGE_POINT_NOT_PERCENT_CHANGE")
    # 数值直接锚定百分号，避免把盈利金额或相邻年份当作同比。
    # 原表格空白必须保留：金额末尾和下一同比单元格不能拼成新数字。
    literals = re.findall(r"(?<![\d.,])([+-]?\d+(?:\.\d+)?)\s*%", spec["quote"].replace("−", "-"))
    wanted = [Decimal(str(v)) for v in spec["values"]]
    if len(wanted) not in (1, 2) or any(not v.is_finite() for v in wanted):
        raise ValueError("INVALID_YOY_VALUES")
    # 区间分隔符 - 与负号在同一行可能重合：人工字段须以原文明确增长/下降解释，保留原值。
    available = [Decimal(v) for v in literals]
    for value in wanted:
        if value not in available:
            raise ValueError("YOY_VALUE_NOT_IN_ANCHOR")
    if spec["direction_word"] in {"增长", "下降"} and spec["direction_word"] not in quote:
        raise ValueError("YOY_DIRECTION_NOT_IN_ANCHOR")
    signed = None
    if spec["base_state"] == "POSITIVE" and all(v >= 0 for v in wanted):
        factor = Decimal(-1) if spec["direction_word"] == "下降" else Decimal(1)
        if spec["direction_word"] in {"增长", "下降"}:
            signed = [str(v * factor) for v in wanted]
    return {
        **spec,
        "reported_values": [str(v) for v in wanted],
        "signed_percent": signed,
        "anchor": {"page": spec["page"], "offset": normalize(page).index(quote), "text": quote},
        "derived_from_money": False,
        "training_ready": False,
    }
