"""独立核验明确写成“亏损：正数”的金额，不修改已冻结的旧金额核验器。"""

import re
from decimal import Decimal

from app.services.fund_earnings_evidence_v1 import review_claim
from app.services.fund_information_history_v1 import normalize


def review_loss_claim(spec, pages):
    """将原文明确的亏损幅度转为有符号金额，保留原值及完整页锚点。

    仅接收一个明确的“亏损：金额”或“亏损：金额—金额”片段；金额必须
    非负、每项都有同一个单位，且与字段逐项一致。不能从同比下降、风险
    提示或同页其他指标推断亏损。原件金额、单位和期间仍由旧核验器检查。
    返回 money 按负数从小到大排列，用于同口径比较；reported_magnitudes
    保留正数幅度和原顺序，避免下游误把盈利字段当成亏损幅度。
    """
    anchor = normalize(spec.get("loss_anchor", ""))
    quote = normalize(spec.get("quote", ""))
    if not anchor or quote.count(anchor) != 1:
        raise ValueError("LOSS_ANCHOR_NOT_UNIQUE_IN_CLAIM")
    unit = spec.get("unit", "")
    if unit not in {"元", "万元", "百万元", "亿元"}:
        raise ValueError("LOSS_UNIT_NOT_SUPPORTED")
    number = r"(\d[\d,，]*(?:\.\d+)?)"
    token = number + re.escape(unit)
    match = re.fullmatch(r"亏损[：:]" + token + r"(?:[—–~～至到]" + token + r")?", anchor)
    if not match:
        raise ValueError("EXPLICIT_LOSS_MAGNITUDE_REQUIRED")
    literals = [v for v in match.groups() if v is not None]
    magnitudes = [Decimal(v.replace(",", "").replace("，", "")) for v in literals]
    wanted = [Decimal(str(v).replace(",", "").replace("，", "")) for v in spec.get("values", [])]
    if any(not v.is_finite() or v < 0 for v in wanted) or magnitudes != wanted:
        raise ValueError("LOSS_MAGNITUDES_MISMATCH")
    claim = review_claim(spec, pages)
    reported = claim["money"]
    signed = []
    for entry in reversed(reported):
        # 同时明确金额字段和折算后 CNY 的符号，原输入不作原地改写。
        signed.append(
            {
                **entry,
                "value": str(-Decimal(entry["value"])),
                "cny": str(-Decimal(entry["cny"])),
                "sign_source": "EXPLICIT_LOSS_WORD",
            }
        )
    return {
        **claim,
        "reported_magnitudes": reported,
        "money": signed,
        "polarity": "LOSS",
        "signed_interval_cny": [signed[0]["cny"], signed[-1]["cny"]],
        "polarity_anchor": {"page": spec["page"], "text": anchor},
        "training_ready": False,
    }
