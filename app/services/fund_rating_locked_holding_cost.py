"""从本份额原件重算最低三年持有情景，不能当成365天可赎回费用。"""

import re
from decimal import Decimal, localcontext

from app.services.fund_rating_market_review import annual_fee_score, normalized_text

MODE = "MINIMUM_THREE_CALENDAR_YEARS_ANNUAL_AVERAGE_COST"
TERMS = (
    "本基金设置投资者最短持有期限(即锁定期)为三年",
    "在锁定期内基金份额持有人不能提出赎回申请",
    "锁定期届满后的下一个工作日起可以提出赎回申请",
    "本基金设置三年的锁定期限,锁定期限后方可赎回,赎回时不收取赎回费",
)


def calculate(summary: str, contract: str, *, code: str, legal_name: str, share_class: str) -> dict:
    """只接受已核版式和身份；新类型、新份额不得自动套用本情景。"""
    summary, contract = normalized_text(summary), normalized_text(contract)
    if (
        code != "013853"
        or share_class != "A"
        or code not in summary
        or normalized_text(legal_name) != "大成匠心卓越三年持有期混合型证券投资基金"
        or normalized_text(legal_name) not in summary
    ):
        raise ValueError("LOCKED_COST_IDENTITY")
    if not all(term in contract for term in TERMS) or TERMS[-1] not in summary:
        raise ValueError("LOCKED_COST_TERMS_MISSING")
    sections = summary.split("三、投资本基金涉及的费用")
    if len(sections) != 2:
        raise ValueError("LOCKED_COST_FEE_SECTION")
    fees = sections[1]
    buy = re.findall(r"(?<![<≤=])M<100万元(\d+(?:\.\d+)?)%", fees)
    operating = re.findall(r"大成匠心卓越三年持有混合A基金运作综合费率\(年化\)(\d+(?:\.\d+)?)%", fees)
    trading = fees.split("(二)基金运作相关费用", 1)[0]
    # 原表第一列跨行，“申购费”和“前收费”之间会夹入第二列的金额档。
    # 只在同一交易费用小节核两个明确标签，金额档仍须唯一。
    if len(buy) != 1 or len(operating) != 1 or not all(term in trading for term in ("申购费", "(前收费)")):
        raise ValueError("LOCKED_COST_RATES_NOT_UNIQUE")
    with localcontext() as context:
        context.prec = 60
        subscription, rate = Decimal(buy[0]) / 100, Decimal(operating[0]) / 100
        if not 0 <= subscription <= 1 or not 0 <= rate < 1:
            raise ValueError("LOCKED_COST_RATE_RANGE")
        # 三年按同一已披露年化费率静态估计，综合运作费不再叠加管理费和托管费。
        # 复用金额算法时传入三年的累计运作成本；这里不使用该算法的一年分数。
        amounts = annual_fee_score(
            {
                "subscription": str(subscription),
                "redemption": "0",
                "operating_cost": str(1 - (1 - rate) ** 3),
                "subscription_mode": "EXTERNAL",
            }
        )
        annual_cost = Decimal(amounts["cost_cny"]) / 3
        score = max(Decimal(0), min(Decimal(100), 100 - annual_cost / 4))
    return {
        "mode": MODE,
        "subscription_rate": str(subscription),
        "annual_operating_rate": str(rate),
        "redemption_rate_after_lock": "0",
        "minimum_calendar_years": 3,
        "redeemable_at_365_days": False,
        "first_redemption_rule": TERMS[2],
        "total_cost_per_10000": amounts["cost_cny"],
        "annual_cost_per_10000": str(annual_cost),
        "score": str(score),
        "verified_terms": list(TERMS),
    }


def replay(inputs: dict, source) -> dict:
    """正式回放重新读原件；冻结费用、锁定条件或分数被改动均拒绝。"""
    reviewed, refs = inputs["reviewed_cost"], inputs["target_refs"]
    scenario = reviewed["scenario"]
    if reviewed["source_hash"] != refs["summary"] or scenario.get("mode") != MODE:
        raise ValueError("LOCKED_COST_SCENARIO_BINDING")
    result = calculate(
        "\n".join(source.pages(refs["summary"])),
        "\n".join(source.pages(refs["prospectus"])),
        code=inputs["fund_code"],
        legal_name=inputs["legal_name"],
        share_class=inputs["share_class"],
    )
    if scenario.get("redeemable_at_365_days") is not False:
        raise ValueError("LOCKED_COST_REDEMPTION_CONFLICT")
    for key in ("mode", "minimum_calendar_years", "first_redemption_rule", "verified_terms"):
        if scenario.get(key) != result[key]:
            raise ValueError("LOCKED_COST_TERMS_CONFLICT")
    for key in (
        "subscription_rate",
        "annual_operating_rate",
        "redemption_rate_after_lock",
        "total_cost_per_10000",
        "annual_cost_per_10000",
        "score",
    ):
        if Decimal(str(scenario.get(key))) != Decimal(result[key]):
            raise ValueError("LOCKED_COST_FROZEN_CONFLICT")
    for key in ("annual_cost_per_10000", "score"):
        if Decimal(str(reviewed.get(key))) != Decimal(result[key]):
            raise ValueError("LOCKED_COST_REVIEWED_CONFLICT")
    return result
