"""其他类型的八维原始指标适配；全部仍属候选，必须有本类完整资料和独立试算才可发布。

信用/久期使用统一压力情景；指数按同目标总回报比较；货币用实际分配收益；FOF 穿透。
本模块不接受预先填写的评分，也不会在专用字段缺失时改套主动权益指标。
"""

from decimal import Decimal
from statistics import median

from app.services.fund_rating_rules import decimal


def ratio(value):
    value = decimal(value)
    if not 0 <= value <= 1:
        raise ValueError("RATING_RATIO_RANGE")
    return value


def distribution_index(history: dict) -> list:
    """货币基金逐估值区间的实际每单位分配额再投资，不使用恒定单位净值或七日年化替代。"""
    distributions = history["distribution_per_unit"]
    bases = history["distribution_base_unit"]
    if len(distributions) != len(history["dates"]) - 1 or len(distributions) != len(bases):
        raise ValueError("RATING_DISTRIBUTION_ALIGNMENT")
    result = [Decimal(1)]
    for amount, base in zip(distributions, bases, strict=True):
        a, b = decimal(amount), decimal(base)
        if b <= 0 or 1 + a / b <= 0:
            raise ValueError("RATING_DISTRIBUTION_RANGE")
        result.append(result[-1] * (1 + a / b))
    return result


def _weights(values: dict) -> dict:
    result = {k: ratio(v) for k, v in values.items()}
    if not result or abs(sum(result.values()) - 1) > Decimal("0.00000001"):
        raise ValueError("RATING_WEIGHTS_NOT_RECONCILED")
    return result


def distance(actual: dict, target: dict) -> Decimal:
    """完整资产权重向量的半数绝对偏离；不能只比较已取得的交集。"""
    a, b = _weights(actual), _weights(target)
    return sum(abs(a.get(k, 0) - b.get(k, 0)) for k in a.keys() | b.keys()) / 2


def specialized_holdings(family: str, holdings: dict, extra: dict) -> dict:
    """完整资产逐元对账，债券用真实可交易额、货币用到期安排、FOF 用底层赎回安排。"""
    net, cash, payable = (decimal(holdings[k]) for k in ("net_assets", "cash", "payable_5d"))
    positions = holdings["positions"]
    if net <= 0 or min(cash, payable) < 0 or not positions:
        raise ValueError("RATING_ASSETS_INVALID")
    amounts = [decimal(p["amount"]) for p in positions]
    if any(v <= 0 for v in amounts) or abs(sum(amounts) + cash - payable - net) > Decimal(".01"):
        raise ValueError("RATING_ASSETS_NOT_RECONCILED")
    total = sum(amounts)
    issuers = {}
    liquid = cash - payable
    credit, duration, lower_credit, equity, premium = (Decimal(0) for _ in range(5))
    for p, amount in zip(positions, amounts, strict=True):
        if not p["issuer"]:
            raise ValueError("RATING_ISSUER_MISSING")
        issuers[p["issuer"]] = issuers.get(p["issuer"], 0) + amount
        weight = amount / net
        cap = decimal(p["saleable_amount"])
        if not 0 <= cap <= amount:
            raise ValueError("RATING_SALEABLE_RANGE")
        if family == "MONEY":
            liquidity = min(cap, decimal(p["maturing_or_redeemable_5d"]))
        elif family == "FOF":
            liquidity = min(cap, decimal(p["redeemable_5d"]))
        else:
            turnover = [decimal(v) for v in p["turnover_20d"]]
            if len(turnover) != 20 or any(v < 0 for v in turnover):
                raise ValueError("RATING_TURNOVER_MISSING")
            liquidity = min(cap, Decimal(".5") * median(turnover))
        if not 0 <= liquidity <= amount:
            raise ValueError("RATING_LIQUIDITY_RANGE")
        liquid += liquidity
        if family in {"BOND", "SHORT_BOND", "MIXED_BOND", "QDII_BOND", "MONEY"}:
            # 明确的信用质量分层与来源损失参数；未披露绝不按最高信用或零损失填充。
            credit += weight * ratio(p["stress_default_probability"]) * ratio(p["loss_given_default"])
            d, convexity = decimal(p["modified_duration"]), decimal(p["convexity"])
            if d < 0 or convexity < 0 or not isinstance(p["lower_credit"], bool):
                raise ValueError("RATING_CREDIT_DURATION_INVALID")
            duration += weight * max(Decimal(0), d * Decimal(".01") - convexity * Decimal(".00005"))
            lower_credit += weight if p["lower_credit"] else 0
        if family == "CONVERTIBLE":
            equity += weight * ratio(p["equity_delta"]) * Decimal(".3")
            premium += weight * decimal(p["conversion_premium"])
    coverage = min(Decimal(1), max(Decimal(0), liquid / net))
    output = {
        "issuer_hhi": sum((v / total) ** 2 for v in issuers.values()),
        "largest_holder": ratio(holdings["largest_holder"]),
    }
    if family in {"BOND", "SHORT_BOND", "MIXED_BOND", "QDII_BOND", "MONEY"}:
        output.update(
            credit_loss_stress=credit,
            duration_stress=duration,
            lower_credit_ratio=lower_credit,
            bond_liquid_5d=coverage,
            maturity_liquid_5d=coverage,
        )
    elif family == "CONVERTIBLE":
        output.update(equity_stress=equity, conversion_premium=premium, convertible_liquid_5d=coverage)
    elif family == "FOF":
        lookthrough = {}
        overlap = {}
        cost = Decimal(0)
        for p, amount in zip(positions, amounts, strict=True):
            w = amount / total
            weights = _weights(p["underlying_weights"])
            for issuer, fraction in weights.items():
                value = w * fraction
                lookthrough[issuer] = lookthrough.get(issuer, 0) + value
                overlap[issuer] = max(overlap.get(issuer, 0), value)
            # 底层费用净额必须扣除文件证实的减免，并确认尚未计入外层标准成本。
            fee, waived = ratio(p["underlying_cost_1y"]), ratio(p["waived_cost_1y"])
            if waived > fee or p["cost_already_in_outer"] is not False:
                raise ValueError("RATING_FOF_COST_DOUBLE_COUNT")
            cost += w * (fee - waived)
        output.update(
            lookthrough_hhi=sum(v**2 for v in lookthrough.values()),
            overlap_ratio=1 - sum(overlap.values()),
            underlying_redeemable_5d=coverage,
            underlying_cost=cost,
        )
    elif family == "QDII_COMMODITY":
        output.update(
            counterparty_hhi=output["issuer_hhi"],
            roll_cost=decimal(extra["roll_cost_amount"]) / net,
            overseas_liquid_5d=coverage,
        )
    elif family == "INDEX":
        output.update(
            replication_deviation=distance(extra["actual_index_weights"], extra["target_index_weights"]),
            liquid_5d=coverage,
        )
    return output


def tracking(index: list, benchmark: list, annual_days: int) -> dict:
    """同币种同日期总回报跟踪误差；跟踪偏离取日均超额年化绝对值，不奖励主动跑赢。"""
    if len(index) != len(benchmark) or len(index) < 3:
        raise ValueError("RATING_BENCHMARK_ALIGNMENT")
    benchmark = [decimal(x) for x in benchmark]
    if any(x <= 0 for x in benchmark):
        raise ValueError("RATING_BENCHMARK_RANGE")
    differences = [index[i] / index[i - 1] - benchmark[i] / benchmark[i - 1] for i in range(1, len(index))]
    mean = sum(differences) / len(differences)
    variance = sum((x - mean) ** 2 for x in differences) / (len(differences) - 1)
    return {"tracking_error": (variance * annual_days).sqrt(), "tracking_bias": abs(mean * annual_days)}
