"""基金级八维规则与确定性运算；候选参数不会因为代码上线自动成为正式规则。"""

from decimal import ROUND_HALF_EVEN, Decimal, localcontext

DIMENSIONS = {
    "return": "收益能力",
    "risk": "风险控制",
    "efficiency": "风险收益性价比",
    "stability": "业绩稳定性",
    "management": "管理与策略",
    "holdings": "持仓结构",
    "liquidity": "规模与流动性",
    "cost": "费用水平",
}
GRADES = {"WEAK": "偏弱", "AVERAGE": "一般", "GOOD": "良好", "EXCELLENT": "优质"}
PRECISION = Decimal("0.000000000001")
WEIGHTS = (20, 20, 15, 15, 10, 10, 5, 5)

# 子项为：规范化指标、方向、维度内权重。不同资产使用独立输入，绝不运行时回退权益公式。
EQUITY = {
    "return": [("return_1y", 1, 30), ("return_3y", 1, 70)],
    "risk": [("drawdown", -1, 70), ("downside", -1, 30)],
    "efficiency": [("sharpe", 1, 100)],
    "stability": [("stability", 1, 100)],
    "management": [("team_months", 1, 50), ("team_changes", -1, 50)],
    "holdings": [("issuer_hhi", -1, 50), ("industry_hhi", -1, 50)],
    "liquidity": [("liquid_5d", 1, 70), ("largest_holder", -1, 30)],
    "cost": [("cost_1y", -1, 100)],
}
CATEGORY_LABELS = {
    "ACTIVE_EQUITY": "主动权益",
    "BOND": "纯债",
    "SHORT_BOND": "短债",
    "MIXED_BOND": "混合债券",
    "CONVERTIBLE": "可转债",
    "INDEX": "被动指数",
    "MONEY": "货币基金",
    "QDII_EQUITY": "境外权益",
    "QDII_BOND": "境外债券",
    "QDII_COMMODITY": "境外商品",
    "FOF": "基金中基金",
    "OTHER": "其他类型",
}


def candidate(family: str) -> dict:
    """返回完整候选配置，调用方必须通过独立的真实资料试算验收后才可激活。

    非权益配置明确采用信用、跟踪、穿透或分配收益；没有相应输入就阻断本类。
    年化因子从已核验估值日历取得，252 不跨市场复用。
    """
    if family not in CATEGORY_LABELS or family == "OTHER":
        raise ValueError("RATING_FAMILY_UNSUPPORTED")
    formula = {k: list(v) for k, v in EQUITY.items()}
    if family in {"BOND", "SHORT_BOND", "MIXED_BOND", "QDII_BOND"}:
        formula["risk"] = [("drawdown", -1, 40), ("credit_loss_stress", -1, 30), ("duration_stress", -1, 30)]
        formula["holdings"] = [("issuer_hhi", -1, 50), ("lower_credit_ratio", -1, 50)]
        formula["liquidity"] = [("bond_liquid_5d", 1, 70), ("largest_holder", -1, 30)]
        if family == "MIXED_BOND":
            formula["management"] = [("allocation_deviation", -1, 50), ("team_changes", -1, 50)]
    elif family == "CONVERTIBLE":
        formula["risk"] = [("drawdown", -1, 50), ("equity_stress", -1, 50)]
        formula["holdings"] = [("issuer_hhi", -1, 50), ("conversion_premium", -1, 50)]
        formula["liquidity"] = [("convertible_liquid_5d", 1, 70), ("largest_holder", -1, 30)]
    elif family == "INDEX":
        formula["management"] = [("tracking_error", -1, 50), ("tracking_bias", -1, 50)]
        formula["holdings"] = [("replication_deviation", -1, 100)]
    elif family == "MONEY":
        formula["return"] = [("distribution_return_1y", 1, 100)]
        formula["risk"] = [("credit_loss_stress", -1, 50), ("duration_stress", -1, 50)]
        formula["efficiency"] = [("distribution_stress_ratio", 1, 100)]
        formula["stability"] = [("distribution_stability", 1, 100)]
        formula["holdings"] = [("issuer_hhi", -1, 50), ("lower_credit_ratio", -1, 50)]
        formula["liquidity"] = [("maturity_liquid_5d", 1, 70), ("largest_holder", -1, 30)]
    elif family == "FOF":
        formula["management"] = [("allocation_deviation", -1, 100)]
        formula["holdings"] = [("lookthrough_hhi", -1, 50), ("overlap_ratio", -1, 50)]
        formula["liquidity"] = [("underlying_redeemable_5d", 1, 70), ("largest_holder", -1, 30)]
        formula["cost"] = [("lookthrough_cost_1y", -1, 100)]
    elif family == "QDII_COMMODITY":
        formula["management"] = [("tracking_error", -1, 50), ("tracking_bias", -1, 50)]
        formula["holdings"] = [("counterparty_hhi", -1, 50), ("roll_cost", -1, 50)]
        formula["liquidity"] = [("overseas_liquid_5d", 1, 70), ("largest_holder", -1, 30)]
    if family == "QDII_EQUITY":
        formula["liquidity"] = [("overseas_liquid_5d", 1, 70), ("largest_holder", -1, 30)]
    weights = WEIGHTS
    if family in {"BOND", "SHORT_BOND", "MIXED_BOND", "QDII_BOND"}:
        weights = (15, 25, 15, 10, 10, 10, 10, 5)
    elif family == "MONEY":
        weights = (15, 25, 10, 10, 5, 10, 20, 5)
    elif family in {"INDEX", "QDII_COMMODITY"}:
        weights = (10, 15, 10, 10, 25, 10, 10, 10)
    elif family == "FOF":
        weights = (15, 20, 15, 10, 15, 10, 10, 5)
    return {
        "family": family,
        "revision": "candidate-2026-10-10",
        "weights": dict(zip(DIMENSIONS, weights, strict=True)),
        "formula": formula,
        "minimum_products": 30,
        "history_months": 36,
        "tie_precision": "0.000000000001",
        "rounding": "ROUND_HALF_EVEN",
        "grace_trading_days": 5,
        "holdings_age_days": 210,
        "verification_age_days": 31,
        "liquidity_days": 5,
        "participation": "0.1",
        "turnover_days": 20,
        "cost_principal": "10000",
        "cost_holding_days": 365,
        "status": "CANDIDATE",
    }


def decimal(value) -> Decimal:
    """拒绝空值、布尔、NaN 和无穷；缺失不得变成真实零值。"""
    if value is None or isinstance(value, bool):
        raise ValueError("RATING_NUMBER_REQUIRED")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("RATING_NUMBER_NONFINITE")
    return result


def stored_score(value) -> Decimal:
    value = decimal(value)
    if not 0 <= value <= 100:
        raise ValueError("RATING_SCORE_RANGE")
    return value.quantize(PRECISION, rounding=ROUND_HALF_EVEN)


def grade(value) -> str:
    """唯一等级映射：先固定持久化精度，再划分等级，Java 与 Vue 不另行判分。"""
    score = stored_score(value)
    return "WEAK" if score < 50 else "AVERAGE" if score < 70 else "GOOD" if score < 90 else "EXCELLENT"


def percentile(value, others: list) -> Decimal:
    if not others:
        raise ValueError("RATING_PEERS_EMPTY")
    x = decimal(value).quantize(PRECISION, rounding=ROUND_HALF_EVEN)
    peers = [decimal(v).quantize(PRECISION, rounding=ROUND_HALF_EVEN) for v in others]
    return Decimal(100) * (sum(p < x for p in peers) + Decimal("0.5") * sum(p == x for p in peers)) / len(peers)


def calculate(members: list[dict], config: dict) -> list[dict]:
    """在已冻结、八维完整的产品样本上计算；异常中止整个类别，不临时删成员。

    同产品仅一个代表，其他份额仍独立计分；排除自身产品后使用相同的比较集。
    返回内部值仅供持久化，公开契约在读取层逐字段构建。
    """
    if set(config["weights"]) != set(DIMENSIONS) or sum(config["weights"].values()) != 100:
        raise ValueError("RATING_WEIGHTS_INVALID")
    if set(config["formula"]) != set(DIMENSIONS):
        raise ValueError("RATING_DIMENSIONS_INVALID")
    for terms in config["formula"].values():
        if not terms or sum(t[2] for t in terms) != 100 or any(t[1] not in (-1, 1) or t[2] <= 0 for t in terms):
            raise ValueError("RATING_FORMULA_INVALID")
    representatives = [m for m in members if m["representative"]]
    if len({m["product_id"] for m in representatives}) != len(representatives):
        raise ValueError("RATING_DUPLICATE_PRODUCT")
    if len(representatives) < config["minimum_products"]:
        raise ValueError("RATING_INSUFFICIENT_PRODUCTS")
    if len({m["fund_code"] for m in members}) != len(members):
        raise ValueError("RATING_DUPLICATE_SHARE")
    if {m["product_id"] for m in members} != {m["product_id"] for m in representatives}:
        raise ValueError("RATING_REPRESENTATIVE_MISSING")
    output = []
    with localcontext() as context:
        context.prec = 40
        for member in members:
            peers = [p for p in representatives if p["product_id"] != member["product_id"]]
            values = {}
            for key, terms in config["formula"].items():
                values[key] = sum(
                    Decimal(weight)
                    / 100
                    * percentile(
                        decimal(member["metrics"][metric]) * direction,
                        [decimal(p["metrics"][metric]) * direction for p in peers],
                    )
                    for metric, direction, weight in terms
                )
            raw = sum(values[k] * Decimal(config["weights"][k]) / 100 for k in DIMENSIONS)
            score = stored_score(raw)
            output.append(
                {
                    "fund_code": member["fund_code"],
                    "raw_score": str(raw),
                    "score": str(score),
                    "grade": grade(score),
                    "dimensions": {k: str(stored_score(v)) for k, v in values.items()},
                }
            )
    return output
