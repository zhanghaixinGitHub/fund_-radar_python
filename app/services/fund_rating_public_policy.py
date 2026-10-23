"""公开披露八维评价政策：确定性绝对区间，非同类排名、非未来收益预测。

锚点和权重在首次案例综合试算前固定，属于本应用政策而非行业标准。
所有指标必须来自完整原件，未知值不能补零；一维缺失即停止总评级。
"""

from decimal import Decimal, localcontext

from app.services.fund_rating_rules import DIMENSIONS, WEIGHTS, decimal, grade, stored_score

BASIS = "PUBLIC_DISCLOSURE_EQUITY_V1"
ETF_BASIS = "PUBLIC_DISCLOSURE_ETF_LINK_V1"
MARKET_BASIS = "PUBLIC_DISCLOSURE_DOMESTIC_EQUITY_V1"
INDEX_BASIS = "PUBLIC_DISCLOSURE_DOMESTIC_INDEX_V1"
RECONCILED_BASIS = "PUBLIC_DISCLOSURE_DOMESTIC_INDEX_RECONCILED_V1"
GLOBAL_BASIS = "PUBLIC_DISCLOSURE_GLOBAL_EQUITY_CNY_V1"
MARKET_BASES = frozenset((MARKET_BASIS, GLOBAL_BASIS))
INDEX_BASES = frozenset((INDEX_BASIS, RECONCILED_BASIS))
REVIEWED_BASES = frozenset(("PUBLIC_REVIEWED_ACTIVE_QUALITY_V2", "PUBLIC_REVIEWED_PASSIVE_QUALITY_V2"))
SNAPSHOT_BASIS = "PUBLIC_REVIEWED_QUALITY_SNAPSHOT_V3"
PUBLIC_BASES = frozenset((BASIS, ETF_BASIS, *MARKET_BASES, *INDEX_BASES, *REVIEWED_BASES, SNAPSHOT_BASIS))
ANCHORS = {
    "return_1y": [(-0.20, 0), (-0.05, 30), (0, 50), (0.05, 70), (0.10, 90), (0.20, 100)],
    "return_3y": [(-0.20, 0), (-0.05, 30), (0, 50), (0.05, 70), (0.10, 90), (0.20, 100)],
    "drawdown": [(0, 100), (0.10, 90), (0.20, 70), (0.35, 50), (0.50, 20), (0.65, 0)],
    "downside": [(0, 100), (0.10, 90), (0.20, 70), (0.30, 50), (0.40, 20), (0.50, 0)],
    "calmar": [(-1, 0), (0, 30), (0.25, 50), (0.5, 70), (1, 90), (2, 100)],
    "rolling_positive": [(0, 0), (1, 100)],
    "monthly_positive": [(0, 0), (1, 100)],
    "team_months": [(0, 20), (6, 50), (12, 70), (36, 100)],
    "team_changes": [(0, 100), (1, 80), (2, 60), (4, 20), (6, 0)],
    "strategy_continuity": [(0, 50), (1, 100)],
    "governance_state": [(0, 20), (1, 50), (2, 100)],
    "issuer_hhi": [(0, 100), (0.02, 100), (0.05, 80), (0.10, 60), (0.25, 30), (1, 0)],
    "industry_hhi": [(0, 100), (0.02, 100), (0.05, 80), (0.10, 60), (0.25, 30), (1, 0)],
    "restricted_ratio": [(0, 100), (0.05, 80), (0.10, 50), (0.15, 20), (0.30, 0)],
    "cash_ratio": [(0, 0), (0.03, 50), (0.05, 70), (0.10, 100)],
    # 0/1/2 为公开风险档，不是持有人精确比例：无>=20%、20%—50%、>=50%。
    "holder_risk_band": [(0, 100), (1, 50), (2, 0)],
    "net_assets": [(0, 0), (50_000_000, 50), (200_000_000, 100)],
    "cost_1y": [(0, 100), (0.005, 90), (0.01, 70), (0.02, 50), (0.03, 20), (0.05, 0)],
}
TERMS = {
    "return": [("return_1y", 30), ("return_3y", 70)],
    "risk": [("drawdown", 70), ("downside", 30)],
    "efficiency": [("calmar", 100)],
    "stability": [("rolling_positive", 70), ("monthly_positive", 30)],
    "management": [("team_months", 40), ("team_changes", 20), ("strategy_continuity", 20), ("governance_state", 20)],
    "holdings": [("issuer_hhi", 50), ("industry_hhi", 50)],
    "liquidity": [("restricted_ratio", 40), ("cash_ratio", 30), ("holder_risk_band", 20), ("net_assets", 10)],
    "cost": [("cost_1y", 100)],
}


def policy(basis: str = BASIS) -> dict:
    if basis in REVIEWED_BASES:
        from app.services.fund_rating_reviewed_quality import policy as reviewed_policy

        return reviewed_policy(basis)
    if basis == GLOBAL_BASIS:
        from app.services.fund_rating_public_global import global_policy

        return global_policy()
    if basis == RECONCILED_BASIS:
        from app.services.fund_rating_public_reconciled import reconciled_policy

        return reconciled_policy()
    if basis == INDEX_BASIS:
        from app.services.fund_rating_public_index import index_policy

        return index_policy()
    """完整可哈希政策；local 限制避免将公开读取事实冒充已取得商业再发布许可。"""
    if basis == MARKET_BASIS:
        from app.services.fund_rating_public_market import market_policy

        return market_policy()
    if basis == ETF_BASIS:
        from app.services.fund_rating_public_etf import etf_policy

        return etf_policy()
    if basis != BASIS:
        raise ValueError("RATING_PUBLIC_BASIS_UNSUPPORTED")
    return {
        "basis": BASIS,
        "family": "ACTIVE_EQUITY",
        "revision": "public-equity-2026-10-14-v1",
        # 展示依据也随不可变快照发布；文案修订不覆盖首个历史结果。
        "public_evidence_version": 2,
        "weights": dict(zip(DIMENSIONS, WEIGHTS, strict=True)),
        "terms": TERMS,
        "anchors": {k: [[str(x), y] for x, y in v] for k, v in ANCHORS.items()},
        "minimum_products": 1,
        "comparison": "ABSOLUTE_NO_PEER_RANK",
        "local_only": True,
        "holdings_age_days": 210,
        "verification_age_days": 31,
        "history_months": 36,
        "status": "REQUIRES_REAL_VALIDATION",
        "rounding": "ROUND_HALF_EVEN",
    }


def interpolate(value, anchors) -> Decimal:
    x = decimal(value)
    points = [(decimal(a), decimal(b)) for a, b in anchors]
    if x <= points[0][0]:
        return points[0][1]
    for (a, sa), (b, sb) in zip(points, points[1:], strict=False):
        if x <= b:
            return sa + (sb - sa) * (x - a) / (b - a)
    return points[-1][1]


def calculate_public(metrics: dict, *, config: dict | None = None) -> dict:
    """八维同时计算；只接受指标，不接受调用方提供评级或分项分值。"""
    config = config or policy()
    if config.get("basis") in REVIEWED_BASES:
        from app.services.fund_rating_reviewed_quality import calculate

        return calculate(metrics, config=config)
    weights = {k: decimal(v) for k, v in config["weights"].items()}
    if set(weights) != set(DIMENSIONS) or any(v <= 0 for v in weights.values()) or sum(weights.values()) != 100:
        raise ValueError("RATING_WEIGHTS_INVALID")
    if set(config["terms"]) != set(DIMENSIONS):
        raise ValueError("RATING_EIGHT_DIMENSIONS_REQUIRED")
    with localcontext() as ctx:
        ctx.prec = 40
        dimensions = {}
        for key, terms in config["terms"].items():
            if not terms or sum(w for _, w in terms) != 100:
                raise ValueError("RATING_FORMULA_INVALID")
            dimensions[key] = sum(
                interpolate(metrics[name], config["anchors"][name]) * weight / 100 for name, weight in terms
            )
        raw = sum(dimensions[k] * weights[k] / 100 for k in DIMENSIONS)
        score = stored_score(raw)
        return {
            "raw_score": str(raw),
            "score": str(score),
            "grade": grade(score),
            "dimensions": {k: str(stored_score(v)) for k, v in dimensions.items()},
        }
