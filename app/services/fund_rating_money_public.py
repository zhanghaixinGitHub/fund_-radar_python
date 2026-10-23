"""可复用货币基金公开口径计算器；金额元、比例0..1、分数0..100。

本模块无网络、数据库、定时任务或业务写入；输入须由原件勾稽后生成。
区间端点为证据边界，绝不自动取中点或重分维内权重。
"""

from decimal import Decimal as D
from decimal import localcontext
from functools import wraps


def precise(function):
    """每次复算固定80位精度，避免修改服务其他计算共享的Decimal上下文。"""

    @wraps(function)
    def wrapped(*args, **kwargs):
        with localcontext() as context:
            context.prec = 80
            return function(*args, **kwargs)

    return wrapped


OUTER = {
    "return": 15,
    "risk": 20,
    "execution": 10,
    "stability": 10,
    "management": 15,
    "holdings": 10,
    "liquidity": 10,
    "fee": 10,
}


def decimal(value):
    """拒绝缺值、布尔值和非有限数；不能把空披露或错误响应转成0。"""
    if value is None or isinstance(value, bool):
        raise ValueError("缺少实际数值")
    number = D(str(value))
    if not number.is_finite():
        raise ValueError("数值非有限")
    return number


def nonnegative(value):
    number = decimal(value)
    if number < 0:
        raise ValueError("金额/比例不得小于0")
    return number


def interval(lower, upper, **extra):
    lo, hi = decimal(lower), decimal(upper)
    if not 0 <= lo <= hi <= 100:
        raise ValueError("分数边界不合法")
    return {"lower": str(lo), "upper": str(hi), "candidate": str(lo) if lo == hi else None, **extra}


@precise
def weighted(items):
    """固定内部权重；缺子项保留0..100界，不把其余子项归一。"""
    if sum(decimal(w) for w, _ in items) != 1:
        raise ValueError("维内权重未合计1")
    lo = sum(decimal(w) * decimal(r["lower"]) for w, r in items)
    hi = sum(decimal(w) * decimal(r["upper"]) for w, r in items)
    return interval(lo, hi)


@precise
def concentration(ratio, unit):
    """单一信用暴露越集中，无法分散的损失暴露越大；单位为候选风险预算。"""
    r, u = nonnegative(ratio), decimal(unit)
    if u <= 0:
        raise ValueError("量尺必须大于0")
    return 100 * u / (u + r)


@precise
def bank_concentration_bounds(known_by_bank, unknown_ncd, unknown_deposits, nav, deposit_complete_banks=()):
    """已知银行为逐证券最大完整/质押下界去重后加已知存款。

    未分配存单可归任何银行；未分配存款不得归已披露完整存款的银行。
    上界允许未知集中，未宣称这个极端组合确实存在。
    """
    nv = decimal(nav)
    if nv <= 0 or not known_by_bank:
        raise ValueError("缺银行下界或NAV")
    known = {k: nonnegative(v) for k, v in known_by_bank.items()}
    un, ud = nonnegative(unknown_ncd), nonnegative(unknown_deposits)
    low = max(known.values()) / nv
    high = max([un + ud] + [v + un + (0 if k in deposit_complete_banks else ud) for k, v in known.items()]) / nv
    return low, high


@precise
def holdings(inputs, unit_scale="1"):
    """银行合并敞口、最大单券、证券评级三子项；未知信用质量保持区间。"""
    scale = decimal(unit_scale)
    nav = decimal(inputs["NAV"])
    low, high = bank_concentration_bounds(
        inputs["known_bank_amounts"],
        inputs["unallocated_ncd"],
        inputs["unallocated_deposits"],
        nav,
        inputs.get("deposit_complete_banks", ()),
    )
    bank = interval(concentration(high, D(".20") * scale), concentration(low, D(".20") * scale))
    largest = nonnegative(inputs["largest_credit_security"]) / nav
    security = interval(concentration(largest, D(".10") * scale), concentration(largest, D(".10") * scale))
    securities = decimal(inputs["securities_total"])
    if securities <= 0:
        raise ValueError("本证券子方法需要实际证券余额")
    # 评级表未知桶的价值不被当成零风险。已核政策债可以从未评级桶重新识别，必须不重复。
    groups = inputs["rating_groups"]
    if sum(nonnegative(g["amount"]) for g in groups) != securities:
        raise ValueError("评级组与全部证券账面额不闭合")
    rating = interval(
        sum(nonnegative(g["amount"]) * decimal(g["score_lower"]) for g in groups) / securities,
        sum(nonnegative(g["amount"]) * decimal(g["score_upper"]) for g in groups) / securities,
    )
    for g in groups:
        interval(g["score_lower"], g["score_upper"])
    result = weighted([(".35", bank), (".35", security), (".30", rating)])
    result["subcomponents"] = {
        "same_bank_concentration": bank,
        "largest_credit_security": security,
        "security_rating_structure": rating,
    }
    result["raw"] = {
        "largest_bank_ratio_lower": str(low),
        "largest_bank_ratio_upper": str(high),
        "largest_credit_security_ratio": str(largest),
        "rating_exactly_classified_securities_ratio": str(
            sum(nonnegative(g["amount"]) for g in groups if g["score_lower"] == g["score_upper"]) / securities
        ),
        "securities_fraction_of_gross_assets": str(securities / decimal(inputs["gross_assets"])),
    }
    return result


@precise
def liquidity(inputs, unit_scale="1"):
    """公开期限结构、受限证券、活期本金/负债、持有人集中度；不预测未来到账。"""
    scale = decimal(unit_scale)
    if scale <= 0:
        raise ValueError("量尺必须大于0")
    asset_ratio = nonnegative(inputs["assets_30day_ratio"])
    liability_ratio = nonnegative(inputs["liabilities_30day_ratio"])
    # 报告比率已是0..1；常见百分数误传应拒绝。
    if asset_ratio > 2 or liability_ratio > 2:
        raise ValueError("期限表比例单位错误")
    net = max(D(0), asset_ratio - liability_ratio)
    maturity = 100 * net / (net + D(".10") * scale)
    securities = decimal(inputs["securities_total"])
    restricted = nonnegative(inputs["restricted_securities"])
    if securities <= 0 or restricted > securities:
        raise ValueError("证券受限金额越界")
    if inputs.get("restriction_scope_complete") is not True:
        raise ValueError("未核完整受限证券范围")
    restriction = 100 * (1 - restricted / securities)
    cash, liab = nonnegative(inputs["ordinary_cash_principal"]), nonnegative(inputs["all_liabilities"])
    payment = D(100) if liab == 0 else 100 * cash / (cash + liab)
    holder = nonnegative(inputs["top10_holder_fraction"])
    if holder > 1:
        raise ValueError("前十持有人比例错误")
    holder_score = concentration(holder, D(".20") * scale)
    subs = {
        k: interval(v, v)
        for k, v in {
            "maturity_structure": maturity,
            "security_transfer_restriction": restriction,
            "cash_liability_match": payment,
            "holder_concentration": holder_score,
        }.items()
    }
    result = weighted(
        [
            (w, subs[k])
            for k, w in [
                ("maturity_structure", ".40"),
                ("security_transfer_restriction", ".20"),
                ("cash_liability_match", ".25"),
                ("holder_concentration", ".15"),
            ]
        ]
    )
    result["subcomponents"] = subs
    result["raw"] = {
        "net_30day_ratio": str(net),
        "restriction_bonds_fraction": str(restricted / securities),
        "ordinary_cash_to_all_liabilities": str(cash / liab) if liab else None,
        "top10_holder_fraction": str(holder),
    }
    return result


@precise
def execution(numeric_observations, category_observations):
    """观测状态由原件和当时合同核准。UNKNOWN不能当作正常75。

    这里只评价事先约定的公开检验集，不把该集改名成全部法定约束。
    """

    def component(rows):
        if not rows or any(r["state"] == "UNKNOWN" for r in rows):
            return interval(0, 100, known=False)
        mapping = {
            "OBSERVED_WITHIN": D(75),
            "OBSERVED_OUTSIDE_RESPONSIBILITY_UNKNOWN": D(50),
            "CONFIRMED_UNCORRECTED": D(25),
            "CONFIRMED_MAJOR_FAILURE": D(0),
        }
        if any(r["state"] not in mapping for r in rows):
            raise ValueError("未定义执行状态")
        score = sum(mapping[r["state"]] for r in rows) / len(rows)
        return interval(score, score, known=True, observations=len(rows))

    n, c = component(numeric_observations), component(category_observations)
    result = weighted([(".60", n), (".40", c)])
    result["subcomponents"] = {"numeric_observations": n, "investment_categories": c}
    return result


@precise
def aggregate(dimensions):
    """只有完整点分参与部分研究点分；区间按原权重另合成，禁止中点冒充完整总分。"""
    if set(dimensions) != set(OUTER):
        raise ValueError("必须保存全部八维")
    point_weight = sum(OUTER[k] for k, v in dimensions.items() if v.get("candidate") is not None)
    interval_weight = sum(
        OUTER[k]
        for k, v in dimensions.items()
        if v.get("candidate") is None
        and v.get("informative_interval", True)
        and (decimal(v["lower"]) > 0 or decimal(v["upper"]) < 100)
    )
    points = sum(
        D(OUTER[k]) * decimal(v["candidate"]) / 100 for k, v in dimensions.items() if v.get("candidate") is not None
    )
    full_lower = sum(D(OUTER[k]) * decimal(v["lower"]) / 100 for k, v in dimensions.items())
    full_upper = sum(D(OUTER[k]) * decimal(v["upper"]) / 100 for k, v in dimensions.items())
    return {
        "point_weight_percent": point_weight,
        "informative_interval_weight_percent": interval_weight,
        "missing_weight_percent": 100 - point_weight - interval_weight,
        "point_dimension_count": sum(v.get("candidate") is not None for v in dimensions.values()),
        "partial_research_score": str(points * 100 / point_weight) if point_weight else None,
        "original_weighted_known_points": str(points),
        "eight_dimension_interval": {
            "lower": str(full_lower),
            "upper": str(full_upper),
            "all_dimensions_have_informative_evidence": point_weight + interval_weight == 100,
        },
        "complete_point_score": str(full_lower) if point_weight == 100 else None,
        "grade": None,
        "interval_midpoint_used": False,
        "formal_activation": False,
    }
