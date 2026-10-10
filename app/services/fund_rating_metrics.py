"""可独立复算的评级输入计算；所有比例使用 0—1，金额为同币种元。"""

from datetime import date
from decimal import Decimal, localcontext
from statistics import median

from app.services.fund_rating_rules import decimal


def total_return(nav: list, dividends: list, splits: list) -> list[Decimal]:
    """净值、每份现金分红和份额折算倍数已按同一除息日对齐；不使用累计净值相除。

    上游必须证实事件覆盖完整；分红为折算后每份金额，先乘折算倍数再除前日净值。
    已体现在净值内的费用不再次扣除，真实下跌不裁剪。
    """
    if len(nav) < 2 or len(nav) != len(dividends) or len(nav) != len(splits):
        raise ValueError("RATING_RETURN_ALIGNMENT")
    n, d, s = ([decimal(x) for x in values] for values in (nav, dividends, splits))
    if any(x <= 0 for x in n + s) or any(x < 0 for x in d):
        raise ValueError("RATING_RETURN_RANGE")
    result = [Decimal(1)]
    for i in range(1, len(n)):
        result.append(result[-1] * (n[i] + d[i]) * s[i] / n[i - 1])
    return result


def history_metrics(
    dates: list[date], index: list, rf: list, month_ends: list[date], annual_days: int, *, require_sharpe: bool = True
) -> dict:
    """需要日历与序列逐日相等；37 个月末取 25 个滚动年度窗口，无风险序列对齐日间隔。"""
    if dates != sorted(set(dates)) or len(dates) != len(index) or (require_sharpe and len(rf) != len(index) - 1):
        raise ValueError("RATING_CALENDAR_ALIGNMENT")
    if len(month_ends) != 37 or month_ends != sorted(set(month_ends)):
        raise ValueError("RATING_MONTH_WINDOW")
    months = [d.year * 12 + d.month for d in month_ends]
    if months != list(range(months[0], months[0] + 37)) or dates[0] != month_ends[0] or dates[-1] != month_ends[-1]:
        raise ValueError("RATING_MONTH_WINDOW")
    if not 200 <= annual_days <= 366:
        raise ValueError("RATING_ANNUALIZATION")
    values = [decimal(x) for x in index]
    if any(v <= 0 for v in values):
        raise ValueError("RATING_RETURN_RANGE")
    endpoints = [values[dates.index(d)] for d in month_ends]
    r = [b / a - 1 for a, b in zip(values, values[1:], strict=False)]
    # 货币类别完全不使用夏普；这里的收益仅用于公共窗口，绝不是默认零无风险收益。
    excess = [a - decimal(b) for a, b in zip(r, rf, strict=True)] if require_sharpe else r
    mean = sum(excess) / len(excess)
    variance = sum((x - mean) ** 2 for x in excess) / (len(excess) - 1)
    if require_sharpe and variance <= 0:
        raise ValueError("RATING_ZERO_VARIANCE")
    peak = values[0]
    drawdown = Decimal(0)
    for value in values:
        peak = max(peak, value)
        drawdown = max(drawdown, 1 - value / peak)
    with localcontext() as ctx:
        ctx.prec = 40
        return {
            "return_1y": endpoints[-1] / endpoints[-13] - 1,
            "return_3y": (values[-1] / values[0]) ** (Decimal("365.25") / (dates[-1] - dates[0]).days) - 1,
            "drawdown": drawdown,
            "downside": (Decimal(annual_days) * sum(min(x, 0) ** 2 for x in r) / len(r)).sqrt(),
            "sharpe": Decimal(annual_days).sqrt() * mean / variance.sqrt() if require_sharpe else None,
            "rolling": [endpoints[i] / endpoints[i - 12] - 1 for i in range(12, 37)],
        }


def stability(members: list[dict]) -> None:
    """固定产品代表的每个窗口中位数；并列记半次，25 窗口不解释为独立试验。"""
    reps = [m for m in members if m["representative"]]
    centers = [median(decimal(m["metrics"]["rolling"][i]) for m in reps) for i in range(25)]
    for m in members:
        wins = [
            Decimal(1) if decimal(x) > y else Decimal("0.5") if decimal(x) == y else Decimal(0)
            for x, y in zip(m["metrics"]["rolling"], centers, strict=True)
        ]
        m["metrics"]["stability"] = str(sum(wins) / 25)
        if m.get("family") == "MONEY":
            m["metrics"]["distribution_stability"] = str(sum(wins) / 25)


def holdings_metrics(positions: list[dict], net_assets, cash, payable, largest_holder) -> dict:
    """完整股票持仓：发行人合并；同口径行业、20 日成交額、已知可售金额缺一不可。"""
    net, cash, payable, holder = map(decimal, (net_assets, cash, payable, largest_holder))
    if net <= 0 or cash < 0 or payable < 0 or not 0 <= holder <= 1 or not positions:
        raise ValueError("RATING_HOLDING_RANGE")
    issuers, industries = {}, {}
    total = Decimal(0)
    liquid = cash - payable
    for p in positions:
        amount, saleable = decimal(p["amount"]), decimal(p["saleable_amount"])
        if not p["issuer"] or not p["industry"] or amount <= 0 or not 0 <= saleable <= amount:
            raise ValueError("RATING_HOLDING_IDENTITY")
        turnovers = [decimal(x) for x in p["turnover_20d"]]
        if len(turnovers) != 20 or any(x < 0 for x in turnovers):
            raise ValueError("RATING_TURNOVER_MISSING")
        total += amount
        issuers[p["issuer"]] = issuers.get(p["issuer"], Decimal(0)) + amount
        industries[p["industry"]] = industries.get(p["industry"], Decimal(0)) + amount
        liquid += min(amount, saleable, Decimal("0.5") * median(turnovers))
    # 没有其他已支持资产的首批权益情景必须逐元对账，不能把未知资产视为现金。
    if abs(total + cash - payable - net) > Decimal("0.01"):
        raise ValueError("RATING_ASSETS_NOT_RECONCILED")
    return {
        "issuer_hhi": sum((x / total) ** 2 for x in issuers.values()),
        "industry_hhi": sum((x / total) ** 2 for x in industries.values()),
        "liquid_5d": min(Decimal(1), max(Decimal(0), liquid / net)),
        "largest_holder": holder,
    }


def standard_cost(management, custody, sales, subscription, redemption, *, mode: str) -> Decimal:
    """一万元、365 日、无涨跌、场外标准费率情景；外扣按实际申购本金基数计算。"""
    m, c, s, buy, sell = map(decimal, (management, custody, sales, subscription, redemption))
    if any(not 0 <= x <= 1 for x in (m, c, s, buy, sell)) or mode not in {"EXTERNAL", "INTERNAL"}:
        raise ValueError("RATING_FEE_RANGE")
    purchase = buy / (1 + buy) if mode == "EXTERNAL" else buy
    return purchase + (1 - purchase) * sell + m + c + s
