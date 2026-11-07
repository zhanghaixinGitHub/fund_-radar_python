"""复用已冻结的95/5基准构造与联合精度核验；不把构造值冒充公布原值。"""

from datetime import date
from decimal import Decimal, localcontext

from app.services.fund_rating_r5 import number


def reconstruct(
    dates: list[str], closes: dict, official: dict, *, annual_cash_rate: str, expected_missing_dates: list[str],
    publisher_level_half_quantum: str = ".000000005",
) -> dict:
    """实际指数报价与合同现金收益按自然日构造；所有已公布日须共同尺度可行。

    指数报价末位0.0001、管理人基准水平末位0.00000001的半单位精度逐日
    累积成界。不能用单日拟合、随意容差、改现金率或补零通过校验。
    """
    if not dates or dates != sorted(set(dates)) or set(closes) != set(dates):
        raise ValueError("R5_BRIDGE_INDEX_CALENDAR_INCOMPLETE")
    common = sorted(set(dates) & official.keys())
    missing = sorted(set(dates) - official.keys())
    if (len(common) < 2 or missing != expected_missing_dates or common[-1] != dates[-1]
            or common != dates[len(missing):]):
        raise ValueError("R5_BRIDGE_PUBLISHER_CROSS_CHECK_INCOMPLETE")
    with localcontext() as context:
        context.prec = 70
        prices = {d: number(closes[d]) for d in dates}
        published = {d: number(official[d]) for d in common}
        rate, epsilon = number(annual_cash_rate), Decimal(".00005")
        if not 0 <= rate <= Decimal(".1") or min(prices.values()) <= epsilon or min(published.values()) <= 0:
            raise ValueError("R5_BRIDGE_VALUE_DOMAIN")
        levels, bounds = {dates[0]: Decimal(1)}, {dates[0]: (Decimal(1), Decimal(1))}
        for previous, current in zip(dates, dates[1:], strict=False):
            gap = (date.fromisoformat(current) - date.fromisoformat(previous)).days
            cash = Decimal(".05") * rate * gap / 365
            levels[current] = levels[previous] * (
                1 + Decimal(".95") * (prices[current] / prices[previous] - 1) + cash
            )
            low, high = bounds[previous]
            bounds[current] = (
                low * (1 + Decimal(".95") * ((prices[current] - epsilon) / (prices[previous] + epsilon) - 1) + cash),
                high * (1 + Decimal(".95") * ((prices[current] + epsilon) / (prices[previous] - epsilon) - 1) + cash),
            )
        # 两种精度分别来自已核官网原响应：水平值八位小数，或累计收益
        # 两位百分数。调用方必须先核原始字面精度并绑定冻结协议，不能传
        # 任意容差；默认保持原天弘精度，不影响既有封存计算。
        if publisher_level_half_quantum not in {".000000005", ".00005"}:
            raise ValueError("R5_BRIDGE_PUBLISHER_PRECISION_UNSUPPORTED")
        precision = Decimal(publisher_level_half_quantum)
        low = max((published[d] - precision) / bounds[d][1] for d in common)
        high = min((published[d] + precision) / bounds[d][0] for d in common)
        if low > high:
            raise ValueError("R5_BRIDGE_PUBLISHED_PRECISION_CONFLICT")
    return {"levels": [str(levels[d]) for d in dates], "official_overlap": len(common),
            "missing_publisher_dates": sorted(set(dates) - published.keys()),
            "common_scale_interval": [str(low), str(high)], "precision_fit": True,
            "publisher_level_half_quantum": publisher_level_half_quantum,
            "scope": "CONTRACT_DERIVED_REFERENCE_NOT_UNPUBLISHED_ISSUER_OBSERVATIONS"}


def reconstruct_normalized_windows(
    dates: list[str], closes: dict, windows: list[dict], *, annual_cash_rate: str
) -> dict:
    """核验两位百分数的实际归一化窗口，不拼接窗口舍入值冒充原日序列。

    每个窗口的公布累计收益以首日为0；其水平值精度为0.0001。沿实际指数
    0.0001报价精度传播上下界，逐点核对全部窗口（含额外长窗抽样点），并
    要求窗口合计覆盖每个估值日。最终参照取固定合同构造，不按得分拟合。
    """
    if not dates or dates != sorted(set(dates)) or set(closes) != set(dates) or not windows:
        raise ValueError("R5_WINDOW_BRIDGE_CALENDAR")
    with localcontext() as context:
        context.prec = 70
        prices = {day: number(value) for day, value in closes.items()}
        rate, epsilon, published_epsilon = number(annual_cash_rate), Decimal(".00005"), Decimal(".00005")
        if min(prices.values()) <= epsilon or not 0 <= rate <= Decimal(".1"):
            raise ValueError("R5_WINDOW_BRIDGE_DOMAIN")
        levels = {dates[0]: Decimal(1)}
        bounds = {dates[0]: (Decimal(1), Decimal(1))}
        for previous, current in zip(dates, dates[1:], strict=False):
            gap = (date.fromisoformat(current) - date.fromisoformat(previous)).days
            cash = Decimal(".05") * rate * gap / 365
            levels[current] = levels[previous] * (1 + Decimal(".95") * (prices[current] / prices[previous] - 1) + cash)
            low, high = bounds[previous]
            bounds[current] = (
                low * (1 + Decimal(".95") * ((prices[current] - epsilon) / (prices[previous] + epsilon) - 1) + cash),
                high * (1 + Decimal(".95") * ((prices[current] + epsilon) / (prices[previous] - epsilon) - 1) + cash),
            )
        covered, checked = set(), []
        for window in windows:
            observed = window["levels"]
            first, last = window["first"], window["last"]
            if (first not in prices or last not in prices or not observed or min(observed) != first
                    or max(observed) != last or number(observed[first]) != 1
                    or any(day not in prices or not first <= day <= last for day in observed)):
                raise ValueError("R5_WINDOW_BRIDGE_WINDOW_SCOPE")
            max_difference = Decimal(0)
            for day, value in observed.items():
                published = number(value)
                low = bounds[day][0] / bounds[first][1]
                high = bounds[day][1] / bounds[first][0]
                if published <= 0 or published + published_epsilon < low or published - published_epsilon > high:
                    raise ValueError("R5_WINDOW_BRIDGE_PUBLISHED_PRECISION_CONFLICT:" + day)
                max_difference = max(max_difference, abs(published - levels[day] / levels[first]))
                covered.add(day)
            checked.append({"first": first, "last": last, "observations": len(observed),
                            "maximum_midpoint_difference": str(max_difference)})
        if covered != set(dates):
            raise ValueError("R5_WINDOW_BRIDGE_PUBLISHER_DAILY_COVERAGE")
    return {"levels": [str(levels[day]) for day in dates], "published_valuation_dates": len(covered),
            "windows": checked, "precision_fit": True, "publisher_percent_decimal_places": 2,
            "scope": "CONTRACT_DERIVED_REFERENCE_NOT_STITCHED_OR_UNPUBLISHED_ISSUER_VALUES"}
