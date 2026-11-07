"""境内直接指数基金的R5原始观察编译，不改变已冻结换分和八维权重。"""

from collections import defaultdict
from datetime import date
from decimal import Decimal, localcontext

from app.services.fund_rating_r5 import number


def performance(dates: list[str], month_ends: list[str], nav: list, benchmark: list) -> dict:
    """同一真实日历的费后表现、相对回撤/下行及跟踪误差；不补非交易日。

    收益/误差单位均为比例。三年年化沿既有365.25/实际天数；跟踪误差
    使用样本标准差，额外下行使用完整观察的负差平方均值，均按252日年化。
    """
    if (
        len(dates) != len(nav) or len(nav) != len(benchmark) or len(dates) < 2
        or dates != sorted(set(dates)) or len(month_ends) != 37
        or month_ends != sorted(set(month_ends)) or not set(month_ends) <= set(dates)
        or dates[0] != month_ends[0] or dates[-1] != month_ends[-1]
    ):
        raise ValueError("R5_DIRECT_INDEX_COMMON_WINDOW")
    months = [date.fromisoformat(d).year * 12 + date.fromisoformat(d).month for d in month_ends]
    if months != list(range(months[0], months[0] + 37)):
        raise ValueError("R5_DIRECT_INDEX_MONTH_WINDOW")
    with localcontext() as context:
        context.prec = 70
        values, base = [number(v) for v in nav], [number(v) for v in benchmark]
        if min(values + base) <= 0:
            raise ValueError("R5_DIRECT_INDEX_NAV_OR_BENCHMARK")
        n, b = dict(zip(dates, values, strict=True)), dict(zip(dates, base, strict=True))
        differences = [values[i] / values[i - 1] - base[i] / base[i - 1] for i in range(1, len(values))]
        count = Decimal(len(differences))
        mean = sum(differences) / count
        if count < 2:
            raise ValueError("R5_DIRECT_INDEX_TRACKING_WINDOW")
        tracking = (sum((x - mean) ** 2 for x in differences) / (count - 1) * 252).sqrt()
        downside = (sum(min(x, Decimal(0)) ** 2 for x in differences) / count * 252).sqrt()
        peak, drawdown = values[0] / base[0], Decimal(0)
        for own, reference in zip(values, base, strict=True):
            relative = own / reference
            peak = max(peak, relative)
            drawdown = max(drawdown, 1 - relative / peak)
        span = Decimal((date.fromisoformat(dates[-1]) - date.fromisoformat(dates[0])).days)
        annualization = Decimal("365.25") / span
        one_start = month_ends[-13]
        one = n[dates[-1]] / n[one_start] - 1
        three = (values[-1] / values[0]) ** annualization - 1
        result = {
            "return_1y": str(one - (b[dates[-1]] / b[one_start] - 1)),
            "return_3y": str(three - ((base[-1] / base[0]) ** annualization - 1)),
            "net_return_1y": str(one), "net_annual_return_3y": str(three),
            "extra_drawdown": str(drawdown), "extra_downside": str(downside), "tracking_error": str(tracking),
        }
        for name, step, first in [("monthly_success", 1, 1), ("annual_success", 12, 12)]:
            result[name] = [
                {"start": month_ends[i - step], "end": month_ends[i],
                 "target": str(n[month_ends[i]] / n[month_ends[i - step]] - 1),
                 "contract_reference": str(b[month_ends[i]] / b[month_ends[i - step]] - 1)}
                for i in range(first, 37)
            ]
        return result


def replication(positions: list[dict], stock_total: str, index_weights: dict, issuers: dict, industries: dict) -> dict:
    """按R5既有股票内部归一口径核全股票和指数，合并真实发行人及细行业。"""
    if not positions or not index_weights or len(positions) > 5000 or len(index_weights) > 5000:
        raise ValueError("R5_DIRECT_INDEX_PORTFOLIO_REQUIRED")
    securities = [p["stock_code"] for p in positions]
    if len(set(securities)) != len(securities):
        raise ValueError("R5_DIRECT_INDEX_DUPLICATE_POSITION")
    with localcontext() as context:
        context.prec = 70
        total = number(stock_total)
        amounts = {p["stock_code"]: number(p["market_value_cny"]) for p in positions}
        weights = {key: number(v) for key, v in index_weights.items()}
        if total <= 0 or min(amounts.values()) <= 0 or min(weights.values()) <= 0 or sum(amounts.values()) != total:
            raise ValueError("R5_DIRECT_INDEX_STOCK_TOTAL")
        # 指数原发布权重保留，股票内部归一沿已冻结R5；不是填补未披露成分。
        target_total = sum(weights.values())
        if target_total <= 0:
            raise ValueError("R5_DIRECT_INDEX_TARGET_WEIGHT")
        all_securities = amounts.keys() | weights.keys()
        if not all_securities <= issuers.keys() or not all_securities <= industries.keys():
            raise ValueError("R5_DIRECT_INDEX_ISSUER_OR_INDUSTRY_MISSING")
        result = {"reported_index_weight_sum": str(target_total)}
        for name, mapping in [("issuer_replication", issuers), ("industry_replication", industries)]:
            actual, target = defaultdict(Decimal), defaultdict(Decimal)
            for sec, amount in amounts.items():
                if not mapping[sec]:
                    raise ValueError("R5_DIRECT_INDEX_EMPTY_MAPPING")
                actual[mapping[sec]] += amount / total
            for sec, weight in weights.items():
                target[mapping[sec]] += weight / target_total
            # 排序保持跨进程Decimal求和及原始观察回放一致，不依赖哈希种子。
            result[name] = str(sum(abs(actual[k] - target[k]) for k in sorted(actual.keys() | target.keys())) / 2)
        return result


def capacity(positions: list[dict], stock_total: str, dates: list[str], market: dict) -> dict:
    """原60交易日K：Σ股票权重×持仓市值/日均成交额；金额原单位为千元。

    缺日不补零；本编译器要求60日完整原响应。上市前或停牌的特殊缺日须
    先由专门证据适配器闭合，不能在此凭股票名称猜原因。
    """
    if len(dates) != 60 or dates != sorted(set(dates)):
        raise ValueError("R5_DIRECT_INDEX_CAPACITY_WINDOW")
    with localcontext() as context:
        context.prec = 70
        total, value, rows = number(stock_total), Decimal(0), []
        if sum(number(p["market_value_cny"]) for p in positions) != total or total <= 0:
            raise ValueError("R5_DIRECT_INDEX_CAPACITY_STOCK_TOTAL")
        for position in positions:
            sec = position["stock_code"]
            if sec not in market or not set(dates) <= market[sec].keys():
                raise ValueError("R5_DIRECT_INDEX_MARKET_DAYS_MISSING:" + sec)
            amounts = [number(market[sec][day]) * 1000 for day in dates]
            if min(amounts) < 0 or sum(amounts) <= 0:
                raise ValueError("R5_DIRECT_INDEX_MARKET_AMOUNT:" + sec)
            adv = sum(amounts) / len(amounts)
            holding = number(position["market_value_cny"])
            contribution = holding / total * holding / adv
            value += contribution
            rows.append({"security": sec, "adv_cny": str(adv), "contribution": str(contribution)})
        return {"K": str(value), "rows": rows}


def share_flow(rows: list[dict]) -> list[dict]:
    """各份额分别对账；横杠拆分项仅在非负份额精确守恒时唯一识别为零。"""
    if not rows or len({r["share"] for r in rows}) != len(rows):
        raise ValueError("R5_DIRECT_INDEX_SHARE_FLOW_SCOPE")
    proof = []
    for row in rows:
        opening, subscribed, redeemed, closing = [
            number(row[k]) for k in ("opening", "subscribed", "redeemed", "closing")
        ]
        if min(opening, subscribed, redeemed, closing) < 0:
            raise ValueError("R5_DIRECT_INDEX_SHARE_FLOW_VALUE")
        identified = closing - opening - subscribed + redeemed
        reported = row.get("split")
        if reported is None and identified != 0 or reported is not None and identified != number(reported):
            raise ValueError("R5_DIRECT_INDEX_SHARE_FLOW_NOT_RECONCILED")
        proof.append({"share": row["share"], "split": str(identified),
                      "basis": "EXACT_CONSERVATION_IDENTIFIED_ZERO" if reported is None else "REPORTED_VALUE"})
    return proof
