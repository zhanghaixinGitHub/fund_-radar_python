"""股票成交负荷研究候选：只产生原始诊断，不进入现行V2契约或评级。

使用报告期末可流通股票市值和此前20个已核交易日的实际成交金额。
计算值是固定假设下的历史成交负荷，不能解释成未来实际卖出天数、
全资产流动性或策略容量上限。受限资产另有归属，不重复放进分子。
"""

from datetime import date
from decimal import Decimal

from app.services.fund_rating_rules import decimal

VERSION = "QUALITY_STOCK_TURNOVER_PILOT_20261017_1"


def stock_turnover_load(
    *,
    positions: list[dict],
    stock_amount: str,
    restricted_rows: list[dict],
    restricted_amount: str,
    days: list[str],
    quotes: dict[str, dict],
    report_date: str,
    stress_fraction: str,
    participation_fraction: str,
) -> dict:
    """对已逐原件核验的完整股票/受限表计算成交负荷，金额均为人民币元。

    quotes为按交易日、证券代码索引的成交金额及其原始响应哈希；调用者
    必须先校验来源权限、响应原字节、证券身份和完整交易日历。本函数再
    核对全表金额及逐证券受限金额，不依据行业名称或缺少记录推断可流通。
    任一正可流通持仓缺少20日成交额时返回精确缺口，整个汇总值保持空缺。
    """
    if len(days) != 20 or days != sorted(set(days)) or days[-1] != report_date:
        raise ValueError("TURNOVER_TWENTY_SESSIONS_REQUIRED")
    if any(date.fromisoformat(day) > date.fromisoformat(report_date) for day in days):
        raise ValueError("TURNOVER_FUTURE_SESSION")
    stress, participation = decimal(stress_fraction), decimal(participation_fraction)
    if not (0 < stress <= 1 and 0 < participation <= 1):
        raise ValueError("TURNOVER_POLICY_DOMAIN")
    by_code = {}
    for row in positions:
        code = row["stock_code"]
        if code in by_code or not code.endswith((".SH", ".SZ")):
            raise ValueError("TURNOVER_SECURITY_SCOPE_OR_DUPLICATE")
        amount, quantity = decimal(row["market_value_cny"]), decimal(row["quantity_shares"])
        if amount < 0 or quantity <= 0:
            raise ValueError("TURNOVER_POSITION_DOMAIN")
        by_code[code] = {"market_value_cny": amount, "quantity_shares": quantity}
    total = decimal(stock_amount)
    if total <= 0 or abs(sum(r["market_value_cny"] for r in by_code.values()) - total) > Decimal("0.02"):
        raise ValueError("TURNOVER_FULL_STOCK_TABLE_NOT_RECONCILED")
    restrictions = {}
    for row in restricted_rows:
        matches = [code for code in by_code if code.split(".")[0] == row["code"]]
        if len(matches) != 1:
            raise ValueError("TURNOVER_RESTRICTED_SECURITY_NOT_BOUND")
        code = matches[0]
        amount, quantity = decimal(row["amount"]), decimal(row["quantity"])
        if amount < 0 or quantity <= 0:
            raise ValueError("TURNOVER_RESTRICTION_DOMAIN")
        prior = restrictions.setdefault(code, {"amount": Decimal(0), "quantity": Decimal(0)})
        prior["amount"] += amount
        prior["quantity"] += quantity
    restricted = decimal(restricted_amount)
    if restricted < 0 or abs(sum(r["amount"] for r in restrictions.values()) - restricted) > Decimal("0.02"):
        raise ValueError("TURNOVER_RESTRICTED_TABLE_NOT_RECONCILED")
    rows, gaps = [], []
    for code, holding in sorted(by_code.items()):
        locked = restrictions.get(code, {"amount": Decimal(0), "quantity": Decimal(0)})
        if locked["amount"] > holding["market_value_cny"] or locked["quantity"] > holding["quantity_shares"]:
            raise ValueError("TURNOVER_RESTRICTED_EXCEEDS_HOLDING")
        free = holding["market_value_cny"] - locked["amount"]
        row = {
            "stock_code": code,
            "stock_value_cny": str(holding["market_value_cny"]),
            "restricted_value_cny": str(locked["amount"]),
            "free_value_cny": str(free),
        }
        if free == 0:
            # 完整受限原表对账出的零流通金额，不是把缺失成交额补零。
            row["state"] = "FULLY_RESTRICTED_EXCLUDED_FROM_FREE_STOCK_BOOK"
            rows.append(row)
            continue
        amounts, hashes, missing = [], [], []
        for day in days:
            quote = quotes.get(day, {}).get(code)
            if not quote or quote.get("amount_cny") is None or not quote.get("source_hash"):
                missing.append(day)
                continue
            amount = decimal(quote["amount_cny"])
            if amount < 0:
                raise ValueError("TURNOVER_NEGATIVE_AMOUNT")
            amounts.append(amount)
            hashes.append(quote["source_hash"])
        if missing:
            row.update(state="MISSING_DAILY_AMOUNT_NOT_ZERO", missing_dates=missing)
            gaps.append({"stock_code": code, "reason": row["state"], "dates": missing})
        elif sum(amounts) == 0:
            row["state"] = "OBSERVED_ZERO_TURNOVER_NO_FINITE_LOAD"
            gaps.append({"stock_code": code, "reason": row["state"], "dates": days})
        else:
            average = sum(amounts) / 20
            row.update(
                state="CALCULATED",
                average_daily_amount_cny=str(average),
                load_days=str(stress * free / (participation * average)),
                amount_source_hashes=hashes,
            )
        rows.append(row)
    free_total = total - restricted
    if free_total <= 0:
        gaps.append({"reason": "NO_POSITIVE_FREE_STOCK_BOOK"})
    result = {
        "version": VERSION,
        "report_date": report_date,
        "sessions": days,
        "stress_fraction": str(stress),
        "participation_fraction": str(participation),
        "stock_amount_cny": str(total),
        "restricted_amount_cny": str(restricted),
        "free_stock_amount_cny": str(free_total),
        "securities": rows,
        "gaps": gaps,
        "state": "NOT_READY" if gaps else "UNSCORED_STOCK_BOOK_DIAGNOSTIC",
        "weighted_mean_load_days": None,
        "maximum_load_days": None,
        "scope": "FREE_SH_SZ_STOCK_BOOK_ONLY_NOT_LIQUIDATION_TIME_OR_TOTAL_ASSET_CAPACITY",
    }
    if not gaps:
        calculated = [r for r in rows if r["state"] == "CALCULATED"]
        result["weighted_mean_load_days"] = str(
            sum(decimal(r["free_value_cny"]) * decimal(r["load_days"]) for r in calculated) / free_total
        )
        result["maximum_load_days"] = str(max(decimal(r["load_days"]) for r in calculated))
    return result
