"""第二版区分基金比较基准与历史披露个股的成交价缺口。"""


def baseline_ready(data: dict) -> bool:
    """不改旧完整性结果；个股缺口交给逐周期的重要性与敏感性审阅。

    仍要求真实本基金基准净值和已核对照行情。这里不产生方向，不把停牌
    当作零涨跌，也不允许用目标ETF净值替代本基金净值。
    """
    context = data.get("time_context", {})
    if data.get("valuation_status") is not None and data["valuation_status"].get("ready") is not True:
        return False
    if context.get("baseline_complete") is True:
        return True
    base = data.get("window", {}).get("base_nav_date")
    markets = context.get("markets", {})
    return bool(
        base
        and context.get("baseline_date") == base
        and context.get("nav_status") == "READY"
        and context.get("latest_nav_date") == base
        and max((str(row.get("nav_date", "")) for row in data.get("nav", [])), default=None) == base
        and markets
        and set(markets) == set(data.get("market", {}))
        and all(
            row.get("status") in {"READY", "READY_LAST_SESSION"}
            and data["market"][code].get("change_pct") is not None
            and (
                data["market"][code].get("date") == base
                or (row.get("status") == "READY_LAST_SESSION" and row.get("valuation_reference"))
            )
            for code, row in markets.items()
        )
    )
