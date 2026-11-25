"""第二版区分基金比较基准与历史披露个股的成交价缺口。"""


def baseline_ready(data: dict) -> bool:
    """不改旧完整性结果；个股缺口交给逐周期的重要性与敏感性审阅。

    仍要求真实本基金基准净值和已核对照行情。这里不产生方向，不把停牌
    当作零涨跌，也不允许用目标ETF净值替代本基金净值。
    """
    from app.services.prediction_window_v2 import is_midnight

    context = data.get("time_context", {})
    if is_midnight(data.get("window")):
        # 此处只放行有已验日期的资料进入逐周期实质判断；缺行情仍由每周期说明影响。
        return bool(data.get("forecast_calendar_verified") and not data.get("reference_nav_date"))
    if data.get("valuation_status") is not None and data["valuation_status"].get("ready") is not True:
        return False
    base = data.get("window", {}).get("base_nav_date")
    reference = data.get("reference_nav_date")
    if reference is not None:
        from app.services.fund_prediction_nav_identity import previous_qdii_nav_date

        if (
            not base
            or reference != previous_qdii_nav_date(data.get("nav", []), base, data)
            or context.get("nav_status") != "READY_PREVIOUS_SESSION"
            or context.get("latest_nav_date") != reference
            or context.get("baseline_date") != base
        ):
            return False
    if context.get("baseline_complete") is True and reference is None:
        return True
    markets = context.get("markets", {})
    return bool(
        base
        and context.get("baseline_date") == base
        and context.get("nav_status") == ("READY_PREVIOUS_SESSION" if reference else "READY")
        and context.get("latest_nav_date") == (reference or base)
        and max((str(row.get("nav_date", "")) for row in data.get("nav", [])), default=None) == (reference or base)
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
