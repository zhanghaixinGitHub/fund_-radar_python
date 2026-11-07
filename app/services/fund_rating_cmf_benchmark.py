"""招商公开完整日曲线原件核准；按原精度验证合同参照，保持公布值与构造值区别。"""

import re
from datetime import date
from decimal import Decimal, localcontext

from app.services.fund_rating_benchmark_bridge import reconstruct_normalized_windows
from app.services.fund_rating_market_review import normalized_text
from app.services.fund_rating_r5 import number


def compile_history(source, inputs: dict, history: dict) -> dict:
    """仅接纳本C份额真实请求、合同表述、全日历和本份额净值均闭合的响应。"""
    from app.services.fund_rating_direct_index import _table
    from app.services.fund_rating_public_etf import _BENCHMARK_EXPRESSION_NORMALIZE, _benchmark_expression

    proof, refs = inputs["cmf_benchmark_reconstruction"], inputs["target_refs"]
    frozen = source.json(proof["protocol_hash"])
    if (frozen["fund_code"] != inputs["fund_code"] or frozen["as_of"] != inputs["as_of_date"]
            or frozen["index_code"] != inputs["index"]["code"] or frozen["issuer_percent_decimal_places"] != 2
            or frozen["issuer_level_half_quantum"] != ".00005" or frozen["index_close_half_quantum"] != ".00005"
            or frozen["frozen_at"] > inputs["verified_at"]):
        raise ValueError("R5_CMF_BRIDGE_FROZEN_SCOPE")
    rate = proof["annual_cash_rate"]
    if rate != ".0035" or "活期存款0.350.35" not in source.plain(proof["rate_source"]):
        raise ValueError("R5_CMF_BRIDGE_RATE_ORIGINAL")
    reference, raw = source.sources[refs["benchmark"]], source.json(refs["benchmark"])
    request = reference.get("params", {})
    if (reference["url"] != "https://www.cmfchina.com/ws-business-server/fund/getIncomeList"
            or request.get("productCode") != inputs["fund_code"] or request.get("startDate") != history["dates"][0]
            or request.get("endDate") != inputs["as_of_date"] or raw.get("code") != 0
            or not isinstance(raw.get("data"), list) or not raw["data"]):
        raise ValueError("R5_CMF_BRIDGE_REQUEST_OR_RESPONSE")
    rows = raw["data"]
    if len({row["date"] for row in rows}) != len(rows):
        raise ValueError("R5_CMF_BRIDGE_DUPLICATE_DAY")
    expression = _benchmark_expression(normalized_text("\n".join(
        source.pages(refs["summary"]) + source.pages(refs["prospectus"])
    )))
    if normalized_text(rows[0].get("indexName", "")).translate(_BENCHMARK_EXPRESSION_NORMALIZE) != expression:
        raise ValueError("R5_CMF_BRIDGE_CONTRACT_EXPRESSION")
    prices = {}
    for row in _table(source.json(proof["index_daily"])):
        day = str(date.fromisoformat(row["trade_date"]))
        if row["ts_code"] != inputs["index"]["code"] or day in prices:
            raise ValueError("R5_CMF_BRIDGE_INDEX_IDENTITY_OR_DUPLICATE")
        prices[day] = row["close"]
    if set(prices) != set(history["dates"]):
        raise ValueError("R5_CMF_BRIDGE_INDEX_CALENDAR")
    nav = {row["nav_date"]: number(row["adjusted_nav"]) for row in history["rows"]}
    first, observed, extras = history["dates"][0], {}, []
    with localcontext() as context:
        context.prec = 70
        for row in rows:
            day = str(date.fromisoformat(row["date"]))
            if day not in nav or not first <= day <= inputs["as_of_date"]:
                raise ValueError("R5_CMF_BRIDGE_DATE_OR_NAV")
            if any(not re.fullmatch(r"-?\d+\.\d{2}", row[key]) for key in ("navYield", "hs300Yield")):
                raise ValueError("R5_CMF_BRIDGE_PUBLISHED_PRECISION")
            if row.get("indexName") and normalized_text(row["indexName"]).translate(
                _BENCHMARK_EXPRESSION_NORMALIZE
            ) != expression:
                raise ValueError("R5_CMF_BRIDGE_CONTRACT_EXPRESSION")
            if abs(1 + number(row["navYield"]) / 100 - nav[day] / nav[first]) > Decimal(".00005"):
                raise ValueError("R5_CMF_BRIDGE_OWN_SHARE_NAV_CONFLICT")
            if day in prices:
                observed[day] = str(1 + number(row["hs300Yield"]) / 100)
            else:
                extras.append(day)
    result = reconstruct_normalized_windows(history["dates"], prices,
        [{"first": first, "last": inputs["as_of_date"], "levels": observed}], annual_cash_rate=rate)
    history["benchmark_index"] = result.pop("levels")
    history["publisher_cross_check"] = {**result, "own_share_observations_checked": len(rows),
                                       "published_nonvaluation_dates_retained": extras}
    return history
