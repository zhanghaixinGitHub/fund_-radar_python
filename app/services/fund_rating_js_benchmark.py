"""嘉实官网正常公开短窗基准原件核准；不联网、不拼接舍入值、不评分。"""

import re
from datetime import date
from decimal import Decimal

from app.services.fund_rating_benchmark_bridge import reconstruct_normalized_windows
from app.services.fund_rating_market_review import normalized_text
from app.services.fund_rating_r5 import number


def compile_history(source, inputs: dict, history: dict) -> dict:
    """真实代码/官网产品ID、合同文字和本C份额净值逐窗核准，再核合同构造。"""
    from app.services.fund_rating_direct_index import _table
    from app.services.fund_rating_public_etf import _BENCHMARK_EXPRESSION_NORMALIZE, _benchmark_expression

    refs = inputs["target_refs"]
    proof = inputs["js_benchmark_reconstruction"]
    frozen = source.json(proof["protocol_hash"])
    if (frozen["fund_code"] != inputs["fund_code"] or frozen["as_of"] != inputs["as_of_date"]
            or frozen["index_code"] != inputs["index"]["code"] or frozen["issuer_percent_decimal_places"] != 2
            or frozen["issuer_level_half_quantum"] != ".00005" or frozen["index_close_half_quantum"] != ".00005"
            or frozen["frozen_at"] > inputs["verified_at"]):
        raise ValueError("R5_JS_BRIDGE_PROTOCOL_SCOPE")
    rate = proof["annual_cash_rate"]
    if rate != ".0035" or "活期存款0.350.35" not in source.plain(proof["rate_source"]):
        raise ValueError("R5_JS_BRIDGE_RATE_ORIGINAL")
    identity = source.json(proof["product_identity"])
    products = [row for row in identity.get("results", []) if row.get("product_code") == inputs["fund_code"]]
    if identity.get("error_no") != "0" or len(products) != 1 or products[0]["product_id"] != proof["product_id"]:
        raise ValueError("R5_JS_BRIDGE_PRODUCT_IDENTITY")
    if normalized_text(products[0]["product_name"]) != normalized_text(inputs["legal_name"]):
        raise ValueError("R5_JS_BRIDGE_LEGAL_IDENTITY")
    expression = _benchmark_expression(normalized_text("\n".join(
        source.pages(refs["summary"]) + source.pages(refs["prospectus"])
    )))
    prices = {}
    for row in _table(source.json(proof["index_daily"])):
        day = str(date.fromisoformat(row["trade_date"]))
        if row["ts_code"] != inputs["index"]["code"] or day in prices:
            raise ValueError("R5_JS_BRIDGE_INDEX_IDENTITY_OR_DUPLICATE")
        prices[day] = row["close"]
    days = history["dates"]
    if set(prices) != set(days):
        raise ValueError("R5_JS_BRIDGE_INDEX_CALENDAR")
    nav = {row["nav_date"]: number(row["adjusted_nav"]) for row in history["rows"]}
    windows, extra_dates, total = [], [], 0
    for digest in proof["windows"]:
        reference, raw = source.sources[digest], source.json(digest)
        request = reference.get("request_params", {})
        if (reference["url"] != "https://www.jsfund.cn/servlet/json" or request.get("funcNo") != "741032"
                or request.get("product_code") != inputs["fund_code"]
                or request.get("product_id") != proof["product_id"]
                or raw.get("error_no") != "0" or not isinstance(raw.get("results"), list)):
            raise ValueError("R5_JS_BRIDGE_PUBLISHER_REQUEST_SCOPE")
        rows = raw["results"]
        if not rows or len({row["date"] for row in rows}) != len(rows):
            raise ValueError("R5_JS_BRIDGE_PUBLISHER_DATES")
        if normalized_text(rows[0].get("index_name", "")).translate(_BENCHMARK_EXPRESSION_NORMALIZE) != expression:
            raise ValueError("R5_JS_BRIDGE_FIRST_CONTRACT_EXPRESSION")
        first = str(date.fromisoformat(request["start_date"]))
        last = str(date.fromisoformat(request["end_date"]))
        observed = {}
        for row in rows:
            day = str(date.fromisoformat(row["date"]))
            if not first <= day <= last or day not in nav:
                raise ValueError("R5_JS_BRIDGE_PUBLISHER_DATE_OR_NAV")
            if any(not re.fullmatch(r"-?\d+\.\d{2}", row[key]) for key in ("nav_yield", "hs300_yield")):
                raise ValueError("R5_JS_BRIDGE_PUBLISHED_PRECISION_CHANGED")
            if row.get("index_name") and normalized_text(row["index_name"]).translate(
                _BENCHMARK_EXPRESSION_NORMALIZE
            ) != expression:
                raise ValueError("R5_JS_BRIDGE_CONTRACT_EXPRESSION")
            actual_nav = 1 + number(row["nav_yield"]) / 100
            if abs(actual_nav - nav[day] / nav[first]) > Decimal(".00005"):
                raise ValueError("R5_JS_BRIDGE_OWN_SHARE_NAV_CONFLICT")
            if day in prices:
                observed[day] = str(1 + number(row["hs300_yield"]) / 100)
                total += 1
            else:
                extra_dates.append(day)
        windows.append({"first": first, "last": last, "levels": observed})
    checked = reconstruct_normalized_windows(days, prices, windows, annual_cash_rate=rate)
    history["benchmark_index"] = checked.pop("levels")
    history["publisher_cross_check"] = {**checked, "own_share_observations_checked": total,
                                       "published_nonvaluation_dates_retained": sorted(set(extra_dates))}
    return history
