"""广发真实累计基准与本C净值核准；共同尺度构造不冒充原公布值。"""

import re
from datetime import date
from decimal import Decimal
from urllib.parse import parse_qs, urlsplit

from app.services.fund_rating_benchmark_bridge import reconstruct
from app.services.fund_rating_market_review import normalized_text
from app.services.fund_rating_r5 import number


def compile_history(source, inputs: dict, history: dict) -> dict:
    """验证实际代码、合同式、字面精度和全部估值日，再核固定95/5参照。"""
    from app.services.fund_rating_direct_index import _table
    from app.services.fund_rating_public_etf import _BENCHMARK_EXPRESSION_NORMALIZE, _benchmark_expression

    refs, proof = inputs["target_refs"], inputs["gf_benchmark_reconstruction"]
    frozen = source.json(proof["protocol_hash"])
    if (frozen["fund_code"] != inputs["fund_code"] or frozen["as_of"] != inputs["as_of_date"]
            or frozen["index_code"] != inputs["index"]["code"] or frozen["candidate_cash_rate"] != ".0035"
            or frozen["publisher_level_half_quantum"] != ".00005"
            or frozen["index_close_half_quantum"] != ".00005" or frozen["frozen_at"] > inputs["verified_at"]):
        raise ValueError("R5_GF_FROZEN_PROTOCOL_SCOPE")
    if proof["annual_cash_rate"] != ".0035" or "活期存款0.350.35" not in source.plain(proof["rate_source"]):
        raise ValueError("R5_GF_RATE_ORIGINAL_REQUIRED")
    uri = urlsplit(source.sources[refs["benchmark"]]["url"])
    if (uri.scheme != "https" or uri.hostname != "www.gffunds.com.cn"
            or uri.path != "/api/v1/funds/yield-history.shtml"
            or parse_qs(uri.query).get("fundcode") != [inputs["fund_code"]]):
        raise ValueError("R5_GF_PUBLISHER_REQUEST_SCOPE")
    expression = _benchmark_expression(normalized_text("\n".join(
        source.pages(refs["summary"]) + source.pages(refs["prospectus"]))))
    page = source.plain(refs["benchmark-page"]).translate(_BENCHMARK_EXPRESSION_NORMALIZE)
    if expression not in page:
        raise ValueError("R5_GF_PUBLISHER_CONTRACT_EXPRESSION")
    raw = source.json(refs["benchmark"])
    rows = raw.get("CREATE")
    if not isinstance(rows, list) or not rows or len({r["NAVDATE"] for r in rows}) != len(rows):
        raise ValueError("R5_GF_PUBLISHER_DUPLICATE_OR_MISSING_DAYS")
    if any(r["FUNDCODE"] != inputs["fund_code"] or r["IDXTYPE"] != "yield_his" for r in rows):
        raise ValueError("R5_GF_PUBLISHER_CODE_OR_PERIOD")
    mapped = {str(date.fromisoformat(r["NAVDATE"])): r for r in rows}
    days = history["dates"]
    if not set(days) <= mapped.keys():
        raise ValueError("R5_GF_PUBLISHER_DAILY_COVERAGE")
    nav = dict(zip(days, map(number, history["adjusted_nav"]), strict=True))
    observed = {}
    for day in days:
        row = mapped[day]
        if any(not isinstance(row[k], str) or not re.fullmatch(r"-?\d+\.\d{2}%", row[k])
               for k in ("YIELD", "YIELD_BENCHMARK")):
            raise ValueError("R5_GF_PUBLISHER_PERCENT_PRECISION")
        if abs(1 + number(row["YIELD"][:-1]) / 100 - nav[day]) > Decimal(".00005"):
            raise ValueError("R5_GF_OWN_ADJUSTED_NAV_CONFLICT")
        observed[day] = str(1 + number(row["YIELD_BENCHMARK"][:-1]) / 100)
    prices = {}
    for row in _table(source.json(proof["index_daily"])):
        day = str(date.fromisoformat(row["trade_date"]))
        if row["ts_code"] != inputs["index"]["code"] or day in prices:
            raise ValueError("R5_GF_INDEX_CODE_OR_DUPLICATE")
        prices[day] = row["close"]
    checked = reconstruct(days, prices, observed, annual_cash_rate=proof["annual_cash_rate"],
                          expected_missing_dates=[], publisher_level_half_quantum=".00005")
    history["benchmark_index"] = checked.pop("levels")
    history["publisher_cross_check"] = {**checked, "own_share_observations_checked": len(days),
                                       "original_outside_window_rows_retained": len(rows) - len(days)}
    return history
