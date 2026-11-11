"""国泰官网同份额、同起点的完整逐日基准；只读原件，不补点或拼接短窗。"""

import json
import re
from datetime import datetime, timedelta
from decimal import Decimal, localcontext
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from app.services.fund_rating_market_review import normalized_text
from app.services.fund_rating_public_etf import _BENCHMARK_EXPRESSION_NORMALIZE, _benchmark_expression
from app.services.fund_rating_r5 import number


def _day(value: str) -> str:
    """官网时间戳为北京时间零点的UTC表示；不按UTC日期误移一天。"""
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T16:00:00\.000Z", value):
        raise ValueError("R5_GT_PUBLISHER_TIMESTAMP")
    return (datetime.fromisoformat(value.replace("Z", "+00:00")) + timedelta(hours=8)).date().isoformat()


def compile_history(source, inputs: dict, history: dict) -> dict:
    """核匿名公开请求、页面列含义和全部估值日，采用官网实际公布的基准。

    两位百分数原值保留原精度，不声称恢复了未公开的小数位。基金列只作本
    份额核对，真正费后回报使用已核净值；另列沪深300不能充当合同基准。
    """
    refs, code = inputs["target_refs"], inputs["fund_code"]
    proof = source.json(inputs["gt_protocol"])
    days = history["dates"]
    if (
        proof.get("fund_code") != code
        or proof.get("as_of") != inputs["as_of_date"]
        or proof.get("frozen_at", "") > inputs["verified_at"]
        or proof.get("first") != days[0]
        or proof.get("last") != days[-1]
        or proof.get("issuer_percent_decimal_places") != 2
        or proof.get("method") != "COMPLETE_ISSUER_PUBLISHED_CURVE_NO_RECONSTRUCTION"
    ):
        raise ValueError("R5_GT_FROZEN_PROTOCOL_SCOPE")
    reference = source.sources[refs["benchmark"]]
    request = reference.get("request", {})
    params = request.get("params", {})
    if (
        reference.get("archive_kind") == "DERIVED"
        or reference["url"] != "https://e.gtfund.com/Etrade/Public/cochin/info.fund.growth_plot"
        or request.get("api") != "info.fund.growth_plot"
        or params.get("fundCode") != code
        or params.get("order") != "asc"
        or _day(params["start"]) != days[0]
        or _day(params["end"]) != days[-1]
    ):
        raise ValueError("R5_GT_PUBLISHER_REQUEST_SCOPE")
    page_ref = source.sources[refs["benchmark-page"]]
    script_ref = source.sources[refs["benchmark-script"]]
    page = source.load(refs["benchmark-page"]).decode("utf-8")
    script = source.load(refs["benchmark-script"]).decode("utf-8")
    script_url = urlsplit(script_ref["url"])
    if (
        page_ref["url"] != "https://e.gtfund.com/etrade/Jijin/view/id/" + code
        or not re.search(r"fundcode:\s*'" + re.escape(code) + r"'", page)
        or not re.search(r"hideBenchmark:\s*null", page)
        or script_url.hostname != "e.gtfund.com"
        or script_url.scheme != "https"
        or script_url.path != "/Public/static/etrade/1/dist/js/jijin_view.js"
        or script_url.path + "?" + script_url.query not in page
        or '业绩比较基准收益率</span>：<span class="J_chart_yeji1">' not in page
    ):
        raise ValueError("R5_GT_PAGE_OR_SCRIPT_IDENTITY")
    compact = re.sub(r"\s+", "", script)
    if any(
        token not in compact
        for token in (
            'cochin("info.fund.growth_plot",{start:start,end:end,fundCode:yejichart.fundcode,order:"asc"}',
            "chartDataYeji.Y.fund=res.map(function(o){returno[1];});",
            "chartDataYeji.Y.company=res.map(function(o){returno[2];});",
            "updateLegend(benchmark,'.J_chart_yeji1');",
        )
    ):
        raise ValueError("R5_GT_PUBLISHER_COLUMN_SEMANTICS")
    expression = _benchmark_expression(
        normalized_text("\n".join(source.pages(refs["summary"]) + source.pages(refs["prospectus"])))
    )
    page_text = normalized_text(BeautifulSoup(page, "html.parser").get_text(" ", strip=True)).translate(
        _BENCHMARK_EXPRESSION_NORMALIZE
    )
    if expression not in page_text:
        raise ValueError("R5_GT_PUBLISHER_CONTRACT_EXPRESSION")
    raw = json.loads(source.load(refs["benchmark"]), parse_float=Decimal)
    if not isinstance(raw, list) or not raw or any(not isinstance(row, list) or len(row) != 3 for row in raw):
        raise ValueError("R5_GT_PUBLISHER_RESPONSE")
    observed = {}
    for timestamp, fund, benchmark in raw:
        day = _day(timestamp)
        if day in observed or not days[0] <= day <= days[-1]:
            raise ValueError("R5_GT_PUBLISHER_DUPLICATE_OR_DATE_SCOPE")
        values = [number(fund), number(benchmark)]
        if any(value.as_tuple().exponent < -2 or value <= -100 for value in values):
            raise ValueError("R5_GT_PUBLISHER_PRECISION_OR_DOMAIN")
        observed[day] = values
    if list(observed) != sorted(observed) or not set(days) <= observed.keys():
        raise ValueError("R5_GT_PUBLISHER_DAILY_COVERAGE")
    extra = sorted(observed.keys() - set(days))
    if extra != proof.get("published_nonvaluation_dates_retained"):
        raise ValueError("R5_GT_PUBLISHER_EXTRA_DATES_CHANGED")
    provider = source.json(refs["provider"])
    if provider.get("dividend_records"):
        raise ValueError("R5_GT_SHARE_DIVIDEND_ADAPTER_REQUIRED")
    rows = {row["nav_date"]: row for row in history["rows"]}
    with localcontext() as context:
        context.prec = 70
        nav = dict(zip(days, map(number, history["adjusted_nav"]), strict=True))
        differences = []
        for day in days:
            row = rows[day]
            if (
                row["ann_date"] > inputs["as_of_date"]
                or len({number(row[key]) for key in ("unit_nav", "accumulated_nav", "adjusted_nav")}) != 1
            ):
                raise ValueError("R5_GT_OWN_SHARE_ADJUSTMENT_OR_FUTURE_NAV")
            difference = abs((nav[day] / nav[days[0]] - 1) * 100 - observed[day][0])
            if difference > Decimal(".005"):
                raise ValueError("R5_GT_OWN_SHARE_PUBLISHED_NAV_CONFLICT")
            differences.append(difference)
        if observed[days[0]] != [0, 0]:
            raise ValueError("R5_GT_PUBLISHER_FIRST_DAY_NOT_NORMALIZED")
        history["benchmark_index"] = [str(1 + observed[day][1] / 100) for day in days]
    history["publisher_cross_check"] = {
        "method": proof["method"],
        "own_share_observations_checked": len(days),
        "issuer_percent_decimal_places": 2,
        "own_percent_half_quantum": ".005",
        "maximum_own_percent_difference": str(max(differences)),
        "published_nonvaluation_dates_retained": extra,
        "unpublished_precision_reconstructed": False,
        "missing_values_filled": 0,
    }
    return history
