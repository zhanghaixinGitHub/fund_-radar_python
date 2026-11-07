"""南方官网完整日曲线核验；原件精度、重复末行和复权异常均保留审计。"""

import json
import re
from datetime import date
from decimal import Decimal, localcontext
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from app.services.fund_rating_market_review import normalized_text
from app.services.fund_rating_public_etf import _BENCHMARK_EXPRESSION_NORMALIZE, _benchmark_expression
from app.services.fund_rating_r5 import number


def _day(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{8}", value):
        raise ValueError("R5_SOUTHERN_DATE_FORMAT")
    return date.fromisoformat(value).isoformat()


def _request(source, digest: str, endpoint: str, params: dict) -> dict:
    """请求必须来自官网正常表单，代码、完整窗口和原字节一并绑定。"""
    ref = source.sources[digest]
    if (
        ref.get("archive_kind") == "DERIVED"
        or ref.get("url") != "https://www.southernfund.com/nfwebApi/fund/" + endpoint
        or ref.get("method") != "POST_FORM"
        or ref.get("params") != params
    ):
        raise ValueError("R5_SOUTHERN_REQUEST_SCOPE")
    raw = json.loads(source.raw[digest], parse_float=Decimal)
    if raw.get("code") != "ETS-5BP00000" or not isinstance(raw.get("data"), dict):
        raise ValueError("R5_SOUTHERN_PUBLISHER_RESPONSE")
    return raw["data"]


def _curve(rows: list, days: list, *, duplicate_end: bool) -> tuple[dict, list]:
    """只合并完全相同的单个末日重复行，冲突重复、缺日或非原公布精度均拒绝。"""
    if not isinstance(rows, list) or not rows:
        raise ValueError("R5_SOUTHERN_DAILY_COVERAGE")
    result, duplicates = {}, []
    for index, row in enumerate(rows):
        day, literal = _day(row["fdate"]), row["upratio"]
        if not isinstance(literal, str) or not re.fullmatch(r"-?\d+\.\d{2}", literal):
            raise ValueError("R5_SOUTHERN_PUBLISHED_PRECISION")
        value = number(literal)
        if value <= -100 or not days[0] <= day <= days[-1]:
            raise ValueError("R5_SOUTHERN_VALUE_OR_DATE_SCOPE")
        if day in result:
            if (
                not duplicate_end
                or duplicates
                or day != days[-1]
                or index != len(rows) - 1
                or _day(rows[index - 1]["fdate"]) != day
                or result[day] != value
            ):
                raise ValueError("R5_SOUTHERN_CONFLICTING_DUPLICATE")
            duplicates.append(day)
        result[day] = value
    if list(result) != sorted(result) or not set(days) <= result.keys():
        raise ValueError("R5_SOUTHERN_DAILY_COVERAGE")
    return result, duplicates


def compile_history(source, inputs: dict, history: dict) -> dict:
    """以双源单位净值、逐笔除息现金和官网收益曲线闭合真实费后回报。

    不修写供应商原净值。原复权曲线发生错误时仅在此纯计算中采用核准的
    本份额单位净值及核准除息分红再投资，并记录全部冲突日。拆分仍显式拒绝；
    两位百分数基准保持官网精度，不重建未公布小数。
    """
    refs, days, code = inputs["target_refs"], history["dates"], inputs["fund_code"]
    proof = source.json(inputs["southern_protocol"])
    if (
        proof.get("fund_code") != code
        or proof.get("as_of") != inputs["as_of_date"]
        or proof.get("first") != days[0]
        or proof.get("last") != days[-1]
        or not proof.get("frozen_at")
        or proof["frozen_at"] > inputs["verified_at"]
        or proof.get("issuer_percent_decimal_places") != 2
        or proof.get("method") != "COMPLETE_ISSUER_PUBLISHED_CURVE_NO_RECONSTRUCTION"
    ):
        raise ValueError("R5_SOUTHERN_FROZEN_PROTOCOL_SCOPE")
    page = source.raw[refs["benchmark-page"]].decode("utf-8")
    scripts = []
    if source.sources[refs["benchmark-page"]]["url"] != (
        "https://www.southernfund.com/new/personal-financing/detail.html?fundCode=" + code
    ):
        raise ValueError("R5_SOUTHERN_PAGE_IDENTITY")
    for key, prefix in (("benchmark-script", "detail-"), ("benchmark-main-script", "main-")):
        ref = source.sources[refs[key]]
        url = urlsplit(ref["url"])
        if (
            url.scheme != "https"
            or url.hostname != "www.southernfund.com"
            or not re.fullmatch(r"/js/" + prefix + r"[a-z0-9]+\.js", url.path)
            or url.path not in page
            or ref.get("archive_kind") == "DERIVED"
        ):
            raise ValueError("R5_SOUTHERN_PAGE_SCRIPT_IDENTITY")
        scripts.append(re.sub(r"\s+", "", source.raw[refs[key]].decode("utf-8")))
    detail, main = scripts
    if any(
        token not in detail
        for token in (
            'url:q.earningsMovements,type:"post",dataType:"json",data:n',
            "n=Object.assign({fundCode:a},n)",
            "i=o.thisfundlist",
            "f=o.brList",
            "x.push(t.upratio)",
            "w.push(t.upratio)",
            'seriesName1:"业绩基准："',
            "series:x,series1:w",
            'url:q.netValue,type:"post"',
            "o=n.navList,i=n.navLJList,r=n.dtlist",
            'url:q.dividends,type:"post"',
            "e=t.jjfhlist,a=t.splitlist",
        )
    ) or any(
        token not in main
        for token in (
            'earningsMovements:"/nfwebApi/fund/earningsMovements"',
            'netValue:"/nfwebApi/fund/netValue"',
            'dividends:"/nfwebApi/fund/dividends"',
        )
    ):
        raise ValueError("R5_SOUTHERN_COLUMN_SEMANTICS")
    expression = _benchmark_expression(
        normalized_text("\n".join(source.pages(refs["summary"]) + source.pages(refs["prospectus"])))
    )
    # 招募书用“标的指数”代词时，只按同一原件明确的法定指数名称展开。
    # 此处核文字等价，不用第三条市场对照曲线jzlist的代码替代合同基准。
    own_documents = normalized_text("\n".join(source.pages(refs["summary"]) + source.pages(refs["prospectus"])))
    index_name = inputs.get("index_name", "")
    if (
        expression.startswith("为标的指数")
        and index_name
        and "本基金的标的指数为" + index_name + "指数" in own_documents
    ):
        expression = expression.removeprefix("为").replace("标的指数", index_name + "指数", 1)
    page_text = normalized_text(BeautifulSoup(page, "html.parser").get_text(" ", strip=True))
    identity = _request(source, refs["issuer-identity"], "overreview", {"fundCode": code})["fund_info"]
    identity_expression = normalized_text(identity["yjbjjz"]).translate(_BENCHMARK_EXPRESSION_NORMALIZE)
    if (
        identity["fundCode"] != code
        or inputs.get("share_class") not in {"A", "C"}
        or normalized_text(identity["fundNameEx"])
        != normalized_text(inputs["legal_name"] + "(" + inputs["share_class"] + "类份额)")
        or expression not in (identity_expression, "本基金业绩比较基准:" + identity_expression)
        or "基金收益走势数据根据基金复权净值计算" not in page_text
        or 'url:q.overreview,type:"post",dataType:"json",data:{fundCode:a}' not in detail
        or _day(identity["contractValidDate"]) < "2009-01-01"
    ):
        raise ValueError("R5_SOUTHERN_CONTRACT_IDENTITY")
    params = {"fundCode": code, "startDate": days[0].replace("-", ""), "endDate": days[-1].replace("-", "")}
    data = _request(source, refs["benchmark"], "earningsMovements", params)
    if data.get("jzzscode") != proof["index_code"] or data.get("transdate"):
        raise ValueError("R5_SOUTHERN_INDEX_OR_ROUTE_CHANGE")
    own, duplicates = _curve(data["thisfundlist"], days, duplicate_end=True)
    benchmark, _ = _curve(data["brList"], days, duplicate_end=False)
    extra = sorted(own.keys() - set(days))
    if (
        sorted(benchmark.keys() - set(days)) != extra
        or extra != proof.get("published_nonvaluation_dates_retained")
        or duplicates != proof.get("identical_own_terminal_duplicates_retained")
    ):
        raise ValueError("R5_SOUTHERN_EXTRA_DATES_CHANGED")
    nav = _request(source, refs["issuer-nav"], "netValue", params)
    if (
        nav.get("startDate") != int(params["startDate"])
        or nav.get("endDate") != int(params["endDate"])
        or len(nav["dtlist"]) != len(nav["navList"])
        or len(nav["dtlist"]) != len(nav["navLJList"])
    ):
        raise ValueError("R5_SOUTHERN_NAV_RESPONSE_SCOPE")
    nav_days = list(map(_day, nav["dtlist"]))
    if nav_days != list(own):
        raise ValueError("R5_SOUTHERN_NAV_DAILY_COVERAGE")
    event_page_size = proof.get("event_page_size", 10)
    if event_page_size not in {10, 100}:
        raise ValueError("R5_SOUTHERN_EVENT_PAGE_SIZE_NOT_VERIFIED")
    events = _request(
        source,
        refs["issuer-events"],
        "dividends",
        {
            "fundCode": code,
            "startDate": "20090101",
            "endDate": params["endDate"],
            "curPage": 1,
            "pageSize": event_page_size,
        },
    )
    # 此适配要求完整单页历史，无论供应商是否漏记，都不能把官网事件排除在外。
    for key in ("jjfhlist", "splitlist"):
        table = events[key]
        if (
            table["total"] != len(table["list"])
            or table["pages"] not in (0, 1)
            or table["pageNum"] != 1
            or table["pageSize"] != event_page_size
        ):
            raise ValueError("R5_SOUTHERN_EVENT_PAGINATION_REQUIRED")
    if events["splitlist"]["list"]:
        raise ValueError("R5_SOUTHERN_SPLIT_ADAPTER_REQUIRED")
    # 分红本身不是缺失；逐笔核官网/授权源日期和每份现金。
    # 有分红时必须走经定报交叉核准的发行人原始增长曲线，不能混用旧现金再投试验口径。
    # 不支持拆分，未知事件不补0；100为官网表格的分页参数，不改变评分规则。
    provider = source.json(refs["provider"])
    cash_events, all_issuer_events = {}, {}
    for row in events["jjfhlist"]["list"]:
        ex_date = _day(row["f8"])
        if (
            ex_date >= days[0]
            and proof.get("dividend_return_method") != "DUAL_SOURCE_UNIT_NAV_CASH_REINVESTMENT_EX_DATE"
        ):
            raise ValueError("R5_SOUTHERN_DIVIDEND_ADAPTER_REQUIRED")
        if (
            row["f1"] != code
            or row["securitiesBo"]["f16"] != code
            or ex_date > days[-1]
            or ex_date in all_issuer_events
            or number(row["f6"]) <= 0
            or number(row["f7"]) < 0
            or number(row["f7f6"]) != number(row["f7"]) / number(row["f6"])
        ):
            raise ValueError("R5_SOUTHERN_DIVIDEND_IDENTITY_OR_AMOUNT")
        all_issuer_events[ex_date] = number(row["f7f6"])
        if ex_date > days[0]:
            if ex_date not in days:
                raise ValueError("R5_SOUTHERN_DIVIDEND_EX_DATE_NOT_IN_CALENDAR")
            cash_events[ex_date] = number(row["f7f6"])
    provider_events = {}
    for row in provider.get("dividend_records", []):
        ex_date = row["ex_date"]
        if days[0] < ex_date <= days[-1]:
            if (
                ex_date in provider_events
                or row["process_status"] != "实施"
                or not row.get("ann_date")
                or row["ann_date"] > inputs["as_of_date"]
            ):
                raise ValueError("R5_SOUTHERN_PROVIDER_EVENT_DUPLICATE_OR_STATUS")
            provider_events[ex_date] = number(row["cash_dividend"])
    if cash_events != provider_events:
        raise ValueError("R5_SOUTHERN_PROVIDER_EVENT_CONFLICT")
    if cash_events and proof.get("dividend_return_method") != "DUAL_SOURCE_UNIT_NAV_CASH_REINVESTMENT_EX_DATE":
        raise ValueError("R5_SOUTHERN_DIVIDEND_FROZEN_PROTOCOL_REQUIRED")
    rows = {row["nav_date"]: row for row in history["rows"]}
    units = dict(zip(nav_days, map(number, nav["navList"]), strict=True))
    accumulated = dict(zip(nav_days, map(number, nav["navLJList"]), strict=True))
    if proof.get("own_return_method") == "ISSUER_PUBLISHED_TOTAL_RETURN_CURVE":
        return _published_total_return(
            source, inputs, history, own, benchmark, units, accumulated, cash_events, duplicates, extra, proof
        )
    if cash_events:
        raise ValueError("R5_SOUTHERN_DIVIDEND_REQUIRES_ISSUER_PUBLISHED_GROWTH")
    with localcontext() as context:
        context.prec = 70
        differences, conflicts = [], []
        growth, reinvested, previous_day = Decimal(1), {}, None
        for day in days:
            if previous_day is not None:
                # 累计净值与单位净值变化的差额必须恰好等于当日每份分红。
                # 此守恒与官网双源核对让错误单位/漏分红不能被复权比例掩盖。
                cash = cash_events.get(day, Decimal(0))
                if accumulated[day] - accumulated[previous_day] != units[day] - units[previous_day] + cash:
                    raise ValueError("R5_SOUTHERN_DISTRIBUTION_ACCUMULATED_NAV_CONFLICT")
                growth *= (units[day] + cash) / units[previous_day]
            reinvested[day] = growth
            previous_day = day
        adjusted = list(map(number, history["adjusted_nav"]))
        for day, old in zip(days, adjusted, strict=True):
            row = rows[day]
            if (
                not row["ann_date"]
                or row["ann_date"] > inputs["as_of_date"]
                or units[day] <= 0
                or number(row["unit_nav"]) != units[day]
                or number(row["accumulated_nav"]) != accumulated[day]
            ):
                raise ValueError("R5_SOUTHERN_OWN_NAV_OR_CASH_CONFLICT")
            difference = abs((reinvested[day] - 1) * 100 - own[day])
            if difference > Decimal(".005"):
                raise ValueError("R5_SOUTHERN_OWN_PUBLISHED_RETURN_CONFLICT")
            differences.append(difference)
            if abs((old / adjusted[0] - 1) * 100 - own[day]) > Decimal(".005"):
                conflicts.append(day)
        if own[days[0]] != 0 or benchmark[days[0]] != 0:
            raise ValueError("R5_SOUTHERN_FIRST_NOT_NORMALIZED")
        history["adjusted_nav"] = [str(reinvested[day]) for day in days]
        history["benchmark_index"] = [str(1 + benchmark[day] / 100) for day in days]
    history["publisher_cross_check"] = {
        "method": proof["method"],
        "own_share_observations_checked": len(days),
        "issuer_percent_decimal_places": 2,
        "own_percent_half_quantum": ".005",
        "maximum_own_percent_difference": str(max(differences)),
        "published_nonvaluation_dates_retained": extra,
        "identical_own_terminal_duplicates_retained": duplicates,
        "unpublished_precision_reconstructed": False,
        "missing_values_filled": 0,
        "return_source": "DUAL_SOURCE_UNIT_NAV_CASH_REINVESTMENT_EX_DATE"
        if cash_events
        else "DUAL_SOURCE_OWN_UNIT_NAV_NO_WINDOW_CASH_OR_SPLIT_EVENT",
        "cash_events_reconciled": {key: str(value) for key, value in sorted(cash_events.items())},
        "issuer_event_rows_retained": len(all_issuer_events),
        "provider_adjusted_nav_conflict_dates": conflicts,
        "provider_adjusted_nav_source_rewritten": False,
    }
    return history


def _published_total_return(
    source, inputs, history, own, benchmark, units, accumulated, cash_events, duplicates, extra, proof
):
    """保留官网明示含分红再投的原曲线精度，不以推算复权值替换原公布观察。

    本份额每个单位/累计净值仍双源一致，分红逐笔一致；全年与半年度披露
    的C类收益和合同基准分别用原精度区间交叉核准。供应商复权异常只留痕。
    """
    days, refs = history["dates"], inputs["target_refs"]
    rows = {row["nav_date"]: row for row in history["rows"]}
    scripts = source.raw[refs["benchmark-script"]].decode("utf-8")
    if "复权净值为单位净值加分红再投资复权计算的净值" not in scripts:
        raise ValueError("R5_SOUTHERN_TOTAL_RETURN_SEMANTICS")
    report_checks, conflicts = [], []
    with localcontext() as context:
        context.prec = 70
        old = list(map(number, history["adjusted_nav"]))
        for index, day in enumerate(days):
            row = rows[day]
            if (
                not row["ann_date"]
                or row["ann_date"] > inputs["as_of_date"]
                or number(row["unit_nav"]) != units[day]
                or number(row["accumulated_nav"]) != accumulated[day]
            ):
                raise ValueError("R5_SOUTHERN_OWN_NAV_OR_CASH_CONFLICT")
            if index:
                previous = days[index - 1]
                if accumulated[day] - accumulated[previous] != units[day] - units[previous] + cash_events.get(
                    day, Decimal(0)
                ):
                    raise ValueError("R5_SOUTHERN_DISTRIBUTION_ACCUMULATED_NAV_CONFLICT")
            if abs((old[index] / old[0] - 1) * 100 - own[day]) > Decimal(".005"):
                conflicts.append(day)
        reports = {**inputs["target_annuals"], "current": refs["half"]}
        checks = [
            ("2024", "过去一年", "2023-12-31", "2024-12-31"),
            ("2025", "过去一年", "2024-12-31", "2025-12-31"),
            ("current", "过去三个月", "2026-03-31", "2026-06-30"),
            ("current", "过去六个月", "2025-12-31", "2026-06-30"),
            ("current", "过去一年", "2025-06-30", "2026-06-30"),
        ]
        for period, label, start, end in checks:
            text = normalized_text("\n".join(source.pages(reports[period])))
            table = re.split(r"3\.2\.1基金份额净值增长率", text)[-1].split("3.2.2")[0]
            section = re.search(r"50ETF联接C(.+?)(?:50ETF联接I|$)", table)
            record = re.search(label + r"(-?\d+\.\d+)%(-?\d+\.\d+)%(-?\d+\.\d+)%", section[1] if section else "")
            if not record:
                raise ValueError("R5_SOUTHERN_C_REPORT_RETURN_REQUIRED:" + period + label)
            for name, curve, reported in [("own", own, record[1]), ("benchmark", benchmark, record[3])]:
                # 原曲线两端百分数和报告收益各有0.005半量子；只核区间相交，
                # 不调整阈值、不恢复小数、不以区间中点替换任何观察。
                q = Decimal(".005")
                low = ((100 + curve[end] - q) / (100 + curve[start] + q) - 1) * 100
                high = ((100 + curve[end] + q) / (100 + curve[start] - q) - 1) * 100
                if high < number(reported) - q or low > number(reported) + q:
                    raise ValueError("R5_SOUTHERN_C_REPORT_RETURN_CONFLICT:" + period + name + label)
                report_checks.append(
                    {
                        "source_hash": reports[period],
                        "series": name,
                        "start": start,
                        "end": end,
                        "reported_percent": reported,
                        "curve_precision_low_percent": str(low),
                        "curve_precision_high_percent": str(high),
                    }
                )
        if own[days[0]] != 0 or benchmark[days[0]] != 0:
            raise ValueError("R5_SOUTHERN_FIRST_NOT_NORMALIZED")
        history["adjusted_nav"] = [str(1 + own[day] / 100) for day in days]
        history["benchmark_index"] = [str(1 + benchmark[day] / 100) for day in days]
    history["publisher_cross_check"] = {
        "method": proof["method"],
        "return_source": "ISSUER_PUBLISHED_TOTAL_RETURN_CURVE",
        "own_share_observations_checked": len(days),
        "issuer_percent_decimal_places": 2,
        "published_nonvaluation_dates_retained": extra,
        "identical_own_terminal_duplicates_retained": duplicates,
        "cash_events_reconciled": {key: str(value) for key, value in sorted(cash_events.items())},
        "report_period_return_checks": report_checks,
        "provider_adjusted_nav_conflict_dates": conflicts,
        "provider_adjusted_nav_source_rewritten": False,
        "unpublished_precision_reconstructed": False,
        "missing_values_filled": 0,
    }
    return history
