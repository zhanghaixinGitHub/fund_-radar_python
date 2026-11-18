"""R5月评窗口的滚动核验；交易日必须来自已绑定的真实日历原件。"""

from datetime import date, timedelta
from calendar import monthrange
from decimal import Decimal as D, localcontext
from pathlib import Path
import re
from copy import deepcopy


def rolling_window(calendar_days, as_of):
    """取截止月及三年前同月最后一个真实交易日，不造普通工作日。"""
    cutoff = date.fromisoformat(as_of)
    days = sorted(set(date.fromisoformat(value) for value in calendar_days))
    if not days or len(days) != len(calendar_days):
        raise ValueError("R5_MONTHLY_OFFICIAL_CALENDAR_DUPLICATE_OR_EMPTY")
    ending = [day for day in days if day.year == cutoff.year and day.month == cutoff.month]
    beginning = [day for day in days if day.year == cutoff.year - 3 and day.month == cutoff.month]
    if not beginning or not ending or max(ending) != cutoff:
        raise ValueError("R5_MONTHLY_OFFICIAL_MONTH_END_REQUIRED")
    start = max(beginning)
    selected = [day for day in days if start <= day <= cutoff]
    month_ends = [max(day for day in selected if (day.year, day.month) == month)
                  for month in sorted({(day.year, day.month) for day in selected})]
    ids = [day.year * 12 + day.month for day in month_ends]
    if len(ids) != 37 or ids != list(range(ids[0], ids[0] + 37)):
        raise ValueError("R5_MONTHLY_COMPLETE_THREE_YEAR_CALENDAR_REQUIRED")
    return list(map(str, selected)), list(map(str, month_ends))


def validate_window(days, month_ends, calendar_days, as_of):
    expected_days, expected_months = rolling_window(calendar_days, as_of)
    if days != expected_days or month_ends != expected_months:
        raise ValueError("R5_MONTHLY_SOURCE_CALENDAR_WINDOW_CONFLICT")
    return expected_days, expected_months


def annual_disclosure_range(as_of):
    """发行人事项沿原一年披露口径滚动，闭合到真实估值截止日。"""
    cutoff = date.fromisoformat(as_of)
    start = (cutoff.replace(day=1, year=cutoff.year - 1) + timedelta(days=32)).replace(day=1)
    return str(start), str(cutoff)


def three_year_disclosure_start(as_of):
    cutoff = date.fromisoformat(as_of)
    year = cutoff.year - 3
    return str(date(year, cutoff.month, monthrange(year, cutoff.month)[1]))


def compile_same_index_performance(inputs, *, root):
    """同指数C联接从真实净值、分红、指数与官网基准复算，缺同行即失败。

    方法保持原95/5合同、365.25年化、252日风险与同指数同行均值稳定性；
    这里输出原始观察，不输出分项分或总分，更不拿旧候选分作输入。
    """
    from app.services.fund_rating_thematic_source import RawArchive, plain, require
    from app.services.fund_rating_thematic_performance import fund_wealth
    from app.services.fund_rating_direct_index_math import performance

    source = RawArchive(inputs, Path(root))
    cohort = source.json(inputs["cohort_ref"])
    code = inputs["fund_code"]
    peers = cohort["peers"]
    codes = [code, *peers]
    require(code == "008888" and peers == ["012630", "012970"], "MONTHLY_SAME_INDEX_FROZEN_COHORT")
    require(set(inputs["funds"]) == set(codes), "MONTHLY_SAME_INDEX_ALL_SHARES")
    calendar = []
    for row in source.provider_rows(inputs["calendar_ref"], ["exchange", "cal_date", "is_open"]):
        require(row["exchange"] == "SSE" and row["is_open"] == 1, "MONTHLY_SAME_INDEX_CALENDAR")
        calendar.append(str(date.fromisoformat(row["cal_date"])))
    days, months = rolling_window(sorted(calendar), inputs["as_of_date"])
    require(sorted(calendar) == days, "MONTHLY_SAME_INDEX_EXACT_CALENDAR")
    for own in codes:
        sha = inputs["funds"][own]["contract_report_ref"]
        source.original(sha)
        body = plain(source.body(sha))
        require(own in body[:20000] and all(word in body for word in ["国证半导体芯片指数", "95%", "5%", "活期"]),
                "MONTHLY_SAME_INDEX_CURRENT_C_CONTRACT")
    index = {}
    for row in source.provider_rows(inputs["index_ref"], ["ts_code", "trade_date", "close"]):
        require(row["ts_code"] == "980017.SZ", "MONTHLY_SAME_INDEX_INDEX_IDENTITY")
        day = str(date.fromisoformat(row["trade_date"]))
        require(day not in index, "MONTHLY_SAME_INDEX_INDEX_DUPLICATE")
        index[day] = D(str(row["close"]))
    require(set(index) == set(days) and min(index.values()) > 0, "MONTHLY_SAME_INDEX_INDEX_WINDOW")
    rate_text = plain(source.body(inputs["cash_rate_ref"]))
    require("活期存款" in rate_text and "0.35" in rate_text, "MONTHLY_SAME_INDEX_CASH_RATE")
    directory = plain(source.body(inputs["cash_rate_directory_ref"]))
    require("现行利率水平" in directory and "2015" in directory, "MONTHLY_SAME_INDEX_CURRENT_RATE_DIRECTORY")
    source.original(inputs["official_benchmark_ref"])
    official = source.json(inputs["official_benchmark_ref"])
    require(len(official["ShowData"]) == len(official["JiZhunName"]), "MONTHLY_SAME_INDEX_OFFICIAL_SCHEMA")
    published = {day: 1 + D(str(value)) / 100 for day, value in zip(official["ShowData"], official["JiZhunName"], strict=True)}
    with localcontext() as context:
        context.prec = 70
        benchmark, bounds = [D(1)], [(D(1), D(1))]
        epsilon = D(".00005")
        for before, day in zip(days, days[1:]):
            cash = D(".05") * D(".0035") * (date.fromisoformat(day) - date.fromisoformat(before)).days / 365
            benchmark.append(benchmark[-1] * (1 + D(".95") * (index[day] / index[before] - 1) + cash))
            lo, hi = bounds[-1]
            bounds.append((lo * (1 + D(".95") * ((index[day] - epsilon) / (index[before] + epsilon) - 1) + cash),
                           hi * (1 + D(".95") * ((index[day] + epsilon) / (index[before] - epsilon) - 1) + cash)))
        fitted = [(n, day) for n, day in enumerate(days) if day in published]
        require(len(fitted) > 1, "MONTHLY_SAME_INDEX_OFFICIAL_OVERLAP")
        lower = max((published[day] - epsilon) / bounds[n][1] for n, day in fitted)
        upper = min((published[day] + epsilon) / bounds[n][0] for n, day in fitted)
        require(lower <= upper, "MONTHLY_SAME_INDEX_BENCHMARK_PRECISION_CONFLICT")
        wealth, audits, values = {}, {}, {}
        for own in codes:
            wealth[own], audits[own] = fund_wealth(source, inputs["funds"][own], own, days)
            values[own] = performance(days, months, wealth[own], benchmark)
        nodes = {}
        products = {own: "CN_PUBLIC_FUND:" + cohort["relationships_to_verify"][own][0] for own in codes}
        for metric in ["return_1y", "return_3y", "extra_drawdown", "extra_downside", "tracking_error"]:
            nodes[metric] = dict(kind="rank", direction="higher" if metric.startswith("return_") else "lower",
                                 target=values[code][metric], expected_products=[products[p] for p in peers],
                                 peers=[dict(product_id=products[p], value=values[p][metric]) for p in peers])
        for name in ["monthly_success", "annual_success"]:
            rows = []
            for n, row in enumerate(values[code][name]):
                rows.append(dict(start=row["start"], end=row["end"], target=row["target"],
                                 reference=str(sum(D(values[p][name][n]["target"]) for p in peers) / len(peers))))
            nodes[name] = dict(kind="windows", expected_windows=len(rows), rows=rows)
    refs = sorted(source.used)
    for node in nodes.values():
        node["source_refs"] = refs
    return dict(observations=nodes, dates=days, month_ends=months, fund_audits=audits,
                metrics=values, used_source_refs=refs,
                benchmark_audit=dict(official_days=len(fitted), reconstructed_days=len(days),
                                     official_last_observed=max(day for _, day in fitted),
                                     reconstructed_after_official=[day for day in days if day > max(d for _, d in fitted)],
                                     joint_precision_interval=[str(lower), str(upper)],
                                     formula="B_t=B_prev*(1+0.95*(I_t/I_prev-1)+0.05*0.0035*calendar_day_gap/365)"))


def licensed_share_returns(source, nav_ref, dividend_ref, ts_code, days, months,
                           *, selected_horizons=("return_1y", "return_3y")):
    """沿既有披露净值增长公式逐份额复算端点；周频缺点不得插值或删同行。"""
    from app.services.fund_rating_thematic_source import iso, require

    points, cash = {}, {}
    for sha, role in [(nav_ref, "NAV"), (dividend_ref, "DIVIDEND")]:
        ref = source.original(sha)
        params = ref.get("params_without_credentials") or ref.get("params") or {}
        require(params.get("ts_code") == ts_code, "MONTHLY_" + role + "_SHARE_REQUEST")
    for row in source.table(nav_ref, symbol=ts_code):
        day = iso(row["nav_date"])
        if days[0] <= day <= days[-1]:
            value = D(str(row["unit_nav"]))
            require(value.is_finite() and value > 0, "MONTHLY_NAV_VALUE")
            require(iso(row["ann_date"]) <= source.research, "MONTHLY_NAV_ANNOUNCEMENT_TIME")
            require(day not in points or points[day] == value, "MONTHLY_NAV_CONFLICTING_DUPLICATE")
            points[day] = value
    require(bool(selected_horizons) and set(selected_horizons) <= {"return_1y", "return_3y"}, "MONTHLY_RETURN_SCOPE")
    needed = [days[-1]] + ([months[-13]] if "return_1y" in selected_horizons else []) + ([days[0]] if "return_3y" in selected_horizons else [])
    missing = [day for day in needed if day not in points]
    require(not missing, "MONTHLY_EXACT_RETURN_ENDPOINT_MISSING:" + ts_code + ":" + ",".join(missing))
    for row in source.table(dividend_ref, symbol=ts_code):
        if row.get("div_proc") != "实施":
            continue
        day = iso(row["ex_date"])
        if not days[0] < day <= days[-1]:
            continue
        require(iso(row["ann_date"]) <= source.asof and day in days, "MONTHLY_DIVIDEND_DATE")
        value = D(str(row["div_cash"]))
        require(value.is_finite() and value > 0, "MONTHLY_DIVIDEND_CASH")
        require(day not in cash or cash[day] == value, "MONTHLY_DIVIDEND_CONFLICT")
        cash[day] = value
    with localcontext() as context:
        context.prec = 70
        def growth(start):
            factor = D(1)
            for day, amount in sorted(cash.items()):
                if start < day <= days[-1]:
                    before = days[days.index(day) - 1]
                    require(before in points and points[before] > amount, "MONTHLY_DIVIDEND_PREVIOUS_NAV_REQUIRED")
                    factor *= points[before] / (points[before] - amount)
            return points[days[-1]] / points[start] * factor
        one = growth(months[-13]) - 1 if "return_1y" in selected_horizons else None
        annual = (growth(days[0]) ** (D("365.25") / (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days) - 1
                  if "return_3y" in selected_horizons else None)
    return dict(return_1y=str(one) if one is not None else None, return_3y=str(annual) if annual is not None else None,
                endpoint_nav={day: str(points[day]) for day in sorted(set(needed))},
                cash_dividends={day: str(value) for day, value in cash.items()},
                observed_nav_days=len(points), full_daily_calendar=set(days) <= set(points))


def _same_index_stock_book(source, digest, code):
    """重读原完整中报股票表及财报勾稽，不把旧穿透金额当新输入。"""
    from app.services.fund_rating_thematic_source import plain, require
    from app.services.fund_quality_v2_disclosures import report_period

    source.original(digest)
    text = source.body(digest)
    require(code in plain(text)[:20000] and str(report_period(text)[1]) == "2026-06-30", "MONTHLY_BOOK_LEGAL_PERIOD")
    starts = list(re.finditer(r"(?m)^\s*(?:\d+\.\d+(?:\.\d+)?)?\s*期末按公允价值[^\n]*股票投资明细", text))
    if not starts:
        starts = list(re.finditer(r"期末按公允价值[^\n]*股票投资明细", text))
    require(bool(starts), "MONTHLY_BOOK_FULL_STOCK_HEADING")
    actual = [m for m in starts if len(text[m.end():].split("报告期内股票投资组合的重大变动", 1)[0]) > 300]
    heading = actual[-1] if actual else starts[-1]
    body = text[heading.end():].split("报告期内股票投资组合的重大变动", 1)[0]
    pattern = (r"(?m)^\s*(\d{1,4})\s+(\d{6})(?=\s|[\u4e00-\u9fffＡ-ＺA-Za-z*])\s*"
               r"([\u4e00-\u9fffＡ-ＺA-Za-z*\-\s]+?)\s+([\d,]+(?:\.\d+)?)\s+([\d,]+\.\d{2})\s+([\d.]+)\s*$")
    parsed = list(re.finditer(pattern, body))
    found = {(int(m[1]), m[2]) for m in parsed}
    raw_starts = {(int(m[1]), m[2]) for m in re.finditer(r"(?m)^\s*(\d{1,4})\s+(\d{6})\s+", body)}
    require(found == raw_starts, "MONTHLY_BOOK_NO_DROPPED_STOCK_ROWS")
    positions = {}
    for match in parsed:
        positions[match[2]] = positions.get(match[2], D(0)) + D(match[5].replace(",", ""))
    total = sum(positions.values(), D(0))
    balances = [D(m[1].replace(",", "")) for m in re.finditer(
        r"(?:股票投资|权益投资)\s+(?:[67]\.\d+\.\d+\.\d+\s+)?([\d,]+\.\d{2})", text)]
    bridge = D(0)
    if not positions:
        require(bool(re.search(r"本基金(?:本)?报告期末未持有(?:股票|境内股票)", body))
                or (bool(re.search(r"其中[：:]股票投资\s+-\s+-", text))
                    and bool(re.search(r"权益投资\s+-\s+-", text)) and "注：无" in body), "MONTHLY_BOOK_EXPLICIT_ZERO_STOCK")
    elif total not in balances:
        sections = list(re.finditer(r"7\.2\s+报告期末按行业分类", text[:heading.start()]))
        section_start = sections[-1].start() if sections else text.rfind("报告期末按行业分类", 0, heading.start())
        require(section_start >= 0 and bool(balances), "MONTHLY_BOOK_VALUATION_BRIDGE_SECTION")
        industry = text[section_start:heading.start()]
        amounts = [D(m[1].replace(",", "")) for m in re.finditer(r"合\s*计\s+([\d,]+\.\d{2})\s+[\d.]+", industry)]
        bridge = balances[0] - total
        require(sum(amounts, D(0)) == total and bridge >= 0
                and bool(re.search(r"(?:股票投资|权益投资)[^\n]{0,40}(?:包含|含)[^\n]{0,15}可退替代款", text)),
                "MONTHLY_BOOK_ORIGINAL_VALUATION_BRIDGE")
    navs = re.findall(r"(?:净资产合计|所有者权益合计|净资产（基金净值）)\s+([\d,]+\.\d{2})", text)
    require(bool(navs) and len(set(navs)) == 1, "MONTHLY_BOOK_NAV_CLOSURE")
    investments = re.findall(r"基金投资\s+([\d,]+\.\d{2})", text)
    return dict(positions=positions, nav=D(navs[0].replace(",", "")), stock_total=total,
                fund_investment=D(investments[0].replace(",", "")) if investments else D(0),
                nonsecurity_valuation_bridge=str(bridge), source_ref=digest)


def compile_same_index_monthly(inputs, *, root):
    """滚动完整R5绑定：表现和容量重算，仍适用的披露观察保留真实报告日。"""
    from app.services.fund_rating_thematic_source import RawArchive, plain, require
    from app.services.fund_rating_r5 import calculate, normalize_packet

    require(inputs.get("version") == "R5_SAME_INDEX_MONTHLY_ORIGINALS_V1", "MONTHLY_COMPLETE_VERSION")
    result = compile_same_index_performance(inputs["performance"], root=root)
    source = RawArchive(inputs, Path(root))
    previous = source.json(inputs["previous_packet_ref"])
    anchors = source.json(inputs["previous_anchors_ref"])
    require(previous["fund_code"] == inputs["fund_code"] == "008888"
            and previous["as_of_date"] < inputs["as_of_date"], "MONTHLY_COMPLETE_PREVIOUS_PERIOD")
    require(not previous.get("monthly_r5_compilation"), "MONTHLY_COMPLETE_FROZEN_BASE_REQUIRED")
    # 重放原完整R5契约，仅消费原始观察，不读取它的旧分项或总分。
    calculate(previous, anchors, root=Path(root))
    nodes = normalize_packet(previous)["observations"]
    nodes.update(result["observations"])
    books = {}
    with localcontext() as context:
        context.prec = 70
        for code, digest in inputs["book_refs"].items():
            books[code] = _same_index_stock_book(source, digest, code)
        pairs = {"008888": "159995", "012630": "159801", "012970": "159813"}
        require(set(books) == set(pairs) | set(pairs.values()), "MONTHLY_ALL_FEEDER_TARGET_BOOKS")
        portfolios = {}
        for code, etf in pairs.items():
            factor = books[code]["fund_investment"] / books[etf]["nav"]
            require(factor > 0, "MONTHLY_FEEDER_TARGET_VALUATION_FACTOR")
            positions = deepcopy(books[code]["positions"])
            for sec, amount in books[etf]["positions"].items():
                positions[sec] = positions.get(sec, D(0)) + amount * factor
            portfolios[code] = positions
        securities = set().union(*(values.keys() for values in portfolios.values()))
        market = {code: {} for code in securities}
        days = result["dates"][-60:]
        require(len(days) == 60, "MONTHLY_CAPACITY_OFFICIAL_SIXTY_DAYS")
        for digest in inputs["market_refs"]:
            source.original(digest)
            for row in source.table(digest):
                sec = row["ts_code"][:6]
                day = str(date.fromisoformat(row["trade_date"]))
                if sec not in market or day not in days:
                    continue
                value = D(str(row["amount"])) * 1000
                require(value.is_finite() and value >= 0, "MONTHLY_CAPACITY_AMOUNT_UNIT")
                require(day not in market[sec] or market[sec][day] == value, "MONTHLY_CAPACITY_CONFLICTING_SOURCE")
                market[sec][day] = value
        capacities, coverage = {}, {}
        for code, positions in portfolios.items():
            total, value = sum(positions.values()), D(0)
            require(total > 0, "MONTHLY_CAPACITY_STOCK_BUDGET")
            for sec, amount in positions.items():
                missing = sorted(set(days) - market[sec].keys())
                require(not missing, "MONTHLY_CAPACITY_REAL_MARKET_DAYS_MISSING:" + sec + ":" + ",".join(missing))
                adv = sum(market[sec][day] for day in days) / len(days)
                require(adv > 0, "MONTHLY_CAPACITY_NO_REAL_TURNOVER")
                value += amount / total * amount / adv
            capacities[code], coverage[code] = str(value), len(positions)
    cohort = source.json(inputs["performance"]["cohort_ref"])
    products = {code: "CN_PUBLIC_FUND:" + cohort["relationships_to_verify"][code][0] for code in pairs}
    refs = sorted(source.used)
    nodes["market_capacity"] = dict(kind="rank", direction="lower", target=capacities["008888"],
                                    expected_products=[products[p] for p in ["012630", "012970"]],
                                    peers=[dict(product_id=products[p], value=capacities[p]) for p in ["012630", "012970"]],
                                    source_refs=refs)
    role = inputs["performance"]["funds"]["008888"]["contract_report_ref"]
    body = plain(source.body(role))
    continuity = nodes["continuity"]
    for interval in continuity["intervals"]:
        if interval["end"] is None:
            require(interval["name"] in body and interval["start"] in body and "基金经理" in body,
                    "MONTHLY_CURRENT_MANAGER_POSITIVE_ROLE")
            interval["source_refs"] = list(dict.fromkeys(interval["source_refs"] + [role]))
    for digest in inputs["current_catalog_refs"]:
        source.original(digest)
        catalog = source.json(digest)
        require(catalog.get("ErrCode") == 0 and all(row["FUNDCODE"] == "008888" for row in catalog["Data"]),
                "MONTHLY_CURRENT_MANAGER_CATALOG_SHARE")
        changes = [row for row in catalog["Data"] if previous["as_of_date"] < row["PUBLISHDATEDesc"][:10] <= inputs["as_of_date"]
                   and "基金经理" in row["TITLE"]]
        require(not changes, "MONTHLY_NEW_MANAGER_CHANGE_REVIEW_REQUIRED")
    continuity.update(period_start=result["dates"][0], period_end=inputs["as_of_date"],
                      current_role_source_ref=role, source_refs=list(dict.fromkeys(continuity["source_refs"] + [role])))
    return dict(observations=nodes, performance=result, capacities=capacities,
                capacity_stock_counts=coverage, capacity_dates=days, used_source_refs=sorted(source.used),
                anchor=anchors[inputs["fund_code"]])


def prepare_monthly_r5(session, inputs, *, root):
    """由原件输入产生完整月评清单；登记、发布仍由外层既有队列负责。"""
    from app.services.fund_rating_pipeline_prepare import prepare_complete
    from app.services.fund_rating_thematic_source import RawArchive
    from app.services.fund_quality_v2_disclosures import report_period

    compiled = compile_same_index_monthly(inputs, root=root)
    source = RawArchive(inputs, Path(root))
    packet = deepcopy(source.json(inputs["previous_packet_ref"]))
    anchors = deepcopy(source.json(inputs["previous_anchors_ref"]))
    packet.update(as_of_date=inputs["as_of_date"], verified_at=inputs["verified_at"],
                  unit_contract="RATIO_RANKS_RATIO_WINDOWS", monthly_r5_compilation=deepcopy(inputs))
    packet["quality_metrics"]["observations"] = compiled["observations"]
    packet["raw_archive"] = [{**ref, "path": ref["file"], "kind": ref["archive_kind"]}
                             for ref in inputs["sources"].values()]
    packet["r4_evidence_registry"] = {sha: {**ref, "path": ref["file"]} for sha, ref in inputs["sources"].items()}
    dates = packet["quality_public"]["dataDates"]
    role = inputs["performance"]["funds"][inputs["fund_code"]]["contract_report_ref"]
    dates.update(nav=inputs["as_of_date"], manager=str(report_period(source.body(role))[1]))
    public = packet["quality_public"]
    public.update(summary="按本期本C份额及同指数同行真实历史重算，完整持仓保留原报告日；本评价不是涨跌预测。",
                  strengths=[], weaknesses=[])
    public["comparison"]["period"] = compiled["performance"]["dates"][0] + " 至 " + inputs["as_of_date"]
    audit = compiled["performance"]["benchmark_audit"]
    public["limitations"] = [
        "官网基准本轮实际观察至" + audit["official_last_observed"]
        + "；其后按本C现行95/5合同、真实指数及央行活期基准重建，不冒充官网观察。",
        "完整持仓和支付覆盖保留" + dates["holdings"] + "真实披露；容量已按本期末最后60个官方交易日重算。",
        "仅两个独立同指数C类参照，不等于全市场排名；历史质量不预测未来收益。",
    ]
    for dimension in public.get("dimensions", []):
        dimension["explanation"] = "本期按原始观察及既定R5公式重算；持仓、费用和治理观察保留各自真实披露日。"
    return prepare_complete(session, packet, anchors, root=Path(root))
