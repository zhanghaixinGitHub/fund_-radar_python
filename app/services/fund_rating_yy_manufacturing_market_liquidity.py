"""018125完整成交承接与已核三子项的独立原件重放候选。

只依赖正式app与标准库，不读取研究目录、旧绝对路径或预先算好的K/分数。
持仓按原报告人民币金额；A/H各用自己的真实开市日，同日汇率不前填。
供应商港股金额显示精度保留区间，未知缺行不当0，重大事项未知仍为None。
"""

import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal as D
from decimal import localcontext
from functools import lru_cache
from pathlib import Path

from bs4 import BeautifulSoup

from app.services.fund_quality_v2_balance_proofs import strict_balance_proof
from app.services.fund_rating_direct_index import Originals, _table
from app.services.fund_rating_inputs import canonical
from app.services.fund_rating_market_review import normalized_text
from app.services.fund_rating_public_stock_tables import parse
from app.services.fund_rating_r5 import exposure_factor, peer_gap_score
from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS, policy
from app.services.fund_rating_yy_manufacturing_liquidity import (
    LEGAL,
    MANAGER,
    PRIMARY,
    original_pages,
    require,
)
from app.services.fund_rating_yy_manufacturing_liquidity import (
    calculate as known_calculate,
)
from app.services.fund_rating_yy_manufacturing_liquidity import (
    check_binding as known_check_binding,
)

VERSION = "R5_018125_MARKET_LIQUIDITY_ORIGINALS_20261030_1"
PEERS = ("001644", "007114", "008084", "010000", "012181", "013479", "014536", "014930", "015058", "017760")
PERIODS = ("2023-12-31", "2024-12-31", "2025-12-31", "2026-06-30")
HK_CODES = ("00179", "02507", "06869", "09880")
HK_CALENDAR = "783afe4db4e8e410bfd016cc2e212287691377b5d0680a8bc83d6e30bc469819"
HK_HOLIDAYS = {
    "2026-01-01",
    "2026-02-17",
    "2026-02-18",
    "2026-02-19",
    "2026-04-03",
    "2026-04-06",
    "2026-04-07",
    "2026-05-01",
    "2026-05-25",
    "2026-06-19",
    "2026-07-01",
    "2026-09-26",
    "2026-10-01",
    "2026-10-19",
    "2026-12-25",
}
HALTS = {
    "688072.SH": ("2026-06-29", "2026-07-13"),
    "300567.SZ": ("2026-07-09", "2026-07-16"),
    "688277.SH": ("2026-07-16", "2026-07-30"),
}
LAYOUT_HASHES = {
    "001644": "d68817a195237495ad25f33c937e02fd8d552a4c478dddd84978157204da11c2",
    "013479": "a751cd31980d3202ea0fc64483097e3a05034101310d3586f6e0a59e378d15fe",
    "014930": "9faa88691652095ba3ae31c310e4130bd7271f2363a8c8c9a399639ecd713c4e",
}


def _source(inputs, root):
    require(
        inputs.get("version") == VERSION
        and inputs.get("fund_code") == "018125"
        and inputs.get("share_class") == "C"
        and inputs.get("basis") == ACTIVE_BASIS
        and inputs.get("legal_name") == LEGAL
        and inputs.get("manager_name") == MANAGER
        and inputs.get("as_of_date") == "2026-09-30"
        and inputs.get("holding_date") == "2026-06-30",
        "YYM_COMPONENT_SCOPE",
    )
    require(
        inputs.get("material_event_pending") is None and inputs.get("performance_release_requested") is False,
        "YYM_NO_PERFORMANCE_OR_MATERIAL_DEFAULT",
    )
    require(
        not any(k in inputs for k in ["score", "dimensions", "observations", "quality_metrics", "packet", "K"]),
        "YYM_RAW_ORIGINALS_REQUIRED",
    )
    c = inputs["capacity"]
    require(tuple(c["peer_codes"]) == PEERS and set(c["reports"]) == {"018125", *PEERS}, "YYM_FIXED_PEER_SET")
    require(c["reports"]["018125"] == PRIMARY and c["hk_calendar_ref"] == HK_CALENDAR, "YYM_PRIMARY_OR_CALENDAR")
    require(c.get("time_contract") == "RETROSPECTIVE_ACTUAL_MARKET_HISTORY_NOT_PIT_SIGNAL", "YYM_TIME_CONTRACT")
    require(c.get("precision_contract") == "ONE_DISPLAY_LAST_UNIT_PLUS_ONE_HKD", "YYM_AMOUNT_PRECISION_CONTRACT")
    require(
        set(c["HK_daily_refs"]) == set(HK_CODES) and set(c["suspensions"]) == set(HALTS), "YYM_MARKET_SECURITY_SCOPE"
    )
    require(set(c["exposure_refs"]) == {"018125", *PEERS}, "YYM_EXPOSURE_SCOPE")
    reviews = c["admission_reviews"]
    require(
        len(reviews) == 40 and {(r["code"], r["period"]) for r in reviews} == {(p, t) for p in PEERS for t in PERIODS},
        "YYM_ADMISSION_REVIEW_COVERAGE",
    )
    financial = {
        PRIMARY,
        *inputs["known_liquidity_inputs"]["sources"],
        *c["reports"].values(),
        *(r["source_hash"] for r in reviews),
        HK_CALENDAR,
    }
    financial.update(claim["source_hash"] for pair in c["suspensions"].values() for claim in pair["claims"])
    historical = {
        c["A_calendar_ref"],
        c["FX_history_ref"],
        c["FX_method_ref"],
        c["A_unit_ref"],
        *c["A_daily_refs"].values(),
        *c["HK_daily_refs"].values(),
        *c["HK_primary_refs"].values(),
        *(h for values in c["exposure_refs"].values() for h in values),
    }
    refs = inputs["sources"]
    require(set(refs) == financial | historical, "YYM_UNKNOWN_OR_MISSING_SOURCE")
    require(
        inputs["known_liquidity_inputs"]["sources"]
        == {h: refs[h] for h in inputs["known_liquidity_inputs"]["sources"]},
        "YYM_KNOWN_COMPONENT_SOURCE_BINDING",
    )
    for h, ref in refs.items():
        require(ref.get("sha256") == h and ref.get("archive_kind") == "ORIGINAL", "YYM_IMMUTABLE_ORIGINAL_REQUIRED")
        if h in financial:
            require(
                isinstance(ref.get("published_on"), str)
                and date.fromisoformat(ref["published_on"]) <= date(2026, 9, 30),
                "YYM_FINANCIAL_PUBLICATION_REQUIRED",
            )
        else:
            # 原始历史回包可在研究日取得；经济日期在各解析器内逐行过滤，不倒填公开日。
            require(ref.get("retrospective_history_or_method") is True, "YYM_RETROSPECTIVE_SOURCE_SCOPE")
    return Originals(inputs, Path(root).resolve())


def _pages(source, digest):
    if digest not in source.pages_cache:
        raw = source.raw[digest]
        if raw.startswith(b"%PDF"):
            source.pages_cache[digest] = original_pages(raw)
        else:
            payload = source.json(digest)
            require(
                payload.get("success") == 1 and isinstance(payload.get("data", {}).get("notice_content"), str),
                "YYM_FULL_NOTICE_CONTENT",
            )
            require(payload["data"].get("is_ai_summary", 0) == 0, "YYM_NOT_AI_SUMMARY")
            body = BeautifulSoup(payload["data"]["notice_content"], "html.parser").get_text("\n", strip=True)
            require(len(normalized_text(body)) > 15000, "YYM_COMPLETE_REPORT_REQUIRED")
            source.pages_cache[digest] = [body]
    return source.pages_cache[digest]


def _layout(code, digest, pages):
    """三个已逐页核实的排版例外只改解析视图，不变股票金额或不可变原件。"""
    if code not in LAYOUT_HASHES:
        return list(pages)
    require(digest == LAYOUT_HASHES[code], "YYM_REVIEWED_LAYOUT_VERSION")
    out, changes = [], 0
    for physical, page in enumerate(pages, 1):
        lines = []
        for offset, line in enumerate(page.splitlines()):
            if code == "001644" and re.fullmatch(r"第\s*\d+\s*页[，,]\s*共\s*\d+\s*页", line.strip()):
                line = ""
            if code == "013479" and physical == 41 and re.match(r"^\s*28\s+H02507\s+西锐\s+", line):
                line = re.sub(r"(?<=\s)H02507(?=\s)", "02507", line)
                changes += 1
            if code == "014930" and physical in (45, 46) and offset == 0:
                require(line.strip() == str(physical), "YYM_ORIGINAL_PHYSICAL_FOOTER")
                line = ""
                changes += 1
            lines.append(line)
        out.append("\n".join(lines))
    require(code != "013479" or changes == 1, "YYM_HK_PREFIX_LAYOUT")
    require(code != "014930" or changes == 2, "YYM_FOOTER_LAYOUT")
    return out


def _admission(source, c):
    """12季股票预算从原回包重算；制造机会集合/运作方式保持已读原件人工判断。"""
    observations = {}
    for code, refs in c["exposure_refs"].items():
        require(len(refs) == 4, "YYM_EXPOSURE_FOUR_YEAR_ORIGINALS")
        quarters = {}
        for h in refs:
            data = source.json(h)["Data"]
            require(data["FundCode"] == code, "YYM_EXPOSURE_PRODUCT_IDENTITY")
            for q in data["QuarterInfos"]:
                day = q["JZRQ"]
                if not "2023-09-30" <= day <= "2026-06-30":
                    continue
                require(
                    len({r["HYDM"] for r in q["HYPZInfo"]}) == len(q["HYPZInfo"]), "YYM_DUPLICATE_EXPOSURE_INDUSTRY"
                )
                domestic = hongkong = ed = eh = D(0)
                for row in q["HYPZInfo"]:
                    require(row["BZDM"] == code and row["FSRQ"] == day, "YYM_EXPOSURE_ROW_IDENTITY")
                    value = D(row["ZJZBL"])
                    require(0 <= value <= 110, "YYM_PERCENT_EXPOSURE_UNIT")
                    error = D(1).scaleb(value.as_tuple().exponent) / 2
                    if re.fullmatch("[A-Z]", row["HYDM"]):
                        domestic += value
                        ed += error
                    elif re.fullmatch(r"\d+", row["HYDM"]):
                        hongkong += value
                        eh += error
                    else:
                        raise ValueError("YYM_EXPOSURE_REGION_UNKNOWN")
                total = domestic + hongkong
                error = ed + eh
                require(0 < total <= 110, "YYM_EXPOSURE_TOTAL")
                quarters[day] = (total, error, max(D(0), domestic - ed) / (total + error))
        require(len(quarters) == 12, "YYM_TWELVE_QUARTERS_REQUIRED")
        observations[code] = quarters
    target = observations["018125"]
    audit = []
    for code in PEERS:
        require(set(observations[code]) == set(target), "YYM_QUARTER_DATE_COVERAGE")
        upper = [
            abs(observations[code][day][0] - target[day][0]) + observations[code][day][1] + target[day][1]
            for day in sorted(target)
        ]
        mean, max_gap, region = sum(upper) / 12, max(upper), min(x[2] for x in observations[code].values())
        require(mean <= 5 and max_gap <= 10 and region >= D(".9"), "YYM_FIXED_PEER_EXPOSURE_NOT_ADMITTED:" + code)
        audit.append(
            {
                "code": code,
                "mean_gap_upper_pp": str(mean),
                "max_gap_upper_pp": str(max_gap),
                "domestic_stock_lower": str(region),
            }
        )
    for review in c["admission_reviews"]:
        require(
            review["requested_C_class_confirmed"] is True
            and review["normal_open_end_RMB"] is True
            and len(review["manual_opportunity_set_judgment"]) >= 20,
            "YYM_MANUAL_STRATEGY_REVIEW_REQUIRED",
        )
        pages = _pages(source, review["source_hash"])
        plain = normalized_text("\n".join(pages))
        heading = normalized_text("\n".join(pages[:3]))
        year = review["period"][:4]
        titles = (
            [year + "年年度报告"]
            if review["period"].endswith("12-31")
            else [year + "年中期报告", year + "年半年度报告"]
        )
        # 富国原封面实际使用“二0二六年”；只补中文年份同义写法，不改经济期。
        chinese = "".join("〇一二三四五六七八九"[int(ch)] for ch in year)
        titles.extend(
            t.replace(year, cn)
            for t in list(titles)
            for cn in (chinese, chinese.replace("〇", "0"), chinese.replace("〇", "零"))
        )
        require(
            any(title in heading for title in titles),
            "YYM_ADMISSION_REPORT_PERIOD:" + review["code"] + ":" + review["period"],
        )
        require(
            review["code"] in normalized_text("\n".join(pages[:15])) and "契约型开放式" in plain and "人民币" in plain,
            "YYM_REVIEWED_C_AND_OPERATION",
        )

        # 原准入索引去掉了标点，仅将它用于定位主题/策略语句；所有金额和预算另按原数值重算。
        def context_text(text):
            return re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", text))

        for claims in review["anchors"].values():
            for claim in claims:
                physical = claim["physical_page_or_HTML_part"]
                require(
                    1 <= physical <= len(pages)
                    and len(context_text(claim["quote"])) >= 20
                    and context_text(claim["quote"]) in context_text(pages[physical - 1]),
                    "YYM_ADMISSION_ORIGINAL_CONTEXT:" + review["code"] + ":" + review["period"] + ":" + str(physical),
                )
    return audit


def _books(source, c):
    result = {}
    for code, digest in c["reports"].items():
        pages = _pages(source, digest)
        require(code in normalized_text("\n".join(pages[:15])), "YYM_CURRENT_C_REPORT_BINDING:" + code)
        balance = strict_balance_proof(pages, observed_on="2026-06-30", source_hash=digest)["point_book"]
        require(balance is not None, "YYM_FULL_REPORT_BALANCE")
        stocks = parse(
            _layout(code, digest, pages), balance["stock_amount"], balance["net_assets"], require_industry=False
        )
        positions = {r["stock_code"]: D(r["market_value_cny"]) for r in stocks["holdings"]}
        require(
            len(positions) == len(stocks["holdings"])
            and abs(sum(positions.values()) - D(stocks["stock_value_cny"])) <= D(".02"),
            "YYM_FULL_STOCK_AMOUNT",
        )
        result[code] = {"holdings": positions, "total": D(stocks["stock_value_cny"])}
    return result


def _calendars(source, c):
    rows = _table(source.json(c["A_calendar_ref"]))
    require(all(r["exchange"] == "SSE" for r in rows), "YYM_A_MARKET_CALENDAR_IDENTITY")
    require(len({r["cal_date"] for r in rows}) == len(rows), "YYM_CALENDAR_DUPLICATE")
    daysA = sorted(
        date.fromisoformat(r["cal_date"][:4] + "-" + r["cal_date"][4:6] + "-" + r["cal_date"][6:]).isoformat()
        for r in rows
        if int(r["is_open"]) == 1 and r["cal_date"] <= "20260930"
    )
    require(len(daysA) >= 120 and daysA[-1] == "2026-09-30", "YYM_A_CALENDAR_END")
    require(set(c["A_daily_refs"]) == set(daysA[-120:]), "YYM_A_DAILY_WINDOW_COVERAGE")
    require("2026" in "".join(_pages(source, HK_CALENDAR)), "YYM_HK_2026_CALENDAR")
    daysHK = []
    day = date(2026, 1, 1)
    while day <= date(2026, 9, 30):
        if day.weekday() < 5 and day.isoformat() not in HK_HOLIDAYS:
            daysHK.append(day.isoformat())
        day += timedelta(days=1)
    return {n: daysA[-n:] for n in (30, 60, 120)}, {n: daysHK[-n:] for n in (30, 60, 120)}


def _halts(source, c, days):
    def chinese(iso):
        a, b, d = iso.split("-")
        return a + "年" + str(int(b)) + "月" + str(int(d)) + "日"

    verified = {}
    for code, (start, end) in HALTS.items():
        pair = c["suspensions"][code]
        require(pair["start"] == start and pair["resumption"] == end, "YYM_ACTUAL_HALT_WINDOW")
        claims = []
        for claim in pair["claims"]:
            pages = _pages(source, claim["source_hash"])
            physical = claim["physical_page"]
            quote = normalized_text(claim["quote"])
            require(
                1 <= physical <= len(pages) and quote in normalized_text(pages[physical - 1]),
                "YYM_SUSPENSION_ORIGINAL_CLAUSE",
            )
            claims.append(quote)
        require(
            any(chinese(start) in x and "停牌" in x for x in claims)
            and any(chinese(end) in x and "开市起复牌" in x for x in claims),
            "YYM_HALT_NOT_FORECAST_END",
        )
        verified[code] = {d for d in days if start <= d < end}
    return verified


def _HK_primary(raw, day):
    """表第一行为前收/卖价/最高/股数，第二行才是收盘/买价/最低/实际金额。"""
    text = raw.decode("utf-8")
    require("DATE: " + date.fromisoformat(day).strftime("%d %b %Y").upper() in text, "YYM_HK_PRIMARY_DAY")
    start = re.search(r"<a\s+name\s*=\s*[\"\']quotations[\"\']>QUOTATIONS</a>", text)
    require(start is not None, "YYM_HK_QUOTATION_SECTION")
    body = text[start.end() :]
    require("PRV.CLO./" in body[:400] and "TURNOVER ($)" in body[:400], "YYM_HK_TURNOVER_HEADER")
    out = {}
    for code in HK_CODES:
        found = list(re.finditer(r"(?m)^[ *#@]*" + str(int(code)) + r"\s+([^\r\n]+?)\s+HKD\s+([^\r\n]+)\r?$", body))
        require(len(found) == 1, "YYM_HK_PRIMARY_SECURITY:" + code)
        match = found[0]
        first = match.group(2).split()
        following = [s.strip() for s in body[match.end() :].splitlines() if s.strip()][:1]
        require(len(following) == 1, "YYM_HK_SECOND_QUOTE_LINE")
        second = following[0].split()
        require(
            len(first) == len(second) == 4 and all(re.fullmatch(r"[\d,.]+", v) for v in first + second),
            "YYM_HK_QUOTE_FIELD_FORMAT",
        )
        out[code] = {"shares": D(first[-1].replace(",", "")), "amount": D(second[-1].replace(",", ""))}
    return out


def _compile(inputs, source, root):
    """完成原件重放后合成区间。研究取得日与行情经济日分开，绝不读取未来元数据。"""
    c = inputs["capacity"]
    with localcontext() as context:
        context.prec = 65
        admission = _admission(source, c)
        books = _books(source, c)
        datesA, datesHK = _calendars(source, c)
        halts = _halts(source, c, datesA[120])
        require(
            "amount" in source.raw[c["A_unit_ref"]].decode("utf-8")
            and "千元" in source.raw[c["A_unit_ref"]].decode("utf-8"),
            "YYM_PROVIDER_UNIT_ORIGINAL",
        )
        dailyA = {}
        for day, h in c["A_daily_refs"].items():
            rows = _table(source.json(h))
            require(all(r["trade_date"] == day.replace("-", "") for r in rows), "YYM_A_ACTUAL_TRADE_DATE")
            require(len({r["ts_code"] for r in rows}) == len(rows), "YYM_A_DUPLICATE_SECURITY")
            dailyA[day] = {r["ts_code"]: D(str(r["amount"])) * 1000 for r in rows if r.get("amount") is not None}
        zero_trace = []
        for code, days in halts.items():
            for day in sorted(days):
                existing = dailyA[day].get(code)
                require(existing is None or existing == 0, "YYM_HALT_CONFLICTS_POSITIVE_TURNOVER")
                if existing is None:
                    dailyA[day][code] = D(0)
                    zero_trace.append({"security": code, "date": day, "basis": "ACTUAL_HALT_AND_RESUMPTION_ORIGINALS"})
        fx = {}
        for cube in ET.fromstring(source.raw[c["FX_history_ref"]]).iter():
            if "time" not in cube.attrib or cube.attrib["time"] > "2026-09-30":
                continue
            rates = {r.attrib.get("currency"): r.attrib.get("rate") for r in cube}
            if "CNY" in rates and "HKD" in rates:
                fx[cube.attrib["time"]] = D(rates["CNY"]) / D(rates["HKD"])
        providers = {}
        for code, h in c["HK_daily_refs"].items():
            require(
                inputs["sources"][h].get("params", {}).get("param")
                == "hk" + code + ",day,2026-01-01,2026-09-30,320,qfq",
                "YYM_HK_PROVIDER_REQUEST_SCOPE",
            )
            payload = source.json(h)
            require(payload.get("code") == 0 and set(payload["data"]) == {"hk" + code}, "YYM_HK_PROVIDER_IDENTITY")
            data = payload["data"]["hk" + code]
            rows = data.get("qfqday", data.get("day"))
            require(
                rows and all(len(r) >= 9 for r in rows) and len({r[0] for r in rows}) == len(rows),
                "YYM_HK_DAILY_FIELDS",
            )
            providers[code] = {r[0]: r for r in rows if r[0] <= "2026-09-30"}
        primary = {d: _HK_primary(source.raw[h], d) for d, h in c["HK_primary_refs"].items()}
        require(
            set(primary) == {d for d in datesHK[60] if d >= "2026-08-31"} and len(primary) == 23,
            "YYM_23_HK_PRIMARY_DAYS",
        )
        crosschecks = []
        for day, quotes in primary.items():
            for code, q in quotes.items():
                row = providers[code][day]
                digits = len(row[8].split(".")[1]) if "." in row[8] else 0
                unit = D(10) ** (4 - digits)
                require(
                    D(row[5]) == q["shares"] and abs(D(row[8]) * 10000 - q["amount"]) <= unit + 1,
                    "YYM_HK_AMOUNT_VOLUME_CROSSCHECK",
                )
                crosschecks.append({"date": day, "code": code, "within_display_unit_plus_1_HKD": True})
        require(len(crosschecks) == 92, "YYM_PRIMARY_CROSSCHECK_COVERAGE")
        results = {}
        for n in (60, 30, 120):
            products = {}
            for code, book in books.items():
                lower = upper = D(0)
                missing = []
                for sec, holding in book["holdings"].items():
                    amounts_lo = []
                    amounts_hi = []
                    if sec.endswith(".HK"):
                        short = sec[:5]
                        for day in datesHK[n]:
                            if day not in providers[short] or day not in fx:
                                missing.append({"security": sec, "date": day, "kind": "HK_DAILY_OR_SAME_DAY_FX"})
                                continue
                            row = providers[short][day]
                            quoted = D(row[8]) * 10000
                            if day in primary:
                                lo = hi = primary[day][short]["amount"]
                            else:
                                digits = len(row[8].split(".")[1]) if "." in row[8] else 0
                                uncertainty = D(10) ** (4 - digits) + 1
                                lo, hi = max(D(0), quoted - uncertainty), quoted + uncertainty
                            amounts_lo.append(lo * fx[day])
                            amounts_hi.append(hi * fx[day])
                    else:
                        for day in datesA[n]:
                            if sec not in dailyA[day]:
                                missing.append({"security": sec, "date": day, "kind": "A_DAILY_UNKNOWN_NOT_ZERO"})
                                continue
                            value = dailyA[day][sec]
                            require(value >= 0, "YYM_NEGATIVE_ACTUAL_TURNOVER")
                            amounts_lo.append(value)
                            amounts_hi.append(value)
                    if len(amounts_lo) != n:
                        continue
                    adv_lo, adv_hi = sum(amounts_lo) / n, sum(amounts_hi) / n
                    require(0 < adv_lo <= adv_hi, "YYM_POSITIVE_ADV_REQUIRED")
                    lower += holding * holding / book["total"] / adv_hi
                    upper += holding * holding / book["total"] / adv_lo
                products[code] = {
                    "K_interval": None if missing else [str(lower), str(upper)],
                    "missing": missing,
                    "stock_count": len(book["holdings"]),
                    "stock_total_CNY": str(book["total"]),
                }
            complete = all(p["K_interval"] is not None for p in products.values())
            score = None
            if complete:
                t = products["018125"]["K_interval"]
                p = [products[k]["K_interval"] for k in PEERS]
                lo = peer_gap_score(D(t[1]), [D(x[0]) for x in p]) * exposure_factor("market_capacity", D(t[1]))
                hi = peer_gap_score(D(t[0]), [D(x[1]) for x in p]) * exposure_factor("market_capacity", D(t[0]))
                require(0 <= lo <= hi <= 100, "YYM_CAPACITY_INTERVAL_ORDER")
                score = [str(lo), str(hi)]
            results[str(n)] = {
                "all_members_complete": complete,
                "A_dates": datesA[n],
                "HK_dates": datesHK[n],
                "products": products,
                "market_capacity_score_interval": score,
            }
        require(
            results["60"]["all_members_complete"] and results["30"]["all_members_complete"],
            "YYM_MAIN_OR_SHORT_WINDOW_INCOMPLETE",
        )
        known = known_calculate(inputs["known_liquidity_inputs"], root=root)
        terms = policy(ACTIVE_BASIS)["terms"]["liquidity"]
        require(
            {k: D(w) for k, w in terms}
            == {"market_capacity": D(40), "restricted": D(20), "known_payment": D(25), "holders": D(15)},
            "YYM_FIXED_WEIGHT_RULE_CHANGED",
        )
        known_part = sum(D(known["subscores"][k]) * D(w) / 100 for k, w in terms if k != "market_capacity")
        capacity_lo, capacity_hi = map(D, results["60"]["market_capacity_score_interval"])
        dimension = {
            "lower": str(known_part + D(".4") * capacity_lo),
            "upper": str(known_part + D(".4") * capacity_hi),
            "candidate": None,
            "missing": [],
            "uncertainty": ["HK_AMOUNT_DISPLAY_PRECISION_INTERVAL"],
        }
    return {
        "dimensions": {"liquidity": dimension},
        "complete_score": None,
        "grade": None,
        "subscores": known["subscores"],
        "capacity_windows": results,
        "audit": {
            "peer_exposure": admission,
            "HK_provider_exchange_checks": crosschecks,
            "proven_suspension_zero_trace": zero_trace,
            "material_event_pending": None,
            "future_quote_metadata_used": False,
            "historical_after_hours_increment_invented": False,
            "A_amount_scope": "daily.amount为原始日成交额（千元）；不猜测ah_amount为新增量，不承诺盘后承接。",
            "FX_scope": "ECB同日EUR交叉CNY/HKD参考率，非执行汇率或PIT信号",
            "known_liquidity": known["audit"],
        },
    }


@lru_cache(maxsize=1)
def _cached(serialized, root_name):
    inputs = json.loads(serialized)
    return _compile(inputs, _source(inputs, root_name), root_name)


def calculate(inputs, *, root):
    """每次先重验全部原件字节，再允许复用同输入的解析结果；不允许热缓存绕过验摘要。"""
    _source(inputs, root)
    return deepcopy(_cached(canonical(inputs), str(Path(root).resolve())))


def check_binding(bundle, inputs):
    known_check_binding(bundle, inputs["known_liquidity_inputs"])
    require(
        bundle.get("fund_code") == inputs.get("fund_code") and bundle.get("share_class") == "C", "YYM_CURRENT_BINDING"
    )


def replay_snapshot_interval(bundle, dimensions, *, root):
    """快照的区间必须与原件重算相等，含缺项/精度边界；不得改成点分。"""
    raw = bundle["yy_manufacturing_market_liquidity_calculation"]
    check_binding(bundle, raw["inputs"])
    expected = calculate(raw["inputs"], root=root)["dimensions"]["liquidity"]
    actual = dimensions["liquidity"]
    # 标准快照返回字段会裁掉研究缺项说明；其语义必须另从已归档研究正文核对。
    declared = bundle["reviewed_document"]["dimensions"]["liquidity"]
    require(
        actual.get("candidate") is None
        and all(D(actual[b]) == D(expected[b]) for b in ("lower", "upper"))
        and declared.get("missing") == expected["missing"]
        and declared.get("uncertainty") == expected["uncertainty"],
        "YYM_SNAPSHOT_ORIGINAL_RECALCULATION",
    )


def prepare(session, request, *, root):
    """只读当前本C份额；在指定归档根预演区间替换，保留其余七维和重大事项未知。"""
    from app.services.fund_rating_pipeline import current_inputs
    from app.services.fund_rating_pipeline_prepare import archive_json, canonical_source
    from app.services.fund_rating_reviewed_snapshot import BASIS, validate_sources
    from app.services.fund_rating_reviewed_snapshot import calculate as snapshot_calculate

    root = Path(root).resolve()
    raw = deepcopy(request["inputs"])
    code = request["fund_code"]
    require(code == "018125" and request["as_of_date"] == raw["as_of_date"], "YYM_REQUEST_SCOPE")
    previous = current_inputs(session, [code]).get(code)
    require(previous and str(previous["as_of_date"]) == request["as_of_date"], "YYM_LATEST_CURRENT_REQUIRED")
    require(
        not request.get("expected_current_ref") or request["expected_current_ref"] == previous["batch_id"],
        "YYM_CURRENT_CHANGED_REVIEW_REQUIRED",
    )
    bundle = deepcopy(next(m["bundle"] for m in previous["input_snapshot"]["members"] if m["fund_code"] == code))
    check_binding(bundle, raw)
    require(bundle.get("basis") == BASIS and not bundle.get("r5_calculation"), "YYM_PARTIAL_CURRENT_REQUIRED")
    for h, ref in raw["sources"].items():
        raw["sources"][h] = canonical_source(root, ref)
    raw["known_liquidity_inputs"]["sources"] = {h: raw["sources"][h] for h in raw["known_liquidity_inputs"]["sources"]}
    value = calculate(raw, root=root)
    reason = request["explanations"]["liquidity"].strip()
    require(len(reason) >= 40, "YYM_BUSINESS_EXPLANATION_REQUIRED")
    doc = bundle["reviewed_document"]
    before = deepcopy(doc["dimensions"])
    doc["dimensions"]["liquidity"] = {**value["dimensions"]["liquidity"], "reason": reason}
    bundle.setdefault("public_explanations", {})["liquidity"] = reason
    bundle.get("public_metrics", {}).pop("liquidity", None)
    # 同一维度只保留一个原件组件，避免旧50—90区间钩子与新精度区间并存。
    bundle.pop("yy_manufacturing_liquidity_calculation", None)
    bundle.update(
        research_format="single",
        research_archive=archive_json(root, doc),
        yy_manufacturing_market_liquidity_calculation={"inputs": raw, "input_archive": archive_json(root, raw)},
    )
    known = {r["sha256"] for r in bundle.setdefault("extra_archives", [])}
    bundle["extra_archives"].extend(ref for h, ref in raw["sources"].items() if h not in known)
    require(
        all(doc["dimensions"][k] == v for k, v in before.items() if k != "liquidity"), "YYM_OTHER_DIMENSIONS_CHANGED"
    )
    validate_sources(bundle, datetime.now(UTC), root=root)
    replay_snapshot_interval(bundle, doc["dimensions"], root=root)
    checked = snapshot_calculate(bundle, root=root)
    require(checked["status"] == "PARTIAL" and checked["score"] is None, "YYM_NO_COMPLETE_SCORE")
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}]}
