"""综合审阅的补充日历核验：仅支持已留原文的境内港股通及境内黄金ETF窗口。"""

import re
from datetime import date
from urllib.parse import urlparse

from app.services.nav_repair import domestic_calendar_supported

QDII_WORKDAY_POLICY = "QDII_EXPLICIT_CN_WORKDAY_NAV_V1"


def verified_qdii_workday_valuation(material: dict, documents: dict) -> bool:
    """只认本基金明确的逐境内工作日估值条款，不把交易开放日当估值日。

    境外休市时股票可能沿用最近价格或作公允价值调整，汇率也可能变化。
    本函数只确认应该估值的日期，不假定那天的净值、价格或收益不变。
    """
    from app.services.direction_1d_protocol import digest

    evidence = material.get("calendar_evidence") or {}
    own = documents.get(evidence.get("prospectus_ref"), {})
    quotes = evidence.get("workday_valuation_quotes") or {}
    body = own.get("body", "")
    if (
        evidence.get("policy") != "QDII_CNY_VERIFIED_SESSION_V1"
        or evidence.get("future_valuation_policy") != QDII_WORKDAY_POLICY
        or own.get("kind") != "PROSPECTUS"
        or material.get("fund_code") not in own.get("codes", [])
        or not body
        or own.get("body_hash") != digest(body)
        or urlparse(own.get("url", "")).hostname not in {"www.cifm.com", "www.jpmorganam.com.cn"}
        or set(quotes) != {"workday", "daily", "holiday_close"}
        or any(not isinstance(quote, str) or not quote or quote not in body for quote in quotes.values())
    ):
        return False
    compact = {key: re.sub(r"\s+", "", quote) for key, quote in quotes.items()}
    # 定义、估值义务和无交易时的估价方法须同时存在，不能只拿一句旧收盘价替代。
    return bool(
        re.fullmatch(r"工作日[：:]指上海证券交易所[、，]深圳证券交易所开放交易的工作日", compact["workday"])
        and "基金管理人应在每个工作日对基金资产估值" in compact["daily"]
        and "暂停估值时除外" in compact["daily"]
        and "权益类证券" in compact["holiday_close"]
        and "估值日无交易" in compact["holiday_close"]
        and "最近交易日的市价（收盘价）估值" in compact["holiday_close"]
        and "重大变化" in compact["holiday_close"]
        and "公允价格" in compact["holiday_close"]
    )


def verified_gold_valuation_days(material: dict, documents: dict) -> bool:
    """黄金多周期必须另核本基金逐工作日估值条款，不能只延长一次报价窗口。"""
    evidence = material.get("calendar_evidence", {})
    own = documents.get(evidence.get("valuation_contract_ref"), {})
    quotes = evidence.get("valuation_day_quotes", {})
    if (
        evidence.get("policy") != "CN_GOLD_ETF_NAV_WINDOW_V1"
        or own.get("kind") not in {"PROSPECTUS", "ANNOUNCEMENT"}
        or material.get("fund_code") not in own.get("codes", [])
        or "黄金" not in own.get("title", "")
        or "联接" not in own.get("title", "")
        or urlparse(own.get("url", "")).hostname
        not in {"www.qhkyfund.com", "qhkyfund.com", "fund.eastmoney.com", "np-cnotice-fund.eastmoney.com"}
        or set(quotes) != {"working_day", "daily_valuation"}
        or any(not isinstance(q, str) or not q or q not in own.get("body", "") for q in quotes.values())
    ):
        return False
    working, daily = [re.sub(r"\s+", "", quotes[key]) for key in ("working_day", "daily_valuation")]
    return bool(
        re.search(r"工作日[：:]指上海证券交易所[、，,]深圳证券交易所的正常交易日", working)
        and "基金管理人应每个工作日对基金资产估值" in daily
        and "暂停估值时除外" in daily
    )


def verified_domestic_window(profile: dict, material: dict, documents: dict, window: dict) -> bool:
    """核对公开日历、投向和本期实际行情，不能凭名称放开QDII等跨境产品。

    日历证据只对冻结的比较基准日和目标日有效。港股通须取得港交所当年完整
    休市表；半日市也保守拒绝。境内黄金联接须核对目标ETF身份及本期人民币行情。
    原国内日历判断保持优先，历史研究和净值缺口计算不读取本补充包。
    """
    if domestic_calendar_supported(profile):
        return True
    description = " ".join(
        str(profile.get(k) or "") for k in ("fund_name", "source_fund_type", "benchmark", "invest_type")
    )
    # 港股通比较基准可以用港元计价，但境内基金份额仍以人民币估值；
    # 只有份额名称本身标为美元/港元时才按非人民币份额拒绝。
    if re.search(r"QDII|FOF|REIT|货币|原油|海外|越南|日本|纳斯达克|全球|标普石油", description, re.I) or re.search(
        r"美元|港元", profile.get("fund_name", "")
    ):
        return False
    evidence = material.get("calendar_evidence", {})
    if evidence.get("base_date") != window["base_nav_date"] or evidence.get("target_date") != window["target_nav_date"]:
        return False
    base, target = date.fromisoformat(window["base_nav_date"]), date.fromisoformat(window["target_nav_date"])
    if base.weekday() >= 5 or target.weekday() >= 5:
        return False
    report = documents.get(evidence.get("report_ref"), {})
    if report.get("kind") != "REPORT" or material["fund_code"] not in report.get("codes", []):
        return False
    mode = evidence.get("policy")
    if mode == "CN_NAV_DISCLOSED_DOMESTIC_ASSETS_V1":
        # “允许港股通”不等于本期实际持有港股。只有本基金完整原文能重建全部
        # 股票、全部属于境内市场，且自己的招募说明书定义相关交易场所估值日，
        # 才采用项目已核实的境内日历；不能只靠删掉材料包里的港股行来放行。
        from app.services.direction_1d_protocol import calendar
        from app.services.fund_rating_public_market import balance
        from app.services.fund_rating_public_stock_tables import parse_disclosed

        prospectus = documents.get(evidence.get("prospectus_ref"), {})
        if (
            prospectus.get("kind") != "PROSPECTUS"
            or material["fund_code"] not in prospectus.get("codes", [])
            or urlparse(prospectus.get("url", "")).hostname
            not in {"np-cnotice-fund.eastmoney.com", "www.founderff.com", "founderff.com"}
            or not re.search(r"估值日为本基金相关的证券[/／]期货交易场所的交易日", prospectus.get("body", ""))
        ):
            return False
        sessions, _ = calendar()
        if base not in sessions or target not in sessions:
            return False
        try:
            balances = balance([report["body"]], require_stock=False)
            positions = parse_disclosed([report["body"]], balances["stock_amount"], balances["net_assets"])["holdings"]
        except (ValueError, KeyError):
            return False
        codes = {row["stock_code"] for row in positions}
        companies = material.get("companies", [])
        company_codes = {company["code"] for company in companies}
        if company_codes != codes:
            # 完整中报仍必须全部保留。新季报的独立披露可以增加需核验的证券，
            # 不能因公司资料做了必要补充，就误把证券集合变化当作境外估值日历。
            # 额外证券仅允许来自本基金、本报告期原文的实际解析；仍不拼成新全仓。
            from app.services.market_holdings_parse import partial_stock_sections

            latest = material.get("source_manifest", {}).get("latest_partial_report", {})
            partial = latest.get("report", {})
            original = documents.get(latest.get("source_ref"), {})
            end = partial.get("endDate", "")
            compact = re.sub(r"\s+", "", original.get("body", ""))
            try:
                period = date.fromisoformat(end)
                rows, _ = partial_stock_sections(original["body"])
            except (ValueError, KeyError, TypeError):
                return False
            new_codes = {row["stockCode"] for row in rows}
            if (
                original.get("kind") != "REPORT"
                or material["fund_code"] not in original.get("codes", [])
                or material["fund_code"] not in compact
                or f"{period.year}年{period.month}月{period.day}日" not in compact
                or end <= material.get("report", {}).get("endDate", "")
                or end > str(base)
                or partial.get("fullDisclosure") is not False
                or new_codes != {row["stockCode"] for row in partial.get("holdings", [])}
                or company_codes != codes | new_codes
            ):
                return False
        return (
            bool(codes)
            and all(code.endswith((".SH", ".SZ", ".BJ")) for code in company_codes)
            and codes <= company_codes
            and all(
                company.get("quote", {}).get("date") == str(base)
                and company.get("quote", {}).get("change_pct") is not None
                for company in companies
            )
        )
    if mode == "CN_NAV_HK_CONNECT_WINDOW_V1":
        calendar_doc = documents.get(evidence.get("calendar_ref"), {})
        body = calendar_doc.get("body", "")
        if (
            profile.get("fund_type") not in {"STOCK", "MIXED", "BOND"}
            or urlparse(calendar_doc.get("url", "")).hostname != "www.hkex.com.hk"
            or calendar_doc.get("kind") != "EXCHANGE_CALENDAR"
            or "Holiday Schedule for Year " + str(target.year) not in body
            or "Saturdays, Sundays" not in body
            or base.year != target.year
            or not ("港股" in description or "港股通" in report.get("body", ""))
        ):
            return False
        months = {
            name: number
            for number, name in enumerate(
                (
                    "January",
                    "February",
                    "March",
                    "April",
                    "May",
                    "June",
                    "July",
                    "August",
                    "September",
                    "October",
                    "November",
                    "December",
                ),
                1,
            )
        }
        closed = {
            date(int(year), months[month], int(day))
            for day, month, year in re.findall(r"(\d{1,2})\s+(" + "|".join(months) + r")\s+(\d{4})", body)
        }
        if len({day for day in closed if day.year == target.year}) < 15 or target in closed:
            return False
        hong_kong = [company for company in material.get("companies", []) if company["code"].endswith(".HK")]
        if base in closed:
            # 香港休市不等于人民币基金不估值。只有本基金条款及官方日历、汇率和
            # 真实最近收盘同时可证实时，才接纳参考行情；日期和涨跌仍保留原交易日。
            from app.services.fund_valuation_sessions import verified_hk_holiday_reference

            return bool(hong_kong) and all(
                verified_hk_holiday_reference(material, documents, c["code"], c.get("quote") or {}, window)
                for c in hong_kong
            )
        return all(
            company.get("quote", {}).get("date") == str(base) and company.get("quote", {}).get("change_pct") is not None
            for company in hong_kong
        )
    if mode == "CN_GOLD_ETF_NAV_WINDOW_V1":
        chain = material.get("exposure_chain", [])
        if not re.search(r"黄金.*ETF.*联接", profile.get("fund_name", ""), re.I) or len(chain) < 2:
            return False
        target_code = chain[1]
        # 目标基金代码必须出现在本基金报告，并有对应的人民币场内ETF行情。
        if target_code not in report.get("body", "") or evidence.get("quote_currency") != "CNY":
            return False
        market = material.get("market", {})
        return bool(market) and all(
            code.split(".")[0] == target_code and quote.get("date") == str(base) and quote.get("change_pct") is not None
            for code, quote in market.items()
        )
    return False
