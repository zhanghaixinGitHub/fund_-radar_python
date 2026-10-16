"""综合审阅的补充日历核验：仅支持已留原文的境内港股通及境内黄金ETF窗口。"""

import re
from datetime import date
from urllib.parse import urlparse

from app.services.nav_repair import domestic_calendar_supported


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
        if len({day for day in closed if day.year == target.year}) < 15 or base in closed or target in closed:
            return False
        hong_kong = [company for company in material.get("companies", []) if company["code"].endswith(".HK")]
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
