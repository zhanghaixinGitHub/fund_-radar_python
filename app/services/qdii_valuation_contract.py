"""人民币QDII的已核验估值窗口；不能把境外盘中报价或代理汇率当成估值结果。"""

import json
import re
from datetime import date, datetime, time
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

POLICY = "QDII_CNY_VERIFIED_SESSION_V1"
LISTED_FOF_POLICY = "QDII_CNY_LISTED_FUND_SESSION_V1"


def verified_qdii_window(profile: dict, material: dict, documents: dict, window: dict) -> bool:
    """只核验本次冻结的人民币股票QDII，不替代预测审阅，也不更改旧结果。

    必须有本基金招募说明书中的估值日、收盘价和人民币中间价条款；境外行情
    分开记录当地交易日、收盘时刻及取得时刻。仅接受已闭市的比较基准日行情，
    目标日行情不进入资料。汇率使用官方人民币中间价，拒绝ECB等相关代理值。
    上市基金组合另须核验母基金估值条款及所列子基金发行人的底层投向；
    非上市基金的滞后净值、未核验的币种或其他未覆盖仓位不能冒称已经补齐。
    """
    evidence = material.get("calendar_evidence") or {}
    listed_fof = evidence.get("policy") == LISTED_FOF_POLICY
    if evidence.get("policy") not in {POLICY, LISTED_FOF_POLICY}:
        return False
    description = str(profile.get("fund_name") or "")
    if not re.search("QDII", description, re.I) or re.search("美元|港元", description, re.I):
        return False
    if re.search("FOF|原油", description, re.I) and not listed_fof:
        return False
    if listed_fof and not re.search("FOF|原油", description, re.I):
        return False
    base, target = window["base_nav_date"], window["target_nav_date"]
    if evidence.get("base_date") != base or evidence.get("target_date") != target:
        return False
    if any(date.fromisoformat(day).weekday() >= 5 for day in (base, target)):
        return False
    prospectus = documents.get(evidence.get("prospectus_ref"), {})
    report = documents.get(evidence.get("report_ref"), {})
    code = material["fund_code"]
    if (
        prospectus.get("kind") != "PROSPECTUS"
        or code not in prospectus.get("codes", [])
        or report.get("kind") != "REPORT"
        or code not in report.get("codes", [])
    ):
        return False
    clauses = evidence.get("valuation_quotes", {})
    if set(clauses) != {"day", "close", "fx"} or any(
        not quote or quote not in prospectus.get("body", "") for quote in clauses.values()
    ):
        return False
    if not all(word in clauses[key] for key, word in (("day", "估值日"), ("close", "收盘价"), ("fx", "中间价"))):
        return False
    applicable_asset = "上市流通的基金" if listed_fof else "有价证券|权益类证券|股票"
    if not re.search(applicable_asset, clauses["close"]) or re.search("发售|募集期", clauses["fx"]):
        return False
    if not re.search("估值|外币资产价值|人民币对主要外汇", clauses["fx"]):
        return False
    if listed_fof:
        # 只承认实际披露且已经核验的上市基金价格；非上市基金的滞后净值仍不能替代。
        # 每只所列基金须有发行人资料解释其底层资产，期货展期和费用不能忽略。
        listed = evidence.get("listed_fund_evidence", {})
        companies = material.get("companies", [])
        if not companies or set(listed) != {c["code"] for c in companies}:
            return False
        for company in companies:
            item = listed[company["code"]]
            product = documents.get(item.get("product_ref"), {})
            parent_quote, product_quote = item.get("parent_quote", ""), item.get("product_quote", "")
            if (
                company.get("asset_type") != "LISTED_FUND"
                or not parent_quote
                or parent_quote not in report.get("body", "")
                or product.get("kind") != "FUND_PRODUCT_INFO"
                or not product_quote
                or product_quote not in product.get("body", "")
                or urlparse(product.get("url", "")).hostname != "www.uscfinvestments.com"
                or "NYSE Arca" not in product_quote
                or "futures" not in product.get("body", "").lower()
            ):
                return False
    if "联接" in description:
        chain = material.get("exposure_chain", [])
        target_report = documents.get(evidence.get("target_report_ref"), {})
        target_clause = evidence.get("target_etf_valuation_quote", "")
        if (
            len(chain) < 2
            or target_report.get("kind") != "REPORT"
            or chain[1] not in target_report.get("codes", [])
            or target_clause not in prospectus.get("body", "")
            or "目标 ETF" not in target_clause
            or "净值" not in target_clause
        ):
            return False
    fx = documents.get(evidence.get("fx_ref"), {})
    if fx.get("kind") != "FX_REFERENCE_DATA" or urlparse(fx.get("url", "")).hostname not in {
        "www.safe.gov.cn",
        "www.chinamoney.com.cn",
    }:
        return False
    try:
        fx_rows = json.loads(fx["body"])["rates"]
        currencies = {row["currency"] for row in fx_rows if row["date"] == base and float(row["cny_per_unit"]) > 0}
        quotes = [c.get("quote") or {} for c in material.get("companies", [])]
        quotes += list(material.get("market", {}).values())
        if not quotes or not material.get("companies"):
            return False
        for quote in quotes:
            market = quote.get("valuation_market")
            currency, zone, hour, minute = {
                "US": ("USD", "America/New_York", 16, 0),
                "HK": ("HKD", "Asia/Hong_Kong", 16, 0),
                "CN": ("CNY", "Asia/Shanghai", 15, 0),
                "JP": ("JPY", "Asia/Tokyo", 15, 30),
            }[market]
            if quote.get("currency") != currency or (currency != "CNY" and currency not in currencies):
                return False
            if quote.get("date") != base or quote.get("session") != "REGULAR_CLOSE" or quote.get("change_pct") is None:
                return False
            close = datetime.combine(date.fromisoformat(base), time(hour, minute), ZoneInfo(zone))
            if datetime.fromisoformat(quote["closed_at"]) != close:
                return False
            source = documents[quote["source_ref"]]
            if source.get("kind") != "MARKET_DATA" or datetime.fromisoformat(source["available_at"]) < close:
                return False
        markets = {q["valuation_market"] for q in quotes}
        for market in markets:
            if market == "CN":
                # 两个境内交易日已由主窗口的验哈希交易日历核对。
                continue
            calendar = documents[evidence["calendar_refs"][market]]
            if calendar.get("kind") != "EXCHANGE_CALENDAR":
                return False
            body, host = calendar.get("body", ""), urlparse(calendar.get("url", "")).hostname
            if market == "US":
                # 原始网页的四列节假日表须保留完整当年列，不能只摘“正常交易”。
                parsed = json.loads(body)
                if host != "www.nyse.com" or parsed["year"] != date.fromisoformat(target).year:
                    return False
                if len(parsed["closed_dates"]) < 10 or len(parsed["early_close_dates"]) < 2:
                    return False
                if any(day in parsed["closed_dates"] + parsed["early_close_dates"] for day in (base, target)):
                    return False
                if "Core Trading Session" not in parsed["source_text"] or "4:00 p.m." not in parsed["source_text"]:
                    return False
            elif market == "HK":
                if (
                    host != "www.hkex.com.hk"
                    or f"Holiday Schedule for Year {date.fromisoformat(target).year}" not in body
                ):
                    return False
                if "Saturdays, Sundays" not in body or len(re.findall(r"\d{1,2}\s+[A-Za-z]+\s+2026", body)) < 15:
                    return False
                for day in (base, target):
                    if date.fromisoformat(day).strftime("%d %B %Y").lstrip("0") in body:
                        return False
            elif market == "JP":
                parsed = json.loads(body)
                if host != "www.jpx.co.jp" or parsed["year"] != date.fromisoformat(target).year:
                    return False
                if len(parsed["closed_dates"]) < 20 or any(day in parsed["closed_dates"] for day in (base, target)):
                    return False
                if (
                    "3:30 pm" not in parsed["hours_source_text"]
                    or urlparse(parsed["hours_source_url"]).hostname != host
                ):
                    return False
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True
