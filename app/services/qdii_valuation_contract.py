"""人民币QDII的已核验估值窗口；不能把境外盘中报价或代理汇率当成估值结果。"""

import json
import re
from datetime import date, datetime, time
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from app.services.fund_listed_product_contract import POLICY as MULTI_LISTED_POLICY
from app.services.fund_listed_product_contract import european_market_ready, verified_products

POLICY = "QDII_CNY_VERIFIED_SESSION_V1"
LISTED_FOF_POLICY = "QDII_CNY_LISTED_FUND_SESSION_V1"


def verified_qdii_window(
    profile: dict, material: dict, documents: dict, window: dict, *, as_of: datetime | None = None
) -> bool:
    """只核验本次冻结的人民币股票QDII，不替代预测审阅，也不更改旧结果。

    必须有本基金招募说明书中的估值日、收盘价和人民币中间价条款；境外行情
    分开记录当地交易日、收盘时刻及取得时刻。仅接受已闭市的比较基准日行情，
    目标日行情不进入资料。汇率使用官方人民币中间价，拒绝ECB等相关代理值。
    上市基金组合另须核验母基金估值条款及所列子基金发行人的底层投向；
    非上市基金的滞后净值、未核验的币种或其他未覆盖仓位不能冒称已经补齐。
    """
    evidence = material.get("calendar_evidence") or {}
    multi_listed = evidence.get("policy") == MULTI_LISTED_POLICY
    listed_fof = evidence.get("policy") in {LISTED_FOF_POLICY, MULTI_LISTED_POLICY}
    if evidence.get("policy") not in {POLICY, LISTED_FOF_POLICY, MULTI_LISTED_POLICY}:
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
    if multi_listed:
        # 七项完整母基金金额与币种先对账，再核对每只发行人/交易所原件。
        # 此分支不放宽原美国基金试点，也不接纳未核验的新产品或交易场所。
        if not verified_products(material, documents, evidence):
            return False
    elif listed_fof:
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
        # 只排除经正式原文核验、已经转为现金权利的原普通股票报价。
        # 其他持仓、指数、美元汇率及交易日历仍完整核验；不删除原持仓或重分配权重。
        from app.services.fund_corporate_actions import verified_security_quote_exception

        exceptions = material.get("market_quote_exceptions", {})
        companies = material.get("companies", [])
        if not isinstance(exceptions, dict) or not set(exceptions) <= {c["code"] for c in companies}:
            return False
        cash_exceptions = []
        quotes = []
        for company in companies:
            if company["code"] in exceptions:
                if as_of is None or as_of.tzinfo is None:
                    return False
                proof = verified_security_quote_exception(
                    {**material, "window": window, "as_of": as_of.isoformat()}, documents, company
                )
                if not proof or proof["kind"] != "CASH_MERGED_ISSUER_REGULAR_QUOTE_NOT_APPLICABLE":
                    return False
                cash_exceptions.append(proof)
            else:
                quotes.append(company.get("quote") or {})
        if cash_exceptions and "USD" not in currencies:
            return False
        quotes += list(material.get("market", {}).values())
        if not quotes or not material.get("companies"):
            return False
        for quote in quotes:
            market = quote.get("valuation_market")
            if multi_listed and market in {"UK", "CH"}:
                if (
                    quote.get("currency") != "USD"
                    or "USD" not in currencies
                    or quote.get("date") != base
                    or quote.get("session") != "REGULAR_CLOSE"
                    or quote.get("change_pct") is None
                    or not european_market_ready(market, quote, documents, evidence, base, target)
                ):
                    return False
                continue
            if market == "HK" and quote.get("date") != base:
                from app.services.fund_valuation_sessions import verified_hk_holiday_reference

                company = next((c for c in material.get("companies", []) if c.get("quote") == quote), None)
                if company and verified_hk_holiday_reference(material, documents, company["code"], quote, window):
                    continue
                return False
            currency, zone, hour, minute = {
                "US": ("USD", "America/New_York", 16, 0),
                "HK": ("HKD", "Asia/Hong_Kong", 16, 10),
                "CN": ("CNY", "Asia/Shanghai", 15, 0),
                "JP": ("JPY", "Asia/Tokyo", 15, 30),
            }[market]
            if quote.get("currency") != currency or (currency != "CNY" and currency not in currencies):
                return False
            if quote.get("date") != base or quote.get("session") != "REGULAR_CLOSE" or quote.get("change_pct") is None:
                return False
            close = datetime.combine(date.fromisoformat(base), time(hour, minute), ZoneInfo(zone))
            if market == "HK":
                # 港股收市竞价在16:08—16:10随机结束。用16:10核验整段交易已经完成，
                # 真实最后报价时刻仍单独保留；不能要求所有证券恰在16:00收盘。
                hours = documents.get(evidence.get("market_hours_ref"), {})
                if (
                    hours.get("kind") != "EXCHANGE_TRADING_HOURS"
                    or urlparse(hours.get("url", "")).hostname != "www.hkex.com.hk"
                    or not all(value in hours.get("body", "") for value in ("16:08", "16:10"))
                ):
                    return False
            if datetime.fromisoformat(quote["closed_at"]) != close:
                return False
            source = documents[quote["source_ref"]]
            if source.get("kind") != "MARKET_DATA" or datetime.fromisoformat(source["available_at"]) < close:
                return False
        markets = {q["valuation_market"] for q in quotes} | ({"US"} if cash_exceptions else set())
        for market in markets:
            if multi_listed and market in {"UK", "CH"}:
                # 上方逐条报价已经把当地时段、实际取得及本期两个日期一并复核。
                continue
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
                        # 只承认上述逐只核验过的香港基准日休市参考，目标日休市仍拒绝。
                        from app.services.fund_valuation_sessions import verified_hk_holiday_reference

                        hong_kong = [c for c in material.get("companies", []) if c["code"].endswith(".HK")]
                        if (
                            day != base
                            or not hong_kong
                            or not all(
                                verified_hk_holiday_reference(
                                    material, documents, c["code"], c.get("quote") or {}, window
                                )
                                for c in hong_kong
                            )
                        ):
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
