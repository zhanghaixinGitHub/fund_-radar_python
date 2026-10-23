"""核验已完成现金收购后原普通股票报价的适用性，不推算基金收款或现值。"""

import re
from datetime import date, datetime
from decimal import Decimal
from urllib.parse import urlparse

from app.services.direction_1d_protocol import ZONE, digest

# 仅接纳实际读过原件的两项事件；新案例仍需先补原件及准入验证。
# merger_date 为收购交割日，first_non_trading 为原股票不再普通交易的首日。
# EA在交割日收盘后停止交易，因此不能将8月4日的真实成交误判为冲突。
CASES = {
    "EA.US": {
        "cik": "712515",
        "merger_date": "2026-08-04",
        "first_non_trading": "2026-08-05",
        "cash_pattern": r"receive\$(\d+(?:\.\d+)?)pershareincash,withoutinterest",
        "exit_phrase": "tradingofthecompanycommonstockonnasdaqwashaltedafterthecloseoftradingontheclosingdate",
    },
    "WBD.US": {
        "cik": "1437107",
        "merger_date": "2026-10-06",
        "first_non_trading": "2026-10-06",
        "cash_pattern": r"receiveanamountincashequalto\$(\d+(?:\.\d+)?),withoutinterest",
        "exit_phrase": "halttradingofwbdcommonstockonnasdaqpriortotheopeningoftradingontheclosingdate",
    },
}


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def verified_cash_merged_security(data: dict, documents: dict, company: dict) -> dict | None:
    """绑定本基金、本期窗口、原持仓行和SEC普通股交割原文。

    此核验只说明原代码的普通成交价不再适用。完整现金条款、停止交易和股东
    权利终止三者缺一不可，债券Form 25、收购计划及仅有停牌词语均不能采用。
    事项必须进入各周期审阅；不填零、不删除旧权重、不计算基金现有现金。
    """
    try:
        code = company["code"]
        case = CASES[code]
        item = data["market_quote_exceptions"][code]
        window = data["window"]
        first = date.fromisoformat(case["first_non_trading"])
        base = date.fromisoformat(window["base_nav_date"])
        as_of = datetime.fromisoformat(data["as_of"])
        if (
            as_of.tzinfo is None
            or company.get("asset_type", "STOCK") != "STOCK"
            or item.get("kind") != "US_COMPLETED_CASH_MERGER"
            or item.get("purpose") != "REGULAR_TRADING_QUOTE_ONLY"
            or item.get("fund_code") != data["fund_code"]
            or item.get("stock_code") != code
            or item.get("base_date") != window["base_nav_date"]
            or item.get("target_date") != window["target_nav_date"]
            or item.get("merger_date") != case["merger_date"]
            or item.get("first_non_trading_date") != str(first)
            or first > base
            or date.fromisoformat(data["report"]["endDate"]) >= first
        ):
            return None
        old_quote = company.get("quote") or {}
        if old_quote.get("date") and date.fromisoformat(old_quote["date"]) >= first:
            return None
        sec = documents[item["sec_ref"]]
        report = documents[item["report_ref"]]
        uri = urlparse(sec["url"])
        chain = set(data.get("exposure_chain", [data["fund_code"]]))
        report_fund = item["report_fund_code"]
        if (
            sec.get("kind") != "ANNOUNCEMENT"
            or code not in sec.get("codes", [])
            or uri.hostname != "www.sec.gov"
            or not re.fullmatch(rf"/Archives/edgar/data/{case['cik']}/\d+/[^/]+\.htm", uri.path)
            or "form8-k" not in _compact(sec["body"])
            or report.get("kind") != "REPORT"
            or report_fund not in chain
            or report_fund not in report.get("codes", [])
            or report_fund not in report["body"]
        ):
            return None
        # 与通用来源入口相同：取得、公开和正文摘要均不得越过真实截止。
        for document in (sec, report):
            available = datetime.fromisoformat(document["available_at"])
            published = date.fromisoformat(document["published_date"])
            known_public = document.get("published_at")
            when = datetime.fromisoformat(known_public) if known_public else None
            if (
                available.tzinfo is None
                or available > as_of
                or published > available.astimezone(ZONE).date()
                or published > as_of.astimezone(ZONE).date()
                or (published == as_of.astimezone(ZONE).date() and when is None)
                or (when is not None and (when.tzinfo is None or when > available or when > as_of))
                or digest(document["body"]) != document["body_hash"]
            ):
                return None
        keys = ["identity_quote", "completion_quote", "cash_quote", "exit_quote", "rights_quote"]
        quotes = [item[key] for key in keys]
        if any(not quote or quote not in sec["body"] for quote in quotes):
            return None
        position = item["position_quote"]
        if not position or position not in report["body"]:
            return None
        symbol = code.removesuffix(".US")
        if not re.search(rf"(?<![A-Za-z0-9]){symbol}\s+US(?![A-Za-z0-9])", position):
            return None
        quantity = Decimal(company["reported_quantity_shares"])
        numbers = [Decimal(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", position)]
        if quantity <= 0 or quantity not in numbers:
            return None
        identity, completion, cash, exit_quote, rights = map(_compact, quotes)
        merger_date = date.fromisoformat(case["merger_date"])
        day = merger_date.strftime("%B") + str(merger_date.day) + "," + str(merger_date.year)
        cash_value = re.search(case["cash_pattern"], cash)
        if (
            "commonstock" not in identity
            or "nasdaq" not in identity
            or not re.search(rf"\b{symbol}\b", item["identity_quote"])
            or day.lower() not in completion
            or "closingdate" not in completion
            or not ("mergedwithandinto" in completion or "completedthepreviouslyannouncedacquisition" in completion)
            or cash_value is None
            or Decimal(cash_value[1]) <= 0
            or "commonstock" not in cash
            or "convertedintotherightto" not in cash
            or case["exit_phrase"] not in exit_quote
            or "commonstock" not in rights
            or not ("ceasedtohaveanyrightsasshareholders" in rights or "ceasedtohaveanyrightsasstockholders" in rights)
            or "otherthan" not in rights
            or "mergerconsideration" not in rights
        ):
            return None
        fact = data["facts"].get(item["fact_id"], {})
        if (
            fact.get("required_in_analysis") is not True
            or fact.get("direction_eligible") is not False
            or set(fact.get("source_refs", [])) != {sec["id"], report["id"]}
            or fact.get("quotes") != quotes + [position]
        ):
            return None
        return {
            "kind": "CASH_MERGED_ISSUER_REGULAR_QUOTE_NOT_APPLICABLE",
            "merger_date": case["merger_date"],
            "first_non_trading_date": str(first),
            "base_date": str(base),
            "source_refs": [sec["id"], report["id"]],
            "required_fact_id": item["fact_id"],
            "cash_consideration_currency": "USD",
            "cash_consideration_per_share": cash_value[1],
            "original_quote_date": old_quote.get("date"),
            "replacement_quote_created": False,
            "cash_received_by_fund_proved": False,
            "fund_current_quantity_or_fair_value_proved": False,
        }
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return None
