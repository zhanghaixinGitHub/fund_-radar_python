"""终止上市证券的普通成交价适用性；不推算换股股数、替代价格或基金公允价值。"""

import re
from datetime import date, datetime
from urllib.parse import urlparse

from app.services.direction_1d_protocol import digest


def verified_security_quote_exception(data: dict, documents: dict, company: dict) -> dict | None:
    """复用原境内吸并核验，现金收购另按SEC普通股原文严格核验。"""
    if company.get("code", "").endswith(".US"):
        from app.services.fund_cash_mergers import verified_cash_merged_security

        return verified_cash_merged_security(data, documents, company)
    return verified_absorbed_security(data, documents, company)


def verified_absorbed_security(data: dict, documents: dict, company: dict) -> dict | None:
    """仅识别本基金报告中、经上交所明确因吸收合并终止交易的原证券代码。

    普通成交价不再存在时，以同一证券的交易所决定和发行人实施条款核验市场
    状态；不是把旧价格更新到今天。该记录绑定当前基准与目标日，并强制分析
    引用完整事项及未知边界。缺原文、非吸收合并、普通停牌、范围不符或出现
    终止交易日以后的所谓成交价时均不能放行。
    """
    try:
        code, own = company["code"], data["fund_code"]
        item = data["market_quote_exceptions"][code]
        w = data["window"]
        if (
            not code.endswith(".SH")
            or item.get("purpose") != "REGULAR_TRADING_QUOTE_ONLY"
            or item.get("fund_code") != own
            or item.get("stock_code") != code
            or item.get("base_date") != w["base_nav_date"]
            or item.get("target_date") != w["target_nav_date"]
        ):
            return None
        effective, base = date.fromisoformat(item["effective_date"]), date.fromisoformat(w["base_nav_date"])
        quote = company.get("quote") or {}
        # 有普通成交价却同时声称已终止交易属于来源冲突，不借“不适用”忽略矛盾。
        if effective > base or (quote.get("date") and date.fromisoformat(quote["date"]) >= effective):
            return None
        exchange, issuer, report = [documents[item[key]] for key in ["exchange_ref", "issuer_ref", "report_ref"]]
        if (
            exchange.get("kind") != "ANNOUNCEMENT"
            or issuer.get("kind") != "ANNOUNCEMENT"
            or report.get("kind") != "REPORT"
            or code not in exchange.get("codes", [])
            or code not in issuer.get("codes", [])
            or own not in report.get("codes", [])
            or own not in report.get("body", "")
            or urlparse(exchange.get("url", "")).hostname != "www.sse.com.cn"
            or not urlparse(exchange["url"]).path.startswith("/disclosure/announcement/listing/stock/")
            or urlparse(issuer.get("url", "")).hostname != "static.cninfo.com.cn"
        ):
            return None
        as_of = datetime.fromisoformat(data["as_of"])
        if as_of.tzinfo is None:
            return None
        for document in [exchange, issuer, report]:
            at = datetime.fromisoformat(document["available_at"])
            if (
                at.tzinfo is None
                or at > as_of
                or document["published_date"] > as_of.date().isoformat()
                or document["published_date"] > at.date().isoformat()
                or digest(document["body"]) != document["body_hash"]
            ):
                return None
        quotes = [item[key] for key in ["exchange_quote", "issuer_quote", "position_quote"]]
        if any(not q or q not in d["body"] for q, d in zip(quotes, [exchange, issuer, report], strict=True)):
            return None
        clean = [re.sub(r"\s+", "", q) for q in quotes]
        when = f"{effective.year}年{effective.month}月{effective.day}日"
        if (
            code[:6] not in clean[0]
            or "因被吸收合并" not in clean[0]
            or f"自{when}起终止其股票在本所上市交易" not in clean[0]
            or code[:6] not in clean[1]
            or not all(term in clean[1] for term in ["换股", "登记日收市后", "另行刊登", "新增股份上市"])
            or code[:6] not in clean[2]
            or company["name"] not in clean[2]
        ):
            return None
        fact = data["facts"].get(item["fact_id"], {})
        if (
            fact.get("required_in_analysis") is not True
            or fact.get("direction_eligible") is not False
            or set(fact.get("source_refs", [])) != {exchange["id"], issuer["id"], report["id"]}
            or fact.get("quotes") != quotes
        ):
            return None
        return {
            "kind": "ABSORBED_ISSUER_REGULAR_QUOTE_NOT_APPLICABLE",
            "effective_date": str(effective),
            "base_date": str(base),
            "source_refs": [exchange["id"], issuer["id"], report["id"]],
            "required_fact_id": item["fact_id"],
            "original_quote_date": quote.get("date"),
            "replacement_quote_created": False,
            "fund_current_quantity_or_fair_value_proved": False,
        }
    except (KeyError, TypeError, ValueError):
        return None
