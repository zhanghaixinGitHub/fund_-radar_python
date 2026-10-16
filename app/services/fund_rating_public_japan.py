"""019449 的公开原件适配，沿用已固定的境外主动权益八维规则。

基金自身人民币净值已经包含汇率影响。日期以管理人公开净值列表为准，
与已授权同份额净值逐日交叉核验，不把日本休市日补成境内估值日。
"""

import hashlib
import json
import re
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from app.services import fund_rating_public_market as market
from app.services.fund_rating_inputs import digest
from app.services.fund_rating_metrics import history_metrics, standard_cost
from app.services.fund_rating_rules import decimal

CODE = "019449"
PROFILE = "CIFM_JAPAN_CNY_OWN_VALUATIONS_V1"
NAME = "摩根日本精选股票型证券投资基金"


def positions(pages: list[str], expected: str) -> tuple[list[dict], list[dict]]:
    """逐行读取全部日本证券，并与独立行业表及资产负债表按人民币分核对。"""
    text = "\n".join(pages)
    table = market.section(text, r"7\.4\s*期末按公允", r"7\.5\s*报告期内权益")
    blocks = re.split(r"(?m)^\s*(\d{1,3})\s+(?=[A-Z])", table)[1:]
    records = []
    for number, block in zip(blocks[::2], blocks[1::2], strict=True):
        # PDF把“日本交易所”折至相邻行；只接受报告中明示日本、当地四位代码的行。
        match = re.search(
            r"\b([0-9A-Z]{4})\s+(?:日本交|易所)\s+日本\s+"
            r"([\d,]+)\s+([\d,]+\.\d{2})\s+([\d.]+)",
            block,
        )
        if int(number) != len(records) + 1 or not match:
            raise ValueError("RATING_JAPAN_COMPLETE_STOCK_ROW")
        ticker, quantity, amount, _ = match.groups()
        # 原文公司英文名跨行跨序号，使用原文明示证券代码作名称，避免拼错公司名称。
        records.append(
            {
                "stock_code": ticker + ".JP",
                "stock_name": ticker,
                "quantity_shares": quantity.replace(",", ""),
                "market_value_cny": amount.replace(",", ""),
            }
        )
    if not records or len({p["stock_code"] for p in records}) != len(records):
        raise ValueError("RATING_JAPAN_STOCK_EMPTY_OR_DUPLICATE")
    total = sum(decimal(p["market_value_cny"]) for p in records)
    if abs(total - decimal(expected)) > Decimal(".02"):
        raise ValueError("RATING_JAPAN_COMPLETE_STOCK_SUM")
    table = market.section(text, r"7\.3\s*期末按行业", r"7\.4\s*期末按公允")
    if "全球行业分类标准(GICS)" not in market.compact(table):
        raise ValueError("RATING_JAPAN_INDUSTRY_TAXONOMY")
    rows = re.findall(r"(?m)^\s*([\u4e00-\u9fffGICS-]+)\s+([\d,]+\.\d{2}|-)\s+([\d.]+|-)\s*$", table)
    industries = [
        {"code": name, "name": name, "value_cny": amount.replace(",", "")}
        for name, amount, _ in rows
        if name != "合计" and amount != "-"
    ]
    if not industries or abs(sum(decimal(r["value_cny"]) for r in industries) - total) > Decimal(".02"):
        raise ValueError("RATING_JAPAN_COMPLETE_INDUSTRY_SUM")
    return records, industries


def disclosures(pages: list[str]) -> dict:
    """每个零值均须对应原报告明确的“无”，不从章节缺失或其他否定句推断。"""
    text = "\n".join(pages)
    governance = market.compact(market.section(text, r"10\.6\.1\s*管理人受调查", r"10\.6\.2"))
    if not re.search(r"报告期内[，,]?管理人未受调查或处罚[。.]?$", governance):
        raise ValueError("RATING_JAPAN_GOVERNANCE_UNVERIFIED")
    strategy = market.compact(market.section(text, r"10\.4\s*基金投资策略", r"10\.5"))
    if "本报告期内无基金投资策略的改变。" not in strategy:
        raise ValueError("RATING_JAPAN_STRATEGY_UNVERIFIED")
    holder = market.compact(market.section(text, r"11\.1\s*报告期内单一投资者", r"11\.2"))
    if not holder.endswith("达到或超过20%的情况无。"):
        raise ValueError("RATING_JAPAN_HOLDER_UNVERIFIED")
    clauses = [
        (r"6\.4\.12\.1\s*因认购", r"6\.4\.12\.2", "流通受限证券"),
        (r"6\.4\.12\.2\s*期末持有", r"6\.4\.12\.3", "流通受限股票"),
        (r"6\.4\.12\.3\.1\s*银行间", r"6\.4\.12\.3\.2", "银行间市场债券正回购"),
        (r"6\.4\.12\.3\.2\s*交易所", r"6\.4\.12\.4", "交易所市场债券正回购"),
        (r"6\.4\.12\.4\s*期末参与", r"6\.4\.13", "证券出借业务的证券"),
    ]
    for start, end, heading_end in clauses:
        clause = market.compact(market.section(text, start, end))
        if clause.split(heading_end, 1)[-1] != "无。":
            raise ValueError("RATING_JAPAN_RESTRICTION_UNVERIFIED:" + heading_end)
    return {
        "report_date": "2026-06-30",
        "governance_state": 2,
        "strategy_continuity": 1,
        "holder_risk_band": 0,
        "restricted_stock_amount": "0",
        "restricted_rows": [],
    }


def issuer_history(provider: dict, detail: dict, nav: dict, as_of: date) -> dict:
    """以本份额完整官方列表确定估值日期，每一行必须与已授权原始净值一致。"""
    info = (detail.get("data") or {}).get("product_info") or {}
    if (
        detail.get("code") != 0
        or info.get("product_code") != CODE
        or info.get("product_legal_name") != NAME + "(QDII)"
        or info.get("money_type") != "0"
    ):
        raise ValueError("RATING_JAPAN_ISSUER_PRODUCT_IDENTITY")
    product_id = info["product_id"]
    envelope = ((nav.get("data") or {}).get("product_nav") or {}).get(product_id) or {}
    rows = envelope.get("list")
    if (
        nav.get("code") != 0
        or envelope.get("pages") != 1
        or envelope.get("pageNum") != 1
        or not isinstance(rows, list)
        or envelope.get("total") != len(rows)
        or envelope.get("pageSize", 0) < len(rows)
    ):
        raise ValueError("RATING_JAPAN_ISSUER_HISTORY_TRUNCATED")
    if len({r["nav_date"] for r in rows}) != len(rows) or any(r["product_id"] != product_id for r in rows):
        raise ValueError("RATING_JAPAN_ISSUER_HISTORY_IDENTITY_OR_DUPLICATE")
    own = {date(int(r["nav_date"][:4]), int(r["nav_date"][4:6]), int(r["nav_date"][6:8])).isoformat(): r for r in rows}
    # 37个月末明确来自同一份额的官方完整序列；不复用境内交易日历或其他份额历史。
    end_month = as_of.year * 12 + as_of.month - 1
    months = [(m // 12, m % 12 + 1) for m in range(end_month - 36, end_month + 1)]
    month_ends = []
    for year, month in months:
        selected = [d for d in own if d[:7] == f"{year:04d}-{month:02d}" and d <= str(as_of)]
        if not selected:
            raise ValueError("RATING_JAPAN_MONTH_END_MISSING")
        month_ends.append(max(selected))
    dates = sorted(d for d in own if month_ends[0] <= d <= str(as_of))
    if dates[-1] != str(as_of) or (as_of - date.fromisoformat(dates[0])).days < 3 * 365:
        raise ValueError("RATING_JAPAN_HISTORY_UNDER_THREE_YEARS")
    if date.fromisoformat(info["found_date"]) > date.fromisoformat(dates[0]):
        raise ValueError("RATING_JAPAN_SHARE_CLASS_AGE")
    source = provider.get("source") or {}
    if (
        provider.get("fund_code") != CODE
        or source.get("enabled") is not True
        or not source.get("authorization_verified_at")
    ):
        raise ValueError("RATING_JAPAN_PROVIDER_IDENTITY_OR_AUTHORIZATION")
    licensed = {r["nav_date"]: r for r in provider["history"]["rows"]}
    if len(licensed) != len(provider["history"]["rows"]):
        raise ValueError("RATING_JAPAN_PROVIDER_DUPLICATE")
    for day in dates:
        actual, stored = own[day], licensed.get(day)
        if not stored or not stored.get("content_hash"):
            raise ValueError("RATING_JAPAN_LICENSED_NAV_MISSING")
        if decimal(actual["relate_price"]) != decimal(stored["unit_nav"]):
            raise ValueError("RATING_JAPAN_NAV_SOURCE_CONFLICT")
        if (
            decimal(actual["cumulative_net"]) != decimal(actual["relate_price"])
            or decimal(actual["adjust_nav"]) != decimal(actual["relate_price"])
            or decimal(stored["adjusted_nav"]) != decimal(stored["unit_nav"])
        ):
            raise ValueError("RATING_JAPAN_EVENTS_REQUIRE_RECONCILIATION")
    return {
        "valuation_profile": PROFILE,
        "dates": dates,
        "month_ends": month_ends,
        "rows": [licensed[d] for d in dates],
        "calendar_hash": digest(dates),
        "adjusted_nav": [str(licensed[d]["adjusted_nav"]) for d in dates],
        "unit_nav": [str(licensed[d]["unit_nav"]) for d in dates],
        "issuer_valuation_verification": {
            "fund_code": CODE,
            "currency": "CNY",
            "days_checked": len(dates),
            "official_total_rows": len(rows),
            "all_unit_navs_matched": True,
        },
    }


def history_values(history: dict, as_of: date) -> dict:
    """相同已验证八维历史指标公式；仅日期集合来自本基金已核验估值日期。"""
    days = list(map(date.fromisoformat, history["dates"]))
    months = list(map(date.fromisoformat, history["month_ends"]))
    rows = {r["nav_date"]: r for r in history["rows"]}
    if (
        history.get("valuation_profile") != PROFILE
        or not days
        or len(months) != 37
        or days != sorted(set(days))
        or days[-1] != as_of
        or set(rows) != set(history["dates"])
        or history["calendar_hash"] != digest(history["dates"])
        or any(str(d) not in rows for d in months)
    ):
        raise ValueError("RATING_JAPAN_VALUATION_DATES_INVALID")
    values = [decimal(rows[str(d)]["adjusted_nav"]) for d in days]
    if any(v <= 0 for v in values):
        raise ValueError("RATING_JAPAN_NAV_NONPOSITIVE")
    result = history_metrics(days, values, [], months, 252, require_sharpe=False)
    result.pop("sharpe")
    monthly = [decimal(rows[str(d)]["adjusted_nav"]) for d in months]
    result["rolling_positive"] = Decimal(sum(v > 0 for v in result["rolling"])) / len(result["rolling"])
    result["monthly_positive"] = Decimal(sum(b > a for a, b in zip(monthly, monthly[1:], strict=False))) / 36
    result["calmar"] = (
        result["return_3y"] / result["drawdown"]
        if result["drawdown"]
        else Decimal(2 if result["return_3y"] > 0 else 0 if result["return_3y"] == 0 else -1)
    )
    return result


def facts(refs: dict, root: Path, code: str, name: str, as_of: date) -> dict:
    """从绑定身份、哈希及公布日期的原件重建事实，拒绝不适用的发行人或资产。"""
    if code != CODE or name != NAME:
        raise ValueError("RATING_JAPAN_SCOPE")

    def read(label):
        ref = refs[label]
        raw = (root / ref["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            raise ValueError("RATING_JAPAN_ORIGINAL_HASH")
        return raw

    report = market.public_document(read("half"), refs["half"], CODE, "中期报告")[0]
    summary = market.public_document(read("summary"), refs["summary"], CODE, "产品资料概要")[0]
    prospectus = market.public_document(read("prospectus"), refs["prospectus"], CODE, "招募说明书")[0]
    report_text = market.compact("\n".join(report))
    if CODE not in report_text or NAME not in report_text or "2026年中期报告" not in report_text:
        raise ValueError("RATING_JAPAN_REPORT_IDENTITY")
    if market.publication(report) > str(as_of):
        raise ValueError("RATING_JAPAN_REPORT_AFTER_CUTOFF")
    provider = json.loads(read("provider"))
    managers = sorted(
        {
            r["manager_name"]
            for r in provider["manager_records"]
            if r["begin_date"] and r["begin_date"] <= str(as_of) and (not r["end_date"] or r["end_date"] > str(as_of))
        }
    )
    info = market.key_information(summary, CODE, NAME, prospectus, managers, profile_only=True)
    if info["published_on"] > str(as_of):
        raise ValueError("RATING_JAPAN_SUMMARY_AFTER_CUTOFF")
    balance = market.balance(report, require_stock=False)
    if (
        any(decimal(balance[k]) != 0 for k in ("fund_amount", "bond_amount"))
        or decimal(balance["assets"]["衍生金融资产"]) != 0
    ):
        raise ValueError("RATING_JAPAN_OTHER_ASSETS_REQUIRE_REVIEW")
    stocks, industries = positions(report, balance["stock_amount"])
    history = issuer_history(provider, json.loads(read("issuer_detail")), json.loads(read("issuer_nav")), as_of)
    return {
        "key_information": info,
        "holdings": {
            "report_date": "2026-06-30",
            "published_on": market.publication(report),
            "positions": stocks,
            "industries": industries,
            "balance": balance,
            "net_assets": balance["net_assets"],
        },
        "disclosure": disclosures(report),
        "history": history,
        "manager": market.provider_team(provider, info, date.fromisoformat(history["dates"][0]), as_of),
    }


def metrics(bundle: dict, as_of: date, frozen: datetime):
    """沿用境外主动权益的全部指标、权重与锚点，未引入日本专用加分。"""
    from app.services.fund_rating_public_inputs import public_valid_until

    if (
        bundle.get("fund_code") != CODE
        or bundle.get("source_profile") != PROFILE
        or bundle.get("basis") != "PUBLIC_DISCLOSURE_GLOBAL_EQUITY_CNY_V1"
        or bundle.get("family") != "ACTIVE_EQUITY"
        or bundle.get("currency") != "CNY"
        or bundle.get("as_of_date") != str(as_of)
    ):
        raise ValueError("RATING_JAPAN_SCOPE")
    result = history_values(bundle["history"], as_of)
    holding, disclosure, manager = (bundle[k] for k in ("holdings", "disclosure", "manager"))
    total = sum(decimal(p["market_value_cny"]) for p in holding["positions"])
    if (
        total <= 0
        or abs(total - decimal(holding["balance"]["stock_amount"])) > Decimal(".02")
        or abs(sum(decimal(p["value_cny"]) for p in holding["industries"]) - total) > Decimal(".02")
    ):
        raise ValueError("RATING_JAPAN_STOCK_OR_INDUSTRY_TOTAL")
    net = decimal(holding["net_assets"])
    result.update(
        issuer_hhi=sum((decimal(p["market_value_cny"]) / total) ** 2 for p in holding["positions"]),
        industry_hhi=sum((decimal(p["value_cny"]) / total) ** 2 for p in holding["industries"]),
        net_assets=net,
        cash_ratio=decimal(holding["balance"]["assets"]["货币资金"]) / net,
        restricted_ratio=decimal(disclosure["restricted_stock_amount"]) / net,
    )
    result.update({k: decimal(disclosure[k]) for k in ("holder_risk_band", "strategy_continuity", "governance_state")})
    result.update({k: decimal(manager[k]) for k in ("team_months", "team_changes")})
    fees = bundle["key_information"]["fees"]
    result["cost_1y"] = standard_cost(
        0,
        0,
        0,
        fees["subscription"],
        fees["redemption"],
        mode=fees["subscription_mode"],
        operating_cost=fees["operating_cost"],
    )
    checked = datetime.fromisoformat(bundle["documents_checked_at"])
    expiry = public_valid_until(as_of, date.fromisoformat(holding["report_date"]), checked, frozen)
    return (
        result,
        {
            "nav": str(as_of),
            "holdings": holding["report_date"],
            "fees": checked.date().isoformat(),
            "manager": checked.date().isoformat(),
        },
        expiry,
    )
