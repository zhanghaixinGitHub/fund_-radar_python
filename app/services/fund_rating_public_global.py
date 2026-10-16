"""人民币境外主动股票基金的原件适配：先核验本基金估值日，再计算八维。

不从境外股票价格拼接基金净值。人民币基金本身的已核验净值已体现汇率影响，
不能把另缺一列外汇价格当作无法计算历史风险的理由。其他资产或表格格式不猜测。
"""

import hashlib
import json
import re
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from app.core.config import get_settings
from app.services import fund_rating_public_market as market
from app.services.fund_rating_rules import decimal

BASIS = "PUBLIC_DISCLOSURE_GLOBAL_EQUITY_CNY_V1"
SCOPE = {"basis": BASIS, "asset": "GLOBAL_ACTIVE_EQUITY", "currency": "CNY"}


def global_policy() -> dict:
    """在真实综合试算前冻结，沿用主动权益八维权重和锚点。"""
    result = deepcopy(market.market_policy())
    result.update(basis=BASIS, scope=SCOPE, revision="public-global-equity-cny-2026-10-15-v1")
    result["input_contract"] = {
        "history": "OWN_CNY_NAV_ALL_REQUIRED_DATES_CROSS_CHECKED_WITH_ISSUER_HISTORY",
        "calendar": "FUND_VALUATION_DATES_VERIFIED_FROM_PUBLISHED_OWN_HISTORY_NOT_FOREIGN_STOCK_CALENDAR",
        "holdings": "ALL_FOREIGN_STOCKS_AND_SINGLE_REPORT_INDUSTRY_TAXONOMY_RECONCILED_TO_CNY_BALANCE",
        "unknowns": "REJECT_NO_FILL_NO_WEIGHT_REDISTRIBUTION",
    }
    return result


def positions(pages: list[str], expected: str) -> tuple[list, list]:
    """当前支持有证券市场后缀的海外表；编号、金额、行业三重对账阻止漏行。"""
    text = "\n".join(pages)
    table = market.section(text, r"7\.4\s*期末按公允价值", r"7\.5\s*报告期内权益")
    blocks = re.split(r"(?m)^[ \t]*(\d{1,3})[ \t\r\n]+(?=[A-Z])", table)[1:]
    records = []
    for number, block in zip(blocks[::2], blocks[1::2], strict=True):
        # 公司名称的断行不改变证券代码；金额和数量必须完整，不能截取前十名。
        match = re.search(
            r"\b([A-Z]{1,6}|\d{5})\s+(US|HK)\s+"
            r"(美国\s+美国|香港联\s*合交易\s*所\s+中国\s*香港)\s+"
            r"([\d,]+)\s+([\d,]+\.\d{2})\s+([\d.]+)",
            block,
        )
        if int(number) != len(records) + 1 or not match:
            raise ValueError("RATING_GLOBAL_COMPLETE_POSITION_ROW_UNVERIFIED")
        code = match[1] + "." + match[2]
        records.append(
            {
                "stock_code": code,
                "stock_name": market.compact(block[: match.start()]),
                "quantity_shares": match[4].replace(",", ""),
                "market_value_cny": match[5].replace(",", ""),
            }
        )
    if not records or len({r["stock_code"] for r in records}) != len(records):
        raise ValueError("RATING_GLOBAL_DUPLICATE_OR_EMPTY_POSITIONS")
    if abs(sum(decimal(r["market_value_cny"]) for r in records) - decimal(expected)) > Decimal(".02"):
        raise ValueError("RATING_GLOBAL_COMPLETE_STOCK_SUM")
    industry = market.section(text, r"7\.3\s*期末按行业", r"7\.4\s*期末按公允价值")
    if not re.search(r"采用彭博行业分类标准|采用全球行业分类标准", market.compact(industry)):
        raise ValueError("RATING_GLOBAL_UNIFIED_TAXONOMY_UNVERIFIED")
    rows = re.findall(r"(?m)^\s*([\u4e00-\u9fff-GICS]+)\s+([\d,]+\.\d{2}|-)\s+([\d.]+|-)\s*$", industry)
    industries = [
        {"code": name, "name": name, "value_cny": "0" if amount == "-" else amount.replace(",", "")}
        for name, amount, _ in rows
        if name != "合计"
    ]
    if not industries or abs(sum(decimal(r["value_cny"]) for r in industries) - decimal(expected)) > Decimal(".02"):
        raise ValueError("RATING_GLOBAL_COMPLETE_INDUSTRY_SUM")
    return records, industries


def facts(refs: dict, root: Path, code: str, name: str, as_of: date) -> dict:
    """分别核验已接入管理人的自身协议；新发行人不得套用既有响应格式。"""
    if code == "019449":
        from app.services.fund_rating_public_japan import facts as japan_facts

        return japan_facts(refs, root, code, name, as_of)
    if code != "017145":
        raise ValueError("RATING_GLOBAL_ISSUER_PROTOCOL_UNSUPPORTED")

    def read(label):
        ref = refs[label]
        raw = (root / ref["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            raise ValueError("RATING_GLOBAL_ORIGINAL_HASH")
        return raw

    report = market.public_document(read("half"), refs["half"], code, "中期报告")[0]
    summary = market.public_document(read("summary"), refs["summary"], code, "产品资料概要")[0]
    actual_name = "华宝海外新能源汽车股票发起式"
    for pages in (report, summary):
        text = market.compact("\n".join(pages))
        if code not in text or actual_name not in text or "QDII" not in text:
            raise ValueError("RATING_GLOBAL_IDENTITY")
    if market.publication(report) > str(as_of) or "2026年中期报告" not in market.compact("\n".join(report)):
        raise ValueError("RATING_GLOBAL_REPORT_PERIOD")
    provider = json.loads(read("provider"))
    if (
        provider.get("fund_code") != code
        or provider["source"]["enabled"] is not True
        or not provider["source"].get("authorization_verified_at")
    ):
        raise ValueError("RATING_GLOBAL_PROVIDER_AUTHORIZATION")
    names = [
        r["manager_name"]
        for r in provider["manager_records"]
        if r["begin_date"] and r["begin_date"] <= str(as_of) and (not r["end_date"] or r["end_date"] > str(as_of))
    ]
    info = market.key_information(summary, code, actual_name, None, names, profile_only=True)
    if info["published_on"] > str(as_of):
        raise ValueError("RATING_GLOBAL_FUTURE_SUMMARY")
    b = market.balance(report, require_stock=False)
    if any(decimal(b[k]) != 0 for k in ("fund_amount", "bond_amount")) or decimal(b["assets"]["衍生金融资产"]) != 0:
        raise ValueError("RATING_GLOBAL_OTHER_ASSETS_REQUIRE_REVIEW")
    stock, industries = positions(report, b["stock_amount"])
    disclosure = market.disclosures(report, stock)
    _, history = market.provider_history(provider, as_of)
    own = json.loads(read("issuer_nav"))
    if own.get("code") != "0000" or not isinstance(own.get("data"), list):
        raise ValueError("RATING_GLOBAL_NAV_RESPONSE")
    rows = own["data"]
    if len({r["navDate"] for r in rows}) != len(rows) or any(r["fundCode"] != code for r in rows):
        raise ValueError("RATING_GLOBAL_NAV_IDENTITY_OR_DUPLICATE")
    by_day = {r["navDate"]: r for r in rows}
    licensed = {r["nav_date"]: r for r in history["rows"]}
    for day in history["dates"]:
        if day not in by_day or decimal(by_day[day]["nav"]) != decimal(licensed[day]["unit_nav"]):
            raise ValueError("RATING_GLOBAL_OWN_VALUATION_DATE_OR_NAV_CONFLICT")
        if decimal(by_day[day]["totalNetValue"]) != decimal(by_day[day]["nav"]) or decimal(
            licensed[day]["adjusted_nav"]
        ) != decimal(licensed[day]["unit_nav"]):
            raise ValueError("RATING_GLOBAL_EVENTS_REQUIRE_RECONCILIATION")
    history["issuer_valuation_verification"] = {
        "fund_code": code,
        "currency": "CNY",
        "days_checked": len(history["dates"]),
        "source_dates": len(rows),
        "all_unit_navs_matched": True,
    }
    return {
        "key_information": info,
        "holdings": {
            "report_date": "2026-06-30",
            "published_on": market.publication(report),
            "positions": stock,
            "industries": industries,
            "balance": b,
            "net_assets": b["net_assets"],
        },
        "disclosure": disclosure,
        "history": history,
        "manager": market.provider_team(provider, info, date.fromisoformat(history["dates"][0]), as_of),
    }


def metrics(bundle: dict, as_of: date, frozen: datetime):
    if bundle["basis"] != BASIS:
        raise ValueError("RATING_GLOBAL_SCOPE")
    if bundle.get("fund_code") == "019449":
        from app.services.fund_rating_public_japan import metrics as japan_metrics

        return japan_metrics(bundle, as_of, frozen)
    return market.metrics({**bundle, "basis": market.BASIS}, as_of, frozen)


def verify_originals(bundle: dict) -> None:
    original = facts(
        {r["label"]: r for r in bundle["raw_archive"]},
        Path(get_settings().fund_insight_directory) / "ratings",
        bundle["fund_code"],
        bundle["fund_name_stem"],
        date.fromisoformat(bundle["as_of_date"]),
    )
    if any(original[k] != bundle[k] for k in original):
        raise ValueError("RATING_GLOBAL_ORIGINAL_FACT_CONFLICT")
