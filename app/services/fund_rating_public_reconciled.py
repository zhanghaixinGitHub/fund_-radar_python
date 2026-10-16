"""指数基金收益来源冲突的独立口径：原件逐日核验，不放宽原有误差阈值。

发行人收益曲线必须与本份额单位净值和逐笔已实施分红再投资相符。原接入的复权值
仍原样归档；新的分析序列有独立哈希，不能回写成供应商的原始数据。
"""

import hashlib
import json
import re
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from app.core.config import get_settings
from app.services import fund_rating_public_index as index
from app.services.fund_rating_inputs import canonical
from app.services.fund_rating_public_market import compact, public_document, publication
from app.services.fund_rating_rules import decimal

BASIS = "PUBLIC_DISCLOSURE_DOMESTIC_INDEX_RECONCILED_V1"
SCOPE = {"basis": BASIS, "asset": "DOMESTIC_EQUITY_INDEX", "currency": "CNY"}


def reconciled_policy() -> dict:
    """沿用指数八维权重及锚点，仅明确新的收益证据准入，不迁就结果调参。"""
    config = deepcopy(index.index_policy())
    config.update(basis=BASIS, scope=SCOPE, revision="public-index-reconciled-2026-10-15-v1")
    config["input_contract"]["history"] = "ISSUER_WEALTH_AND_OWN_UNITS_AND_DISCLOSED_CASH_REINVESTMENT_RECONCILED_DAILY"
    config["input_contract"]["source_conflicts"] = (
        "PRESERVE_ORIGINAL_PROVIDER_VALUES_USE_SEPARATE_DERIVED_SERIES_NO_TOLERANCE_RELAXATION"
    )
    return config


def _cash_events(provider: dict, refs: dict, root: Path, code: str, as_of: date) -> dict:
    """目前核验通过的原件格式为万家 006730 的两个分红年度，其他格式拒绝猜测。"""
    if code != "006730":
        raise ValueError("RATING_RECONCILED_DIVIDEND_DOCUMENT_FORMAT_UNSUPPORTED")
    texts = {}
    for label, title in (("dividend_2025", "中期报告"), ("dividend_2026", "分红公告")):
        ref = refs[label]
        raw = (root / ref["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            raise ValueError("RATING_RECONCILED_DIVIDEND_HASH")
        pages, _ = public_document(raw, ref, code, title)
        if publication(pages) > str(as_of):
            raise ValueError("RATING_RECONCILED_FUTURE_DIVIDEND")
        text = compact("\n".join(pages))
        if code not in text or "万家中证500指数增强" not in text:
            raise ValueError("RATING_RECONCILED_DIVIDEND_IDENTITY")
        texts[label] = text
    # 表格列序和日期都来自原件；每十份金额除以十，而非误用基金总份额作分母。
    first = texts["dividend_2025"]
    first = first[first.rfind("6.4.11利润分配情况") :]
    first = first.split("6.4.12", 1)[0]
    c_table = first.split("万家中证500指数增强发起式C", 1)[-1]
    if not re.search(r"2025年1月10日.*?0\.1430", c_table):
        raise ValueError("RATING_RECONCILED_DIVIDEND_2025_TABLE")
    second = texts["dividend_2026"]
    if not all(
        q in second
        for q in (
            "单位：元/10份基金份额",
            "0.6000.560",
            "除息日2026年1月12日",
            "2026年1月12日除息后的基金份额净值",
            "006729006730",
        )
    ):
        raise ValueError("RATING_RECONCILED_DIVIDEND_2026_TABLE")
    expected = {"2025-01-10": Decimal(".0143"), "2026-01-12": Decimal(".056")}
    events = [r for r in provider["dividend_records"] if "2023-09-28" < r["ex_date"] <= str(as_of)]
    if len(events) != 2 or {r["ex_date"]: decimal(r["cash_dividend"]) for r in events} != expected:
        raise ValueError("RATING_RECONCILED_DIVIDEND_EVENT_CONFLICT")
    if any(r["process_status"] != "实施" for r in events):
        raise ValueError("RATING_RECONCILED_DIVIDEND_NOT_IMPLEMENTED")
    return expected


def reconcile(history: dict, issuer: dict, provider: dict, refs: dict, root: Path, code: str, as_of: date) -> dict:
    """每个估值日核对代码、单位净值、现金事件和累计收益，任何一天不符即拒绝。"""
    if refs["benchmark"].get("format") != "ISSUER_DAILY_BENCHMARK_WJ":
        raise ValueError("RATING_RECONCILED_ISSUER_SOURCE")
    raw = json.loads((root / refs["benchmark"]["file"]).read_bytes())
    units = {date.fromisoformat(r["tradeDate"]).isoformat(): decimal(r["unitNv"]) for r in raw["results"]}
    events = _cash_events(provider, refs, root, code, as_of)
    return reconcile_series(history, issuer, units, events)


def reconcile_series(history: dict, issuer: dict, units: dict, events: dict) -> dict:
    """可独立测试的逐日核对；不借用另一份额，不用放宽容差消除冲突。"""
    result = deepcopy(history)
    days = history["dates"]
    original = {r["nav_date"]: r for r in history["rows"]}
    wealth = [1 + decimal(issuer[d]["syNavRate"]) / 100 for d in days]
    if any(units.get(d) != decimal(original[d]["unit_nav"]) for d in days):
        raise ValueError("RATING_RECONCILED_UNIT_NAV_CONFLICT")
    expected = [Decimal(1)]
    for previous, current in zip(days, days[1:], strict=False):
        expected.append(expected[-1] * (units[current] + events.get(current, Decimal(0))) / units[previous])
    # 管理人公开累计收益百分数精确到四位小数；累计曲线两端的最坏舍入界。
    bound = Decimal(".000001") / min(wealth)
    difference = max(abs(w / wealth[0] - e) for w, e in zip(wealth, expected, strict=True))
    if difference > bound:
        raise ValueError("RATING_RECONCILED_CASH_REINVESTMENT_CONFLICT")
    provider_values = list(map(decimal, history["adjusted_nav"]))
    conflict = max(abs(v / provider_values[0] - w / wealth[0]) for v, w in zip(provider_values, wealth, strict=True))
    result["rows"] = []
    for day, value in zip(days, wealth, strict=True):
        row = {
            **original[day],
            "provider_adjusted_nav": original[day]["adjusted_nav"],
            "provider_content_hash": original[day]["content_hash"],
            "adjusted_nav": str(value),
            "derived_source": "ISSUER_WEALTH_RECONCILED_WITH_CASH_REINVESTMENT",
        }
        row["content_hash"] = hashlib.sha256(
            canonical({k: v for k, v in row.items() if k != "content_hash"}).encode()
        ).hexdigest()
        result["rows"].append(row)
    result["adjusted_nav"] = list(map(str, wealth))
    result["return_source_reconciliation"] = {
        "units_checked": len(days),
        "cash_events": {k: str(v) for k, v in events.items()},
        "maximum_cash_reinvestment_difference": str(difference),
        "rounding_bound": str(bound),
        "maximum_original_provider_difference": str(conflict),
        "original_provider_retained": True,
    }
    return result


def facts(refs: dict, root: Path, code: str, name: str, as_of: date) -> dict:
    return index.facts(refs, root, code, name, as_of, history_reconciler=reconcile)


def metrics(bundle: dict, as_of: date, frozen: datetime):
    if bundle["basis"] != BASIS:
        raise ValueError("RATING_RECONCILED_SCOPE")
    return index.metrics({**bundle, "basis": index.BASIS}, as_of, frozen)


def verify_originals(bundle: dict) -> None:
    original = facts(
        {r["label"]: r for r in bundle["raw_archive"]},
        Path(get_settings().fund_insight_directory) / "ratings",
        bundle["fund_code"],
        bundle["fund_name_stem"],
        date.fromisoformat(bundle["as_of_date"]),
    )
    if any(original[k] != bundle[k] for k in original):
        raise ValueError("RATING_RECONCILED_ORIGINAL_FACT_CONFLICT")
