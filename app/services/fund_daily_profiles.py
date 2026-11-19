"""持仓资料档案和每日补件清单。"""

import json
import re
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path

from app.services.direction_1d_protocol import ZONE
from app.services.fund_exposure_common import now, read
from app.services.fund_materials_store import versioned_save
from app.services.material_content import source_exists

TEMPLATES = {
    "domestic": {
        "nav_source": "Tushare fund_nav 的本份额正式净值",
        "nav_fallback": "核对基金管理人或监管披露的本份额同期净值",
        "fx_source": "按本期实际投向核对，境内人民币资产无需换汇",
    },
    "overseas": {
        "nav_source": "Tushare fund_nav 的本份额正式净值",
        "nav_fallback": "所需日期未披露时，仅参考前一个已核实沪深交易日的本份额正式净值",
        "fx_source": "合同指定汇率；缺口保留待办",
    },
}
HORIZONS = ["T1", "T5_V1", "T20_V1", "M6_V1"]


@lru_cache(maxsize=1)
def profiles():
    path = Path(__file__).resolve().parents[1] / "data/daily_fund_profiles.json"
    return json.loads(path.read_text(encoding="utf-8"))


def confirmed_us_exception(code, day, contract):
    """只接受登记过的基金、具体日期及本份额现用条款。"""
    proof = profiles().get(code, {}).get("verified_valuation_dates", {})
    if str(day) not in proof.get("market_dates", {}) or proof.get("fund_code") != code:
        return None
    start, end = [date.fromisoformat(value) for value in proof["scope"]]
    clause = re.sub(r"\s+", "", proof["valuation_day_clause"])
    if start <= day <= end and clause in re.sub(r"\s+", "", contract.get("body", "")):
        return proof
    return None


def save_profiles(directory, rows):
    from app.services.nav_repair import domestic_calendar_supported

    records = {}
    lines = ["# 当前正份额持仓资料档案"]
    for row in sorted(rows, key=lambda row: row["fund_code"]):
        code = row["fund_code"]
        saved = profiles().get(code, {})
        entry = {
            **TEMPLATES.get(saved.get("template"), {}),
            **saved,
            "name": row["fund_name"],
            "benchmark": row.get("benchmark"),
            "share_currency": saved.get("share_currency"),
            "retry_time": "09:00 Asia/Shanghai",
        }
        if not entry.get("valuation_policy") and domestic_calendar_supported(row):
            entry["valuation_policy"] = "现有沪深官方日历规则"
        records[code] = entry
        quote_sources = [
            f"{key} {value or '待核'}" for key, value in entry.get("market_references", {}).items()
        ]
        lines.extend([
            "", f"## {code} {entry['name']}", "",
            f"- 份额币种 {entry['share_currency'] or '待核'}",
            f"- 底层基金 {', '.join(entry.get('target_funds', [])) or '见业绩基准及原报告'}",
            f"- 业绩基准 {entry['benchmark'] or '待核'}",
            f"- 行情来源 {', '.join(quote_sources) or '待核'}",
            f"- 净值来源 {entry.get('nav_source', '待核')}",
            f"- 净值备用条件 {entry.get('nav_fallback', '待核')}",
            f"- 估值日 {entry.get('valuation_clause') or entry.get('valuation_policy') or '待核本基金条款'}",
            f"- 汇率 {' '.join((entry.get('fx_clause') or entry.get('fx_source') or '待核').split())}",
            f"- 重查 {entry['retry_time']}",
        ])
        if evidence := entry.get("verified_valuation_dates"):
            lines.append(f"- 已核实估值日 {', '.join(evidence['market_dates'])}")
    versioned_save(directory / "fund-profiles.json", records)
    (directory / "fund-profiles.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_pending(directory, state):
    instant = now().astimezone(ZONE)
    window = state["window"]
    cutoff = datetime.fromisoformat(window["deadline_at"])
    retry = datetime.combine(date.fromisoformat(window["target_nav_date"]), time(9), ZONE)
    if instant >= retry:
        retry = instant + timedelta(minutes=30)
    current_profiles = read(directory / "fund-profiles.json")
    tasks = []

    def add(code, field, attempt, fallback):
        tasks.append({
            "fund_code": code,
            "field": field,
            "required_date": window["base_nav_date"],
            "affects": HORIZONS,
            "attempt": attempt,
            "next_source": fallback,
            "retry_at": retry.isoformat() if retry < cutoff else None,
        })

    for code in state["scope"]:
        item = state["funds"].get(code, {})
        profile = current_profiles.get(code, {})
        nav = item.get("base_nav_refresh", {})

        if item and nav.get("status") not in {"AVAILABLE", "BASE_NAV_PRESENT"}:
            add(code, "NAV", nav, profile.get("nav_fallback"))
        path = item.get("materials_path")
        if not path or not source_exists(Path(path)):
            add(code, "MATERIALS", item.get("failure"), "本份额的定期报告、招募说明书及净值披露")
            continue
        material = read(Path(path))
        docs = {doc["id"]: doc for doc in material["documents"]}
        quotes = {c["code"]: c.get("quote") or {} for c in material.get("companies", [])}
        quotes.update(material.get("market", {}))
        attempts = {
            a.get("security") or a.get("code"): a
            for key in ("market_attempts", "hk_attempts", "overseas_attempts")
            for a in item.get(key, []) or [] if isinstance(a, dict)
        }
        for symbol, quote in quotes.items():
            if quote.get("date") != window["base_nav_date"] or quote.get("change_pct") is None:
                source = docs.get(quote.get("source_ref"), {})
                add(code, "CLOSE:" + symbol, {"last_source": source.get("url"),
                                        "last_obtained_at": source.get("available_at"),
                                        "last_date": quote.get("date"),
                                        "latest_attempt": attempts.get(symbol)},
                    profile.get("market_references", {}).get(symbol) or source.get("url"))
        fx = next((row for row in item.get("overseas_attempts", [])
                   if isinstance(row, dict) and row.get("kind") == "OFFICIAL_FX"), None)
        if fx and fx.get("status") != "SUCCEEDED":
            add(code, "FX", fx, profile.get("fx_clause") or "核对本份额合同指定汇率")
        elif profile.get("valuation_policy") == "QDII_CNY_VN_CONTRACT_SESSION_V1":
            from app.services.vietnam_valuation_contract import assess_vietnam_valuation

            status = assess_vietnam_valuation({"fund_name": profile["name"]}, material, docs, window, as_of=instant)
            if not status["supported"]:
                add(code, "VALUATION_CALENDAR", status["gaps"], profile.get("contract_source"))
            if not status["fx_ready"]:
                add(code, "FX:VND/USD", status["gaps"], "估值日伦敦 16:00 彭博折算率，或合同指定的托管人来源")
    versioned_save(
        directory / "supplement-tasks.json",
        {"window": window, "checked_at": instant.isoformat(), "tasks": tasks},
    )
