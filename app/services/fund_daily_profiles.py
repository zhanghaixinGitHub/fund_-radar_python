"""逐基金补数清单的固定项目、取数来源和当天勾选状态。

每只当前持仓基金长期使用同一份清单。固定项目和来源由基金档案维护，
发现漏项或来源变更时更新档案。清单只显示当天的勾选状态，不追加往日流水。
任务核实所需日期的数据后才打勾；仍缺数据、来源或日期未核实的项目留空。
"""

import json
import re
from datetime import date, datetime, time, timedelta
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.services.direction_1d_protocol import ZONE
from app.services.fund_exposure_common import _publish, now, read
from app.services.fund_materials_store import versioned_save
from app.services.fund_supplement_requirements import (
    FX_SOURCE, SECTIONS, checklist_requirements, disclosure_directory, nav_sources, section_for,
)
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
SOURCE_RESULTS = {
    "ADDRESS_CHANGED": "原地址已确认变更",
    "NOT_PUBLISHED": "所需日期尚未披露",
    "TEMPORARY_FAILURE": "本次暂未取得",
    "ACCESS_REQUIRED": "需要授权或原件",
    "MISMATCH": "返回数据与所需口径不符",
}


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
    configured = profiles()
    lines = ["# 当前正份额持仓资料档案"]
    for row in sorted(rows, key=lambda row: row["fund_code"]):
        code = row["fund_code"]
        saved = configured.get(code, {})
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
    return records


def save_fund_checklist(folder, code, profile, window, day, tasks=(), ready=(), sources=None):
    """只替换当天清单。每个勾都必须有本次核对结果。"""
    if not re.fullmatch(r"\d{6}", code):
        raise ValueError("INVALID_DAILY_FUND_CODE")
    folder.mkdir(parents=True, exist_ok=True)
    sources = sources or {}
    pending = {task["field"]: task for task in tasks}
    lines = [
        f"# {code} {profile['name']}补数清单",
        "",
        f"检查日期 {day} · 基准净值日 {window['base_nav_date']} · 目标净值日 {window['target_nav_date']}",
        "",
        "勾选表示今天已核对所需时点、口径及实际资料。待核项目用于每日补数，达到验收条件才打勾。",
        "",
    ]

    def item(key, label, source, requirement=None):
        mark = "✓ 已核实" if key in ready and key not in pending else "□ 待核实"
        lines.append(f"- **{mark}**　{label}。来源 {source or '待核'}。")
        if requirement and requirement.get("verification"):
            lines.append(f"  - 时点或区间 {requirement['data_window']}。验收 {requirement['verification']}。")
        if key in pending and pending[key].get("next_source"):
            next_source = pending[key]["next_source"]
            if next_source != source:
                lines.append(f"  - 待核位置 {next_source}")
        followup = pending.get(key, {}).get("source_review")
        if followup:
            lines.append(f"  - 本次取件 {SOURCE_RESULTS[followup['status']]}。{followup['summary']}。")

    if profile.get("benchmark"):
        lines.extend([f"合同基准 {profile['benchmark']}。", ""])
    if note := profile.get("supplement_focus", {}).get("scope_note"):
        lines.extend([note, ""])
    lines.extend(["## 共同资料", ""])
    item("NAV", f"本份额正式净值（`{code}.OF`）", nav_sources(code))
    if profile.get("nav_fallback"):
        lines.append(f"  - 备用条件 {profile['nav_fallback']}。")
    notices = disclosure_directory(code)
    item("MATERIALS", "本份额最新报告和披露持仓", "；".join(filter(None, (profile.get("report_source"), notices))))
    item("NOTICES", "本份额公告目录及适用公告", notices)
    valuation = profile.get("valuation_clause") or profile.get("valuation_policy")
    item("VALUATION_CALENDAR", "基金合同和适用的估值日规则",
         "；".join(filter(None, (profile.get("contract_source") or notices,
                                 " ".join(valuation.split()) if valuation else None,
                                 "核实际市场的交易所日历"))))
    if profile.get("template") != "domestic" or "FX" in pending or "FX:VND/USD" in pending:
        fx = profile.get("fx_clause") or profile.get("fx_source")
        key = "FX:VND/USD" if "FX:VND/USD" in pending else "FX"
        item(key, "估值使用的汇率", f"{FX_SOURCE}；" + (" ".join(fx.split()) if fx else "核本份额合同指定渠道"))
    for target in profile.get("target_funds", []):
        item("TARGET:" + target, f"底层基金 `{target}` 最新报告及持仓",
             disclosure_directory(target))
    registered = profile.get("market_references", {})
    retired = profile.get("retired_market_references", {})
    pending_symbols = {key.split(":", 1)[1] for key in pending if key.startswith("CLOSE:")}
    symbols = set(registered) | set(retired) | set(sources) | pending_symbols
    def quote_item(symbol):
        retirement = retired.get(symbol)
        if retirement and retirement["effective_date"] <= window["base_nav_date"]:
            description = retirement.get("description", "已终止交易，核对摘牌及换股")
            item("CLOSE:" + symbol, f"`{symbol}` {description}", retirement["source"])
            return
        source = sources.get(symbol) or registered.get(symbol)
        if source and symbol == "VN30":
            url = urlsplit(source)
            params = parse_qsl(url.query)
            if {"from", "to"} <= {key for key, _ in params}:
                query = urlencode([(key, value) for key, value in params if key not in {"from", "to"}])
                source = urlunsplit(url._replace(query=query)) + "（from、to 按所需行情日填写）"
        item("CLOSE:" + symbol, f"`{symbol}` 基准日行情", source)
    for symbol in sorted(set(registered) | set(retired)):
        quote_item(symbol)
    requirements = checklist_requirements(code, profile)
    for section in SECTIONS:
        lines.extend(["", f"## {section}", ""])
        for requirement in requirements:
            use = requirement["use"]
            if section_for(use) == section:
                label = requirement["item"]
                if " · " in use:
                    label = use.split(" · ", 1)[1] + " · " + label
                item(requirement.get("id", requirement["item"]), label, requirement["source"], requirement)
    if securities := sorted(symbols - set(registered) - set(retired)):
        lines.extend(["", "## 逐证券行情", "",
                      "这是资料包的行情取数范围。对勾核的是行情日期和来源。实际持仓及权重以对应日期的基金披露核对。", ""])
        for symbol in securities:
            quote_item(symbol)
    _publish(folder / f"{code}.md", ("\n".join(lines) + "\n").encode("utf-8"), replace=True)


def reset_fund_checklists(directory, records, window, day):
    """新的一天沿用固定清单，只清空勾选；同一天续办使用原有核对状态。"""
    folder = directory.parent / "fund-supplements"
    for code, profile in sorted(records.items()):
        save_fund_checklist(folder, code, profile, window, day)


def read_source_reviews(directory, day, window):
    """每次只应用同一天和同一估值窗口的实取记录。"""
    path = directory / "source-rechecks.json"
    if not path.is_file():
        return {}
    value = read(path)
    if value.get("day") != day or any(
        value.get(key) != window[key] for key in ("base_nav_date", "target_nav_date")
    ):
        return {}
    return {
        (row["fund_code"], row["field"]): {
            **row, "summary": " ".join(row["summary"].split()).rstrip("。"),
        }
        for row in value["items"]
        if row.get("status") in SOURCE_RESULTS and row.get("summary")
        and (row.get("attempted_at") or "")[:10] == day and row.get("attempted_source")
    }


def save_pending(directory, state, *, records=None, fund_code=None, aggregate=True):
    """用当天的取数核对结果重绘清单，原有业务回执仍供续办和校验使用。"""
    instant = now().astimezone(ZONE)
    window = state["window"]
    cutoff = datetime.fromisoformat(window["deadline_at"])
    retry = datetime.combine(date.fromisoformat(window["target_nav_date"]), time(9), ZONE)
    if instant >= retry:
        retry = instant + timedelta(minutes=30)
    current_profiles = read(directory / "fund-profiles.json") if records is None else records
    tasks = []
    folder = directory.parent / "fund-supplements"
    day = instant.date().isoformat()
    source_reviews = read_source_reviews(directory, day, window)
    month_end = (instant.date().replace(day=1) - timedelta(days=1)).isoformat()
    ratings = state.get("ratings") or {}
    by_rating = {r["fundCode"]: r for r in ratings.get("readback", [])}
    advice = {r["fund_code"]: r for r in (state.get("advice") or {}).get("items", [])}

    def add(code, field, attempt, fallback, **details):
        task = {
            "fund_code": code,
            "field": field,
            "required_date": window["base_nav_date"],
            "affects": HORIZONS,
            "attempt": attempt,
            "next_source": fallback,
            "retry_at": retry.isoformat() if retry < cutoff else None,
            **details,
        }
        if followup := source_reviews.get((code, field)):
            task["source_review"] = followup
            if followup.get("next_source"):
                task["next_source"] = followup["next_source"]
        tasks.append(task)

    def add_requirements(code, requirements, ready):
        for requirement in requirements:
            key = requirement.get("id", requirement["item"])
            if key in ready:
                continue
            use = requirement["use"]
            section = section_for(use)
            date_key = requirement.get("evidence_date", "base_nav_date")
            dates = {"评分": month_end, "买卖建议": day, "预测结果核验": None}
            required_day = requirement.get("required_date", dates.get(section, window[date_key]))
            add(code, key, None, requirement["source"], item=requirement["item"], use=use,
                required_date=required_day, data_window=requirement.get("data_window") or requirement["item"],
                verification=requirement.get("verification") or "核对基金或证券身份、所需时点、数据口径及原件",
                collection=requirement.get("collection", "按已登记取数入口或人工取证"),
                affects=requirement.get("horizons", HORIZONS if section == "四周期预测" else []))

    for code in [fund_code] if fund_code else state["scope"]:
        saved = state["funds"].get(code, {})
        item = saved if saved.get("materials_checklist_day") == day else {}
        profile = current_profiles.get(code, {})
        requirements = checklist_requirements(code, profile)
        ready, observed_sources = set(), {}
        notices = disclosure_directory(code)
        report_source = "；".join(filter(None, (profile.get("report_source"), notices)))
        first = len(tasks)
        if ratings.get("checklist_day") == day and code in ratings.get("checklist_month_covered", []):
            rated = {r["key"] for r in by_rating.get(code, {}).get("dimensions", [])
                     if r.get("score") is not None and r.get("sources")}
            ready.update(r["id"] for r in requirements if r.get("rating_dimension") in rated)
        decision = advice.get(code, {})
        if decision.get("read_back") and (decision.get("checked_at") or "").startswith(day):
            ready.update(decision.get("checklist_verified", []))
            if decision.get("checklist_current_predictions"):
                ready.add("advice_predictions")
        reviewed = saved.get("requirement_checks", {})
        if (reviewed.get("day") == day
                and reviewed.get("base_nav_date") == window["base_nav_date"]
                and reviewed.get("target_nav_date") == window["target_nav_date"]):
            for requirement in requirements:
                if requirement.get("check_key") in reviewed.get("verified", []):
                    ready.add(requirement.get("id", requirement["item"]))
        if item.get("calendar_checklist_day") == day:
            ready.add("VALUATION_CALENDAR")
        nav = saved.get("base_nav_refresh", {}) if saved.get("nav_checklist_day") == day else {}

        if nav.get("status") in {"AVAILABLE", "BASE_NAV_PRESENT"}:
            ready.add("NAV")
        else:
            add(code, "NAV", nav, nav_sources(code))
        path = item.get("materials_path")
        if not path or not source_exists(Path(path)):
            add(code, "MATERIALS", item.get("failure"), report_source)
            add(code, "NOTICES", None, notices)
            if "VALUATION_CALENDAR" not in ready:
                add(code, "VALUATION_CALENDAR", None, profile.get("contract_source") or notices)
            if profile.get("template") == "overseas":
                add(code, "FX", None, FX_SOURCE + "；" + (profile.get("fx_clause") or "核对本份额合同指定汇率"))
            for target in profile.get("target_funds", []):
                add(code, "TARGET:" + target, None, disclosure_directory(target))
            for symbol, source in profile.get("market_references", {}).items():
                add(code, "CLOSE:" + symbol, None, source)
            add_requirements(code, requirements, ready)
            save_fund_checklist(folder, code, profile, window, day, tasks[first:], ready)
            continue
        material = read(Path(path))
        if material["fund_code"] != code:
            raise ValueError("DAILY_CHECKLIST_MATERIAL_FUND_MISMATCH")
        docs = {doc["id"]: doc for doc in material["documents"]}
        for requirement in requirements:
            prefix = requirement.get("base_fact_prefix")
            required_day = window[requirement.get("evidence_date", "base_nav_date")]
            oldest = (date.fromisoformat(required_day)
                      - timedelta(days=requirement.get("max_source_age_days", 0))).isoformat()
            if prefix and any(
                key.startswith(prefix) and any(
                    ref in docs and docs[ref].get("raw_sha256")
                    and oldest <= (docs[ref].get("published_date") or "") <= required_day
                    for ref in fact.get("source_refs", [])
                )
                for key, fact in material.get("facts", {}).items()
            ):
                ready.add(requirement.get("id", requirement["item"]))
        latest = item.get("latest_report") or {}
        manifest = material.get("source_manifest", {})
        report = manifest.get("latest_partial_report", {}).get("report") or material.get("report", {})
        if (latest.get("id") and latest["id"] == report.get("id")) or item.get("report_checklist_day") == day:
            ready.add("MATERIALS")
        else:
            add(code, "MATERIALS", None, report_source)
        notices = manifest.get("current_notice_check", {})
        if (notices.get("status") == "SUCCEEDED"
                and notices.get("coverage", {}).get("end", "") >= day
                and not notices.get("publication_time_required")):
            ready.add("NOTICES")
        else:
            add(code, "NOTICES", notices, disclosure_directory(code))
        if profile.get("template") == "domestic" and profile.get("valuation_policy"):
            ready.add("VALUATION_CALENDAR")
        for target, reference in manifest.get("target_partial_reports", {}).items():
            if reference.get("report", {}).get("id"):
                ready.add("TARGET:" + target)
        for target in profile.get("target_funds", []):
            if "TARGET:" + target not in ready:
                add(code, "TARGET:" + target, None, disclosure_directory(target))
        quotes = {symbol: {} for symbol in profile.get("market_references", {})}
        quotes.update({c["code"]: c.get("quote") or {} for c in material.get("companies", [])})
        quotes.update(material.get("market", {}))
        attempts = {
            a.get("security") or a.get("code"): a
            for key in ("market_attempts", "hk_attempts", "overseas_attempts")
            for a in item.get(key, []) or [] if isinstance(a, dict)
        }
        retired = {symbol for symbol, retirement in profile.get("retired_market_references", {}).items()
                   if retirement["effective_date"] <= window["base_nav_date"]}
        ready.update("CLOSE:" + symbol for symbol in retired)
        for symbol, quote in quotes.items():
            if symbol in retired:
                continue
            source = docs.get(quote.get("source_ref"), {})
            if source.get("url"):
                observed_sources[symbol] = source["url"]
            if quote.get("date") != window["base_nav_date"] or quote.get("change_pct") is None:
                add(code, "CLOSE:" + symbol, {"last_source": source.get("url"),
                                        "last_obtained_at": source.get("available_at"),
                                        "last_date": quote.get("date"),
                                        "latest_attempt": attempts.get(symbol)},
                    profile.get("market_references", {}).get(symbol) or source.get("url"))
            elif source.get("url"):
                ready.add("CLOSE:" + symbol)
        fx = next((row for row in item.get("overseas_attempts", [])
                   if isinstance(row, dict) and row.get("kind") == "OFFICIAL_FX"), None)
        if profile.get("valuation_policy") == "QDII_CNY_VN_CONTRACT_SESSION_V1":
            from app.services.vietnam_valuation_contract import assess_vietnam_valuation

            status = assess_vietnam_valuation({"fund_name": profile["name"]}, material, docs, window, as_of=instant)
            if not status["supported"]:
                add(code, "VALUATION_CALENDAR", status["gaps"], profile.get("contract_source"))
            else:
                ready.add("VALUATION_CALENDAR")
            if not status["fx_ready"]:
                add(code, "FX:VND/USD", status["gaps"], "估值日伦敦 16:00 彭博折算率，或合同指定的托管人来源")
            else:
                ready.add("FX")
        elif fx:
            if fx.get("status") == "SUCCEEDED":
                ready.add("FX")
            else:
                add(code, "FX", fx, fx.get("planned_url") or profile.get("fx_clause") or "核对本份额合同指定汇率")
        elif profile.get("template") == "overseas":
            add(code, "FX", None, FX_SOURCE + "；" + (profile.get("fx_clause") or "核对本份额合同指定汇率"))
        if "VALUATION_CALENDAR" not in ready and profile.get("valuation_policy") != "QDII_CNY_VN_CONTRACT_SESSION_V1":
            add(code, "VALUATION_CALENDAR", None, profile.get("contract_source") or disclosure_directory(code))
        add_requirements(code, requirements, ready)
        save_fund_checklist(folder, code, profile, window, day, tasks[first:], ready, observed_sources)
    if aggregate:
        versioned_save(
            directory / "supplement-tasks.json",
            {"window": window, "checked_at": instant.isoformat(), "tasks": tasks},
        )
    return tasks
