"""只核验原估值日历排除目标日的阻断证据，不放宽预测准入或推算下一目标。"""

import json
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlparse

from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import now, read


def _base_nav_available(code, base, at):
    """只读既有有效来源，基准净值尚未实际取得时不能只用休市说明收口。"""
    from app.db.session import get_engine
    from app.repositories import direction_1d as repo

    with get_engine().connect() as connection:
        source = repo.source(connection)
        day = date.fromisoformat(base)
        rows = repo.navs(connection, code, source["source_id"], day, day)
    return any(r["updated_at"] <= at and (r.get("ann_date") is None or r["ann_date"] <= at.date()) for r in rows)


def verified_non_trading_target(proof: dict, window: dict) -> bool:
    """人民币日本股票QDII的有限阻断：本期资料齐备，但官方明确目标日休市。

    输入仍被原预测适配器拒绝，绝不生成预测。回执须绑定本份额、本窗口和
    实际资料版本；公司资料、基准日收市及本基金估值条款缺失均继续处理。
    其他市场或混合交易日历不套用此分支，不把普通解析失败改名为外部阻断。
    """
    try:
        from app.services.fund_comprehensive_materials import checked_documents

        material = read(Path(proof["calendar_materials_path"]))
        code, base, target = proof["fund_code"], window["base_nav_date"], window["target_nav_date"]
        if material["fund_code"] != code or digest(material) != proof["calendar_materials_hash"]:
            return False
        evidence = material["calendar_evidence"]
        if (
            evidence.get("policy") != "QDII_CNY_VERIFIED_SESSION_V1"
            or evidence.get("base_date") != base
            or evidence.get("target_date") != target
            or set(evidence.get("calendar_refs", {})) != {"JP"}
        ):
            return False
        documents = {d["id"]: d for d in checked_documents(now(), code, material)}
        calendar = documents[evidence["calendar_refs"]["JP"]]
        parsed = json.loads(calendar["body"])
        day = date.fromisoformat(target)
        source_day = day.strftime("%b.") + f" {day.day} (" + day.strftime("%a.") + ")"
        if (
            calendar.get("kind") != "EXCHANGE_CALENDAR"
            or urlparse(calendar["url"]).hostname != "www.jpx.co.jp"
            or urlparse(calendar["url"]).path != "/english/corporate/about-jpx/calendar/index.html"
            or parsed["year"] != day.year
            or len(set(parsed["closed_dates"])) < 20
            or target not in parsed["closed_dates"]
            or base in parsed["closed_dates"]
            or source_day not in parsed["source_text"]
        ):
            return False
        prospectus, report = documents[evidence["prospectus_ref"]], documents[evidence["report_ref"]]
        clauses = evidence["valuation_quotes"]
        if (
            prospectus.get("kind") != "PROSPECTUS"
            or report.get("kind") != "REPORT"
            or code not in prospectus.get("codes", [])
            or code not in report.get("codes", [])
            or set(clauses) != {"day", "close", "fx"}
            or any(not q or q not in prospectus["body"] for q in clauses.values())
            or "估值日" not in clauses["day"]
            or "交易场所" not in clauses["day"]
            or "收盘价" not in clauses["close"]
            or "中间价" not in clauses["fx"]
        ):
            return False
        companies = material["companies"]
        if not companies or any(not c["code"].endswith(".JP") or not c.get("financials") for c in companies):
            return False
        for quote in [c.get("quote", {}) for c in companies] + list(material["market"].values()):
            source = documents[quote["source_ref"]]
            if (
                quote.get("date") != base
                or quote.get("valuation_market") != "JP"
                or quote.get("currency") != "JPY"
                or quote.get("session") != "REGULAR_CLOSE"
                or quote.get("change_pct") is None
                or float(quote["close"]) <= 0
                or float(quote["previous_close"]) <= 0
                or source.get("kind") != "MARKET_DATA"
                or datetime.fromisoformat(source["available_at"]) < datetime.fromisoformat(quote["closed_at"])
            ):
                return False
        return (
            proof.get("base_date") == base
            and proof.get("target_date") == target
            and _base_nav_available(code, base, now())
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False
