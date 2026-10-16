"""任意基金的综合资料适配；只消费已留原文的本机资料包，不套用其他基金资料。"""

from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlparse

from app.core.config import get_settings
from app.db.session import get_engine
from app.repositories import direction_1d as repo
from app.services.comprehensive_prediction_policy import policy
from app.services.direction_1d_protocol import ZONE, digest, input_days, window
from app.services.fund_exposure_common import read
from app.services.fund_information_contracts import time_context


def prepare(now, code: str, materials_path=None) -> dict:
    """冻结该基金的公开资料与真实数据库时钟；缺少投向/行情/公告时不生成方向。

    资料包使用既有 hash/payload 格式。正文、发布时间、关联投向及行情由本机采集
    和聊天审阅整理，导入不接受浏览器上传；本函数复核身份、字节及日期边界。
    """
    now = now.astimezone(ZONE)
    path = Path(materials_path or Path(get_settings().fund_material_directory) / "comprehensive" / f"{code}.json")
    if not path.is_file():
        raise ValueError("COMPREHENSIVE_MATERIALS_REQUIRED")
    material = read(path)
    if material.get("fund_code") != code:
        raise ValueError("COMPREHENSIVE_FUND_MISMATCH")
    w = window(now)
    base = date.fromisoformat(w["base_nav_date"])
    report = material["report"]
    if (
        date.fromisoformat(report["endDate"]) > base
        or report["publishedDate"] > str(now.date())
        or not 0 <= (base - date.fromisoformat(report["endDate"])).days <= 210
    ):
        raise ValueError("COMPREHENSIVE_EXPOSURE_STALE")
    documents = deepcopy(material["documents"])
    by_id = {d["id"]: d for d in documents}
    if len(by_id) != len(documents):
        raise ValueError("COMPREHENSIVE_DUPLICATE_SOURCE")
    for document in documents:
        uri = urlparse(document["url"])
        available = datetime.fromisoformat(document["available_at"])
        published_at = datetime.fromisoformat(document["published_at"]) if document.get("published_at") else None
        if (
            uri.scheme not in {"https", "http"}
            or not uri.hostname
            or uri.username
            or uri.password
            or available.tzinfo is None
            or available > now
            or document["published_date"] > str(now.date())
            or available.astimezone(ZONE).date().isoformat() < document["published_date"]
            or (
                published_at is not None
                and (published_at.tzinfo is None or published_at > available or published_at > now)
            )
            or (published_at is None and document["published_date"] == str(now.date()))
            or digest(document["body"]) != document["body_hash"]
        ):
            raise ValueError("COMPREHENSIVE_SOURCE_INVALID")
        # 同日资料只有日期不能证明截止前已公开；保留缺口，不能默认按零点处理。
        document.setdefault("date", document["published_date"])
        document.setdefault("source_hash", document["body_hash"])
        document.setdefault("body_truncated", document.get("body_scope") != "FULL_TEXT")
    facts = deepcopy(material["facts"])
    for fact in facts.values():
        categories = {
            "MARKET": "市场行情",
            "DISCLOSURES": "公告",
            "FINANCIALS": "财务",
            "NEWS": "新闻",
            "POLICY": "政策",
        }
        fact["category"] = categories.get(fact.get("category"), fact.get("category"))
        refs, quotes = fact.get("source_refs", []), fact.get("quotes", [])
        if not refs or not set(refs) <= set(by_id) or not quotes:
            raise ValueError("COMPREHENSIVE_FACT_SOURCE_REQUIRED")
        if any(not any(quote in by_id[ref]["body"] for ref in refs) for quote in quotes):
            raise ValueError("COMPREHENSIVE_FACT_QUOTE_INVALID")
        # 事实正文只能由已经匹配的原文组成，不能在导入包里另写未经来源支持的数字或摘要。
        fact["text"] = "\n".join(quotes)
        sources = [
            {"title": by_id[ref]["title"], "url": by_id[ref]["url"], "publishedDate": by_id[ref]["published_date"]}
            for ref in refs
        ]
        fact.update(source=sources[0], sources=sources)
    with get_engine().begin() as connection:
        source = repo.source(connection)
        profiles = repo.profiles(connection, [code])
        if not profiles or profiles[0]["status"] != "ACTIVE":
            raise ValueError("SOURCE_UNAVAILABLE")
        profile = profiles[0]
        from app.services.comprehensive_calendar import verified_domestic_window
        from app.services.qdii_valuation_contract import verified_qdii_window

        if not verified_domestic_window(profile, material, by_id, w) and not verified_qdii_window(
            profile, material, by_id, w
        ):
            raise ValueError("COMPREHENSIVE_CALENDAR_POLICY_REQUIRED")
        if "联接" in profile["fund_name"] and len(material.get("exposure_chain", [])) < 2:
            raise ValueError("COMPREHENSIVE_ETF_LOOKTHROUGH_REQUIRED")
        rows = repo.navs(connection, code, source["source_id"], input_days(base)[0], base)
        # 来源可能提前返回估值日数据，但公告日期仍在未来；这类记录尚不可用。
        # 保留库内原记录与首次响应，只在本次冻结输入中排除，不能倒填公告时间。
        rows = [
            row
            for row in rows
            if row["updated_at"] <= now
            and (row.get("ann_date") is None or row["ann_date"] <= now.date())
        ]
        observed = repo.observe(connection, code, source, rows, now) if rows else []
    data = {
        "fund_code": code,
        "fund_name": profile["fund_name"],
        "product_family_id": str(profile["fund_master_id"]),
        "source_id": str(source["source_id"]),
        "retention_days": source["retention_days"],
        "as_of": now.isoformat(),
        "window": w,
        "report": deepcopy(report),
        "companies": deepcopy(material["companies"]),
        "market": deepcopy(material["market"]),
        "documents": documents,
        "facts": facts,
        "nav": observed,
        "latest_nav_date": str(rows[-1]["nav_date"]) if rows else None,
        "source_manifest": material["source_manifest"],
        "inventory": deepcopy(material["inventory"]),
        "comprehensive_coverage": deepcopy(material["comprehensive_coverage"]),
        "analysis_policy_version": policy()["version"],
        "materials_path": str(path.resolve()),
        "materials_hash": digest(material),
        "exposure_chain": material.get("exposure_chain", [code]),
        "stock_market_applicable": material.get("stock_market_applicable", True),
        "calendar_evidence": deepcopy(material.get("calendar_evidence")),
    }
    if len(data["inventory"]) != 16:
        raise ValueError("COMPREHENSIVE_INVENTORY_INVALID")
    data["time_context"] = time_context(data)
    return data
