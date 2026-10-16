"""本期补充日历的关键边界：港股通不能误放开QDII或非人民币份额。"""

from copy import deepcopy

import pytest
from app.services.comprehensive_calendar import verified_domestic_window
from app.services.nav_repair import domestic_calendar_supported


def sample():
    profile = {"fund_name": "境内科技混合C", "fund_type": "MIXED", "benchmark": "中证港股通指数(港元)"}
    holidays = "\n".join(f"{day} January 2026" for day in range(1, 16))
    documents = {
        "report": {"kind": "REPORT", "codes": ["007356"], "body": "本基金可投资港股通股票。"},
        "calendar": {
            "kind": "EXCHANGE_CALENDAR",
            "url": "https://www.hkex.com.hk/2026.pdf",
            "body": "Holiday Schedule for Year 2026\nSaturdays, Sundays\n" + holidays,
        },
    }
    window = {"base_nav_date": "2026-10-14", "target_nav_date": "2026-10-15"}
    material = {
        "fund_code": "007356",
        "companies": [{"code": "00700.HK", "quote": {"date": "2026-10-14", "change_pct": 0.5}}],
        "calendar_evidence": {
            "policy": "CN_NAV_HK_CONNECT_WINDOW_V1",
            "base_date": "2026-10-14",
            "target_date": "2026-10-15",
            "report_ref": "report",
            "calendar_ref": "calendar",
        },
    }
    return profile, material, documents, window


def test_hkd_benchmark_is_not_hkd_share_class():
    assert verified_domestic_window(*sample())


@pytest.mark.parametrize(
    "problem", ["qdii", "hkd_share", "wrong_fund", "wrong_window", "stale_quote", "wrong_domain", "holiday"]
)
def test_incomplete_or_foreign_scope_is_rejected(problem):
    profile, material, documents, window = deepcopy(sample())
    if problem == "qdii":
        profile["fund_name"] += "(QDII)"
    elif problem == "hkd_share":
        profile["fund_name"] += "-港元"
    elif problem == "wrong_fund":
        documents["report"]["codes"] = ["000001"]
    elif problem == "wrong_window":
        material["calendar_evidence"]["target_date"] = "2026-10-16"
    elif problem == "stale_quote":
        material["companies"][0]["quote"]["date"] = "2026-10-13"
    elif problem == "wrong_domain":
        documents["calendar"]["url"] = "https://example.com/2026.pdf"
    else:
        documents["calendar"]["body"] += "\n15 October 2026"
    assert not verified_domestic_window(profile, material, documents, window)


def test_chinese_ashare_sp_index_remains_domestic_but_us_sp_index_does_not():
    assert domestic_calendar_supported({"fund_name": "南方标普中国A股大盘红利低波50ETF联接-C", "fund_type": "STOCK"})
    assert not domestic_calendar_supported({"fund_name": "标普500ETF联接(QDII)", "fund_type": "STOCK"})


def test_domestic_gold_requires_its_own_target_and_quote():
    profile = {"fund_name": "前海开源黄金ETF联接-C", "fund_type": "OTHER"}
    window = {"base_nav_date": "2026-10-14", "target_nav_date": "2026-10-15"}
    documents = {"report": {"kind": "REPORT", "codes": ["021740"], "body": "目标基金代码159812"}}
    material = {
        "fund_code": "021740",
        "exposure_chain": ["021740", "159812"],
        "market": {"159812.SZ": {"date": "2026-10-14", "change_pct": -1.16}},
        "calendar_evidence": {
            "policy": "CN_GOLD_ETF_NAV_WINDOW_V1",
            "base_date": "2026-10-14",
            "target_date": "2026-10-15",
            "report_ref": "report",
            "quote_currency": "CNY",
        },
    }
    assert verified_domestic_window(profile, material, documents, window)
    material["market"] = {"000300.SH": {"date": "2026-10-14", "change_pct": -1.0}}
    assert not verified_domestic_window(profile, material, documents, window)


def test_unannounced_nav_is_excluded_from_real_freeze(monkeypatch, tmp_path):
    """来源提前给了估值日数值，也不能当作截止前已公告的净值；原行保持不变。"""
    from contextlib import nullcontext
    from datetime import date, datetime
    from types import SimpleNamespace

    from app.services import fund_comprehensive_materials as service
    from app.services.direction_1d_protocol import digest
    from app.services.fund_exposure_common import save

    now = datetime.fromisoformat("2026-10-14T17:00:00+08:00")
    document = {
        "id": "source",
        "kind": "REPORT",
        "title": "公开报告",
        "url": "https://example.com/report",
        "body": "真实资料原文。",
        "body_hash": digest("真实资料原文。"),
        "body_scope": "FULL_TEXT",
        "published_date": "2026-08-31",
        "available_at": "2026-10-14T16:00:00+08:00",
    }
    material = {
        "fund_code": "006730",
        "report": {"endDate": "2026-06-30", "publishedDate": "2026-08-31"},
        "documents": [document],
        "facts": {"context": {"category": "MARKET", "source_refs": ["source"], "quotes": [document["body"]]}},
        "companies": [],
        "market": {},
        "source_manifest": {},
        "inventory": [{} for _ in range(16)],
        "comprehensive_coverage": {},
    }
    path = tmp_path / "materials.json"
    save(path, material)
    rows = [
        {"nav_date": date(2026, 10, 13), "ann_date": date(2026, 10, 14), "updated_at": now, "unit_nav": "1.0"},
        {"nav_date": date(2026, 10, 14), "ann_date": date(2026, 10, 15), "updated_at": now, "unit_nav": "1.1"},
    ]
    before = deepcopy(rows)
    monkeypatch.setattr(service, "get_engine", lambda: SimpleNamespace(begin=lambda: nullcontext(None)))
    monkeypatch.setattr(service.repo, "source", lambda c: {"source_id": "test", "retention_days": 365})
    monkeypatch.setattr(
        service.repo,
        "profiles",
        lambda c, codes: [
            {"fund_name": "境内指数增强", "fund_type": "STOCK", "status": "ACTIVE", "fund_master_id": "test"}
        ],
    )
    monkeypatch.setattr(service.repo, "navs", lambda *args: rows)
    observed = []

    def observe(connection, code, source, available, frozen_at):
        observed.extend(available)
        return [{"nav_date": str(row["nav_date"]), "unit_nav": row["unit_nav"]} for row in available]

    monkeypatch.setattr(service.repo, "observe", observe)
    result = service.prepare(now, "006730", path)
    assert result["latest_nav_date"] == "2026-10-13"
    assert result["time_context"]["nav_status"] == "STALE"
    assert [row["nav_date"] for row in observed] == [date(2026, 10, 13)]
    assert rows == before
