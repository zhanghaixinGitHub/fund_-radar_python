"""官方休市只关闭真实日历阻断，不能放行预测或掩盖资料、身份和日期问题。"""

from copy import deepcopy
from datetime import datetime

import pytest
from app.services import fund_prediction_calendar_block as block
from app.services.direction_1d_protocol import ZONE, digest
from app.services.fund_exposure_common import save

WINDOW = {"base_nav_date": "2026-11-02", "target_nav_date": "2026-11-03"}


def fixture_material():
    import json

    day, close, fx = "本基金估值日为相关证券交易场所的交易日。", "股票按收盘价估值。", "人民币采用中间价。"
    data = {
        "year": 2026,
        "closed_dates": [f"2026-01-{n:02}" for n in range(1, 21)] + ["2026-11-03"],
        "source_text": "Nov. 3 (Tue.) Culture Day",
    }
    bodies = {
        "calendar": json.dumps(data),
        "prospectus": day + close + fx,
        "report": "019449 本基金原报告。",
        "quote": "实际十一月二日东京收市报价。",
    }
    documents = [
        {
            "id": key,
            "kind": {
                "calendar": "EXCHANGE_CALENDAR",
                "prospectus": "PROSPECTUS",
                "report": "REPORT",
                "quote": "MARKET_DATA",
            }[key],
            "url": "https://www.jpx.co.jp/english/corporate/about-jpx/calendar/index.html"
            if key == "calendar"
            else "https://example.org/" + key,
            "body": body,
            "body_hash": digest(body),
            "codes": ["019449"],
            "published_date": "2026-11-02",
            "available_at": "2026-11-02T17:00:00+08:00",
        }
        for key, body in bodies.items()
    ]
    quote = {
        "date": "2026-11-02",
        "valuation_market": "JP",
        "currency": "JPY",
        "session": "REGULAR_CLOSE",
        "change_pct": 1,
        "close": 101,
        "previous_close": 100,
        "source_ref": "quote",
        "closed_at": "2026-11-02T15:30:00+09:00",
    }
    return {
        "fund_code": "019449",
        "documents": documents,
        "companies": [{"code": "8035.JP", "financials": [{"sourceRef": "report"}], "quote": quote}],
        "market": {"TOPIX": deepcopy(quote)},
        "calendar_evidence": {
            "policy": "QDII_CNY_VERIFIED_SESSION_V1",
            **{"base_date": WINDOW["base_nav_date"], "target_date": WINDOW["target_nav_date"]},
            "calendar_refs": {"JP": "calendar"},
            "prospectus_ref": "prospectus",
            "report_ref": "report",
            "valuation_quotes": {"day": day, "close": close, "fx": fx},
        },
    }


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "fund",
        "target",
        "hash",
        "official_text",
        "host",
        "financials",
        "stale_quote",
        "intraday",
        "mixed_market",
        "missing_clause",
        "missing_nav",
    ],
)
def test_requires_real_same_window_calendar_and_other_required_materials(tmp_path, monkeypatch, problem):
    monkeypatch.setattr(block, "now", lambda: datetime(2026, 11, 3, 1, tzinfo=ZONE))
    monkeypatch.setattr(block, "_base_nav_available", lambda *args: problem != "missing_nav")
    material = fixture_material()
    if problem == "fund":
        material["fund_code"] = "000001"
    if problem == "target":
        material["calendar_evidence"]["target_date"] = "2026-11-04"
    if problem in {"official_text", "host"}:
        doc = material["documents"][0]
        if problem == "official_text":
            doc["body"] = doc["body"].replace("Nov. 3", "Nov. 4")
        else:
            doc["url"] = "https://other.example/calendar"
        doc["body_hash"] = digest(doc["body"])
    if problem == "financials":
        material["companies"][0]["financials"] = []
    if problem == "stale_quote":
        material["companies"][0]["quote"]["date"] = "2026-10-30"
    if problem == "intraday":
        material["companies"][0]["quote"]["session"] = "OPEN"
    if problem == "mixed_market":
        material["companies"][0]["code"] = "0700.HK"
    if problem == "missing_clause":
        material["calendar_evidence"]["valuation_quotes"]["day"] = "未核验"
    path = tmp_path / "materials.json"
    save(path, material)
    proof = {
        "fund_code": "019449",
        "base_date": WINDOW["base_nav_date"],
        "target_date": WINDOW["target_nav_date"],
        "calendar_materials_path": str(path),
        "calendar_materials_hash": "bad" if problem == "hash" else digest(material),
    }
    assert block.verified_non_trading_target(proof, WINDOW) is (problem is None)
