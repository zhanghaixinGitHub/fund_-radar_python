import json
from types import SimpleNamespace

import pytest
from app.services import fund_rating_bosera_qdii_performance as component


def test_official_history_uses_new_window_requests_and_rejects_future_rows(monkeypatch):
    source = SimpleNamespace(window_start="2023-10-31", as_of="2026-10-30")
    responses = {}
    for year in range(2023, 2027):
        responses[f"unit-{year}"] = (
            {"params": {"fundCode": "018853", "startDate": "20231001" if year == 2023 else f"{year}0101",
                        "endDate": "20261030" if year == 2026 else f"{year}1231"}},
            json.dumps({"retCode": "0", "data": {"title": "unit", "resultList": [{f"{year}1030": "1.2000"}]}}),
        )
    monkeypatch.setattr(component, "checked_receipt", lambda s, key: responses[key])
    values, receipts = component.series_from_bosera(source, "unit", "unit")
    assert len(receipts) == 4 and values["2026-10-30"] > 0
    receipt, _ = responses["unit-2026"]
    responses["unit-2026"] = receipt, json.dumps(
        {"retCode": "0", "data": {"title": "unit", "resultList": [{"20261102": "1.2000"}]}}
    )
    with pytest.raises(ValueError, match="FUTURE_NAV_VALUATION"):
        component.series_from_bosera(source, "unit", "unit")


@pytest.mark.parametrize("has_event", [False, True])
def test_unadjusted_requires_actual_closed_event_originals(monkeypatch, has_event):
    rows = [["cash"]] if has_event else []
    dividend = {"code": 0, "data": {"count": len(rows), "has_more": False, "items": rows}}
    html = ('<title>基金(018853)基金分红送配</title><table><tr><th>年份</th><th>拆分折算日</th>'
            '<th>拆分类型</th><th>拆分折算比例</th></tr><tr><td>暂无拆分信息!</td></tr></table>')
    responses = {
        "target-div": ({"params": {"ts_code": "018853.OF"}}, json.dumps(dividend)),
        "target-split": ({"url": "https://fundf10.eastmoney.com/fhsp_018853.html"}, html),
    }
    monkeypatch.setattr(component, "checked_receipt", lambda source, key: responses[key])
    if has_event:
        with pytest.raises(ValueError, match="DIVIDEND_PROOF_REQUIRED"):
            component.verify_empty_events(None, "target", "018853")
    else:
        component.verify_empty_events(None, "target", "018853")
