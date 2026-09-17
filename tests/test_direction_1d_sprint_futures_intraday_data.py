"""日内特征只使用同一日原始OHLC，实时只复用验证过的父源。"""

import json
from datetime import date

import pytest
from app.services import direction_1d_sprint_futures_intraday_data as s


def test_same_contract_intraday_features_and_flat_range():
    q = {"open": 100, "close": 102, "high": 104, "low": 96}
    assert s.features(q)["intraday_return_pct"] == pytest.approx(2)
    assert s.features(q)["close_location"] == 0.5
    assert s.features({k: 100 for k in q}) == {"intraday_return_pct": 0, "close_location": 0, "zero_range": True}


@pytest.mark.parametrize(
    "key,value", [("open", 0), ("close", True), ("high", 99), ("low", 103), ("close", float("nan"))]
)
def test_invalid_ohlc_rejected(key, value):
    q = {"open": 100, "close": 102, "high": 104, "low": 96}
    q[key] = value
    with pytest.raises(ValueError):
        s.features(q)


def test_exact_t_only_and_original_row_preserved(monkeypatch):
    monkeypatch.setattr(s.base, "calendar", lambda: ([date(2026, 9, 15), date(2026, 9, 16), date(2026, 9, 17)], "c"))
    raw = {"available": True, "x": 2}
    values = {"2026-09-16": {"intraday_return_pct": 2, "close_location": 0.5, "zero_range": False}}
    enriched = s.extend(raw, "2026-09-16", "2026-09-17", values)
    assert enriched["futures_intraday"]["date"] == "2026-09-16"
    assert s.original_row({"y": 1, "market": enriched}) == {"y": 1, "market": raw}
    assert not s.extend(raw, "2026-09-15", "2026-09-16", values)["futures_intraday"]["available"]


@pytest.fixture
def actual_parent(monkeypatch, tmp_path):
    parent = {
        "at": "2026-09-17T08:14:30+08:00",
        "available": True,
        "snapshot": {"points": {"2026-09-16": {"futures_close": 102}}},
    }
    monkeypatch.setattr(s.current, "load_live", lambda t, u, ph: parent)
    monkeypatch.setattr(s.current, "live_root", lambda: tmp_path)
    p = tmp_path / "2026-09-17/raw/fut_daily.json"
    p.parent.mkdir(parents=True)
    p.write_text(
        json.dumps(
            {
                "code": 0,
                "data": {
                    "fields": s.current.parser.FIELDS["fut_daily"],
                    "items": [["IF.CFX", "20260916", 100, 100, 104, 96, 102, 101, 100, 200]],
                },
            }
        ),
        encoding="utf-8",
    )
    return parent, p


def test_reuses_exact_validated_parent_source_without_http(actual_parent):
    parent, _ = actual_parent
    value = s.live("2026-09-16", "2026-09-17", "parent-plan")
    assert value["parent_source_hash"] == s.base.digest(parent)
    assert value["at"] == parent["at"] and value["new_source_requests"] == 0
    assert value["snapshot"]["points"]["2026-09-16"]["close_location"] == 0.5


def test_unavailable_parent_never_fills_from_history(actual_parent):
    parent, path = actual_parent
    parent["available"] = False
    path.unlink()
    value = s.live("2026-09-16", "2026-09-17", "parent-plan")
    assert not value["available"] and value["snapshot"]["points"] == {}


def test_raw_quote_must_match_parent_close(actual_parent):
    parent, _ = actual_parent
    parent["snapshot"]["points"]["2026-09-16"]["futures_close"] = 101
    with pytest.raises(ValueError, match="LIVE_CLOSE_CHANGED"):
        s.live("2026-09-16", "2026-09-17", "parent-plan")
