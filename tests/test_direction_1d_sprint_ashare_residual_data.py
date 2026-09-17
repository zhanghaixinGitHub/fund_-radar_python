"""现金边界、中国长假、异常价格与真实原始字节复用的业务测试。"""

import hashlib

import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_ashare_residual_data as d

from test_direction_1d_sprint_ashare_data import points, quote


def test_opening_gap_and_known_china_move_are_included():
    p = points() | {"cash": {}}
    assert d.features("2026-09-14", "2026-09-15", p, 3.0) == pytest.approx([7.0, 1])
    # 同一个开到收变化，开盘缺口不同应改变残差。
    p["ASHR"]["2026-09-11"] = quote(110, 110)
    assert d.features("2026-09-14", "2026-09-15", p, 3.0) == pytest.approx([-3.0, 1])


def test_cash_ex_date_interval_excludes_anchor_and_future():
    p = points() | {"cash": {"2026-09-11": 99, "2026-09-14": 2, "2026-09-15": 99}}
    assert d.features("2026-09-14", "2026-09-15", p, 3.0) == pytest.approx([9.0, 1])
    p["ASHR"]["2026-09-14"] = quote(98, 98)
    assert d.features("2026-09-14", "2026-09-15", p, 0.0) == pytest.approx([0.0, 1])


def test_long_china_holiday_requires_all_us_sessions_and_original_anchor():
    days = d.interval("2025-01-27", "2025-02-05")
    assert days[0] == "2025-01-24" and days[-1] == "2025-02-04" and len(days) > 2
    p = {"ASHR": {day: quote(100, 100) for day in days}, "cash": {"2025-01-28": 1}}
    p["ASHR"][days[-1]] = quote(102, 102)
    assert d.features("2025-01-27", "2025-02-05", p, 1) == pytest.approx([2, 1])
    del p["ASHR"][days[1]]
    with pytest.raises(ValueError, match="INTERVAL_MISSING"):
        d.features("2025-01-27", "2025-02-05", p, 1)


def test_no_new_us_session_keeps_parent():
    p = points("2025-01-09", "2025-01-10") | {"cash": {}}
    assert d.features("2025-01-09", "2025-01-10", p, 1) == [0, 0]


def test_gross_raw_price_jump_routes_to_parent_and_does_not_change_price():
    p = points() | {"cash": {}}
    p["ASHR"]["2026-09-14"] = quote(50, 50)
    assert d.features("2026-09-14", "2026-09-15", p, 0) == [0, 0]
    assert p["ASHR"]["2026-09-14"]["close"] == 50


def test_capture_reuses_parent_and_full_raw_input(monkeypatch, tmp_path):
    monkeypatch.setattr(b, "ROOT", tmp_path)
    raw = b"existing history bytes"
    meta = {"body_sha256": hashlib.sha256(raw).hexdigest()}
    folder = d.source.root() / "2026-09-15/ASHR/raw"
    folder.mkdir(parents=True)
    (folder / "0700.bin").write_bytes(raw)
    b.save(folder / "0700.json", meta)
    parent = {
        "base": "2026-09-14",
        "target": "2026-09-15",
        "raw_refs": {"ASHR": {"slot": "0700", "hash": b.digest(meta)}},
    }
    calls = []
    monkeypatch.setattr(d.source, "capture", lambda at: calls.append(at) or parent)
    monkeypatch.setattr(d.source, "load", lambda target: parent)
    monkeypatch.setattr(d.source, "parse", lambda *args: points()["ASHR"])
    monkeypatch.setattr(d.source, "fetch_etf", lambda *args: pytest.fail("new GET prohibited"))
    monkeypatch.setattr(d, "cash_snapshot", lambda: {"cash": {}})
    value = d.capture("clock")
    assert calls == ["clock"] and value["rows"] == points() | {"cash": {}}
    (folder / "0700.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="LIVE_RAW_CHANGED"):
        d.load("2026-09-15")


def test_cash_edit_cannot_be_hidden_by_rehashing_json(monkeypatch, tmp_path):
    import json

    monkeypatch.setattr(b, "ROOT", tmp_path)
    table = {
        "columns": [{"value": k} for k in ["Ex-Date", "Record date", "Pay date", "US$ / Share"]],
        "values": [{"column_0": {"sortValue": "2025-12-19T00:00:00"}, "column_3": {"sortValue": 0.75}}],
    }
    raw = json.dumps(
        {
            "pdpResult": {
                "pageSections": {
                    "keyFacts": {"accordionItems": [{"id": "fundinformation-distributions", "table": table}]}
                }
            }
        }
    ).encode()
    (tmp_path / "primary.bin").write_bytes(raw)
    snapshot = {
        "source_file": "primary.bin",
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "cash": {"2025-12-19": 0.75},
    }
    path = tmp_path / "ashr-cash-residual-source-v1/cash-distributions.json"
    b.save(path, snapshot)
    assert d.cash_snapshot() == snapshot
    snapshot["cash"]["2025-12-19"] = 4
    b.save(path, snapshot, replace=True)
    with pytest.raises(ValueError, match="CASH_PARSED_CHANGED"):
        d.cash_snapshot()
