"""份额字段单位、异常规则、六日期时点和复用原始采集的证据边界。"""

import hashlib
from datetime import datetime
from xml.sax.saxutils import escape

import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_fxi_shares_data as d


def body(shares="1000", omit=None):
    rows = [["As Of", "NAV per Share", "Ex-Dividends", "Shares Outstanding", "Non-FV NAV"]]
    for day in (16, 15, 14, 11, 10, 9, 8, 4, 3, 2):
        if day != omit:
            value = str(shares) if day == 11 else "1100" if day == 14 else "1000"
            rows.append([f"Sep {day:02d}, 2026", "100", "0", value, "100"])
    xml = "".join(
        "<ss:Row>"
        + "".join('<ss:Cell><ss:Data ss:Type="String">' + escape(v) + "</ss:Data></ss:Cell>" for v in row)
        + "</ss:Row>"
        for row in rows
    )
    return (
        '<ss:Workbook xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">'
        '<ss:Worksheet ss:Name="Historical"><ss:Table>' + xml + "</ss:Table></ss:Worksheet></ss:Workbook>"
    ).encode()


def test_six_cash_dates_and_percentage_units_exclude_later_values():
    points = d.parse(body())
    assert d.required("2026-09-15", "2026-09-16") == [
        "2026-09-04",
        "2026-09-08",
        "2026-09-09",
        "2026-09-10",
        "2026-09-11",
        "2026-09-14",
    ]
    assert d.features("2026-09-15", "2026-09-16", points) == pytest.approx([10, 10, 0, 1])
    points["2026-09-15"]["shares"] = float("nan")
    assert d.features("2026-09-15", "2026-09-16", points) == pytest.approx([10, 10, 0, 1])
    assert d.parse(body("1,000"))["2026-09-11"]["shares"] == 1000
    with pytest.raises(ValueError, match="NOT_ADJACENT"):
        d.required("2026-09-15", "2026-09-17")


@pytest.mark.parametrize("value,reason", [("--", "EXPLICIT_MISSING"), ("0", "INVALID_ZERO")])
def test_explicit_missing_masks_without_dropping_dates(value, reason):
    points = d.parse(body(value))
    assert points["2026-09-11"]["unavailable_reason"] == reason
    assert d.features("2026-09-15", "2026-09-16", points) == [0, 0, 0, 0]


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "N/A", "1,00", " 1000", "1e3"])
def test_malformed_values_are_not_missing(value):
    with pytest.raises(ValueError, match="NUMBER_INVALID"):
        d.parse(body(value))


@pytest.mark.parametrize("value", [1500, 500, 2000])
def test_structural_jump_is_flagged_not_called_flow(value):
    assert d.features("2026-09-15", "2026-09-16", d.parse(body(value))) == [0, 0, 1, 0]


def test_missing_date_and_fabricated_mask_are_rejected():
    with pytest.raises(ValueError, match="REQUIRED_DATE_ABSENT"):
        d.features("2026-09-15", "2026-09-16", d.parse(body(omit=9)))
    points = d.parse(body())
    points["2026-09-11"]["available"] = False
    with pytest.raises(ValueError, match="REASON_CHANGED"):
        d.features("2026-09-15", "2026-09-16", points)


@pytest.mark.parametrize("fault", ["sheet", "field", "sparse", "entity", "badxml", "oversize", "order"])
def test_invalid_structure_is_rejected(fault):
    raw = body()
    changes = {
        "sheet": (b'Name="Historical"', b'Name="Unknown"'),
        "field": (b"Shares Outstanding", b"MarketPrice"),
        "sparse": (b"<ss:Cell>", b'<ss:Cell ss:Index="2">'),
        "entity": (b"<ss:Table>", b"<ss:Table><!DOCTYPE test>"),
        "badxml": (b"</ss:Row>", b"</wrong>"),
        "order": (b"Sep 11, 2026", b"Sep 16, 2026"),
    }
    if fault == "oversize":
        raw = b"x" * (d.MAX_BYTES + 1)
    else:
        raw = raw.replace(*changes[fault], 1)
    with pytest.raises((ValueError, d.ET.ParseError)):
        d.parse(raw)


@pytest.fixture
def ready(tmp_path, monkeypatch):
    calendar, sessions = b.calendar(), d.overnight.sessions()
    monkeypatch.setattr(b, "ROOT", tmp_path)
    monkeypatch.setattr(b, "calendar", lambda: calendar)
    monkeypatch.setattr(d.overnight, "sessions", lambda: sessions)
    clock = [datetime(2026, 9, 16, 7, 10, tzinfo=b.ZONE)]
    monkeypatch.setattr(b, "now", lambda: clock[0])
    monkeypatch.setattr(
        b, "window", lambda at: {"status": "OPEN", "base_nav_date": "2026-09-15", "target_nav_date": "2026-09-16"}
    )
    b.save(tmp_path / "protocol.json", {"deadline_at": "2026-09-17T12:11:38+08:00"})
    calls = []
    monkeypatch.setattr(d.parent, "fetch_fxi", lambda: (calls.append(1) or body(), {}))
    return clock, calls


def test_capture_reuses_parent_without_any_new_request(ready):
    _, calls = ready
    with pytest.raises(ValueError, match="PARENT_INPUT_UNAVAILABLE"):
        d.capture(b.now())
    assert calls == []
    parent = d.parent.capture(b.now())
    value = d.capture(b.now())
    assert value["parent_input_hash"] == b.digest(parent) and len(value["rows"]) == 6
    assert d.capture(b.now()) == value and calls == [1]
    path = d.parent.root() / "2026-09-16/raw/0700.xls"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="LIVE_RAW_CHANGED"):
        d.load("2026-09-16")


def test_rehashed_share_values_cannot_override_raw_source(ready):
    d.parent.capture(b.now())
    value = d.capture(b.now())
    value["rows"]["2026-09-09"]["shares"] += 100
    b.save(d.root() / "2026-09-16/input.json", value, replace=True)
    with pytest.raises(ValueError, match="LIVE_INPUT_CHANGED"):
        d.load("2026-09-16")


def test_parent_success_does_not_hide_missing_older_share_date(ready, monkeypatch):
    monkeypatch.setattr(d.parent, "fetch_fxi", lambda: (body(omit=9), {}))
    d.parent.capture(b.now())
    with pytest.raises(ValueError, match="REQUIRED_DATE_ABSENT"):
        d.capture(b.now())


def test_late_readback_cannot_be_eligible(ready, monkeypatch):
    clock, _ = ready
    d.parent.capture(b.now())
    original_load = d.load

    def late(target):
        value = original_load(target)
        clock[0] = datetime(2026, 9, 16, 8, 30, tzinfo=b.ZONE)
        return value

    monkeypatch.setattr(d, "load", late)
    with pytest.raises(ValueError, match="READBACK_LATE"):
        d.capture(b.now())


def test_history_recomputes_source_even_when_json_is_rehashed(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "ROOT", tmp_path)
    folder, original = tmp_path / "round-41", tmp_path / "fxi-data-feasibility-v1"
    folder.mkdir()
    original.mkdir()
    raw, prototype = body(), b"synthetic"
    (original / "response.xls").write_bytes(raw)
    (folder / "shares-feasibility.py").write_bytes(prototype)
    old_plan = {"url": d.URL}
    b.save(original / "plan.json", old_plan)
    capture = {"plan_hash": b.digest(old_plan), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    b.save(original / "result.json", capture)
    proposal = {
        "raw_sha256": capture["sha256"],
        "capture_hash": b.digest(capture),
        "script_sha256": hashlib.sha256(prototype).hexdigest(),
    }
    b.save(folder / "proposal-before-training.json", proposal)
    snapshot = {"proposal_hash": b.digest(proposal), "points": d.parse(raw)}
    b.save(folder / "shares-history.json", snapshot)
    result = {"proposal_hash": b.digest(proposal), "history_hash": b.digest(snapshot), "status": "FEASIBLE_NOT_TRAINED"}
    b.save(folder / "input-feasibility.json", result)
    assert d.history() == snapshot["points"]
    snapshot["points"]["2026-09-09"]["shares"] += 100
    b.save(folder / "shares-history.json", snapshot, replace=True)
    b.save(folder / "input-feasibility.json", result | {"history_hash": b.digest(snapshot)}, replace=True)
    with pytest.raises(ValueError, match="HISTORICAL_PARSED_CHANGED"):
        d.history()
