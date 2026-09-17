"""最新美股现金日期、父文件复用、补取预算及实际到达边界。"""

from datetime import datetime
from xml.sax.saxutils import escape

import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_fxi_overnight_data as d


def body(nonfv="100", omit=None):
    rows = [["As Of", "NAV per Share", "Ex-Dividends", "Shares Outstanding", "Non-FV NAV"]]
    for day, nav in ((16, 200), (15, 112), (14, 110), (11, 105), (10, 100)):
        if day != omit:
            rows.append([f"Sep {day:02d}, 2026", str(nav), "0", "1000", str(nonfv) if day == 14 else "100"])
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


def test_newest_us_cash_date_before_target_deadline_and_no_later_date():
    assert d.required("2026-09-15", "2026-09-16") == ["2026-09-14", "2026-09-15"]
    assert d.parent.required("2026-09-15", "2026-09-16") == ["2026-09-11", "2026-09-14"]
    points = d.parse(body())
    assert d.features("2026-09-15", "2026-09-16", points) == pytest.approx([12, 2, 1])
    points["2026-09-16"]["nav_per_share"] = float("nan")
    assert d.features("2026-09-15", "2026-09-16", points) == pytest.approx([12, 2, 1])
    with pytest.raises(ValueError, match="NOT_ADJACENT"):
        d.required("2026-09-15", "2026-09-17")


@pytest.mark.parametrize("value", ["--", "0"])
def test_original_missingness_policy_still_applies(value):
    assert d.features("2026-09-15", "2026-09-16", d.parse(body(value))) == [0, 0, 0]
    with pytest.raises(ValueError, match="REQUIRED_ROW_ABSENT"):
        d.features("2026-09-15", "2026-09-16", d.parse(body(omit=15)))


def test_actual_receipt_must_follow_selected_cash_close():
    with pytest.raises(ValueError, match="SESSION_NOT_CLOSED"):
        d.closed_before(datetime(2026, 9, 16, 3, tzinfo=b.ZONE), "2026-09-15", "2026-09-16")
    d.closed_before(datetime(2026, 9, 16, 7, tzinfo=b.ZONE), "2026-09-15", "2026-09-16")


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
    parent_calls, own_calls = [], []
    monkeypatch.setattr(d.parent, "fetch_fxi", lambda: (parent_calls.append(1) or body(), {}))
    monkeypatch.setattr(d, "fetch_fxi", lambda: (own_calls.append(1) or body(), {}))
    return clock, parent_calls, own_calls


def test_valid_parent_fullfile_reused_without_new_request(ready):
    _, parent_calls, own_calls = ready
    parent = d.parent.capture(b.now())
    value = d.capture(b.now())
    assert value["raw_ref"]["kind"] == "ROUND31" and value["raw_ref"]["parent_input_hash"] == b.digest(parent)
    assert list(value["rows"]) == ["2026-09-14", "2026-09-15"]
    assert d.capture(b.now()) == value and parent_calls == [1] and own_calls == []
    path = d.parent.root() / "2026-09-16/raw/0700.xls"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="RAW_CHANGED"):
        d.load("2026-09-16")


def test_old_parent_dates_require_one_separate_bounded_capture(ready, monkeypatch):
    _, _, own_calls = ready
    monkeypatch.setattr(d.parent, "fetch_fxi", lambda: (body(omit=15), {}))
    d.parent.capture(b.now())
    value = d.capture(b.now())
    assert value["raw_ref"]["kind"] == "OWN" and own_calls == [1]
    assert d.capture(b.now()) == value and own_calls == [1]
    value["rows"]["2026-09-15"]["nav_per_share"] += 10
    b.save(d.root() / "2026-09-16/input.json", value, replace=True)
    with pytest.raises(ValueError, match="PARSED_INPUT_CHANGED"):
        d.load("2026-09-16")


def test_parent_corruption_does_not_trigger_a_fallback_request(ready):
    _, _, own_calls = ready
    d.parent.capture(b.now())
    path = d.parent.root() / "2026-09-16/raw/0700.xls"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="RAW_CHANGED"):
        d.capture(b.now())
    assert own_calls == []


def test_missing_current_date_consumes_slot_and_can_try_next_fixed_slot(ready, monkeypatch):
    clock, _, calls = ready
    monkeypatch.setattr(d, "fetch_fxi", lambda: (calls.append(1) or body(omit=15), {}))
    for hour, minute in [(7, 0), (7, 30), (8, 0)]:
        clock[0] = datetime(2026, 9, 16, hour, minute, tzinfo=b.ZONE)
        with pytest.raises(ValueError, match="REQUIRED_ROW_ABSENT"):
            d.capture(b.now())
        with pytest.raises(ValueError, match="SLOT_OR_BUDGET_EXHAUSTED"):
            d.capture(b.now())
    assert calls == [1, 1, 1]
    assert not (d.root() / "2026-09-16/input.json").exists()


def test_late_response_is_saved_but_not_used(ready, monkeypatch):
    clock, _, _ = ready

    def late():
        clock[0] = datetime(2026, 9, 16, 8, 30, tzinfo=b.ZONE)
        return body(), {}

    monkeypatch.setattr(d, "fetch_fxi", late)
    with pytest.raises(ValueError, match="RESPONSE_LATE"):
        d.capture(b.now())
    assert not (d.root() / "2026-09-16/input.json").exists()


def test_parent_reference_cannot_be_rehashed_to_another_input(ready):
    d.parent.capture(b.now())
    value = d.capture(b.now())
    value["raw_ref"]["parent_input_hash"] = "invalid"
    b.save(d.root() / "2026-09-16/input.json", value, replace=True)
    with pytest.raises(ValueError, match="PARENT_INPUT_CHANGED"):
        d.load("2026-09-16")


def test_own_raw_and_request_time_are_verified(ready):
    value = d.capture(b.now())
    assert value["raw_ref"]["kind"] == "OWN"
    path = d.root() / "2026-09-16/requests/0700.json"
    request = b.read(path)
    b.save(path, request | {"at": "2026-09-16T08:30:00+08:00"}, replace=True)
    with pytest.raises(ValueError, match="RESPONSE_CHANGED_OR_LATE"):
        d.load("2026-09-16")


def test_late_readback_cannot_become_an_eligible_input(ready, monkeypatch):
    clock, _, _ = ready
    original_load = d.load

    def late(target):
        value = original_load(target)
        clock[0] = datetime(2026, 9, 16, 8, 30, tzinfo=b.ZONE)
        return value

    monkeypatch.setattr(d, "load", late)
    with pytest.raises(ValueError, match="READBACK_LATE"):
        d.capture(b.now())
