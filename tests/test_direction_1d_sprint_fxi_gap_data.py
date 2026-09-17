"""发行方文件的缺失语义、现金日期、原始证据、读取预算和截止边界。"""

import hashlib
from contextlib import nullcontext
from datetime import datetime
from xml.sax.saxutils import escape

import pytest
from app.integrations import ishares_sprint_fxi as client
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_fxi_gap_data as d


def body(nonfv="100", omit=None):
    fields = ["As Of", "NAV per Share", "Ex-Dividends", "Shares Outstanding", "Non-FV NAV"]
    rows = [fields]
    for day, nav in [(16, 200), (15, 180), (14, 110), (11, 105), (10, 100)]:
        if day != omit:
            rows.append([f"Sep {day:02d}, 2026", str(nav), "0", "1000", str(nonfv) if day == 11 else "100"])
    xml = "".join(
        "<ss:Row>"
        + "".join('<ss:Cell><ss:Data ss:Type="String">' + escape(value) + "</ss:Data></ss:Cell>" for value in row)
        + "</ss:Row>"
        for row in rows
    )
    return (
        '<ss:Workbook xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">'
        '<ss:Worksheet ss:Name="Historical"><ss:Table>' + xml + "</ss:Table></ss:Worksheet></ss:Workbook>"
    ).encode()


def test_gap_units_lagged_cash_dates_and_future_exclusion():
    points = d.parse(body())
    assert d.required("2026-09-15", "2026-09-16") == ["2026-09-11", "2026-09-14"]
    assert d.features("2026-09-15", "2026-09-16", points) == pytest.approx([10, 5, 1])
    points["2026-09-15"]["nav_per_share"] = 99999
    assert d.features("2026-09-15", "2026-09-16", points) == pytest.approx([10, 5, 1])
    assert d.parse(body())["2026-09-15"]["nav_per_share"] == 180
    with pytest.raises(ValueError, match="NOT_ADJACENT"):
        d.required("2026-09-15", "2026-09-17")


@pytest.mark.parametrize("raw,reason", [("--", "EXPLICIT_MISSING"), ("0", "INVALID_ZERO")])
def test_unavailable_fields_keep_distinct_reasons_and_zero_mask(raw, reason):
    points = d.parse(body(raw))
    assert points["2026-09-11"]["unavailable_reason"] == reason
    assert points["2026-09-11"]["available"] is False
    assert d.features("2026-09-15", "2026-09-16", points) == [0, 0, 0]


def test_absent_row_or_fabricated_mask_cannot_hide_invalid_input():
    with pytest.raises(ValueError, match="REQUIRED_ROW_ABSENT"):
        d.features("2026-09-15", "2026-09-16", d.parse(body(omit=11)))
    points = d.parse(body())
    points["2026-09-11"]["available"] = False
    with pytest.raises(ValueError, match="AVAILABILITY_REASON_CHANGED"):
        d.features("2026-09-15", "2026-09-16", points)


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "N/A"])
def test_other_invalid_numbers_are_not_silently_masked(value):
    with pytest.raises(ValueError):
        d.parse(body(value))


@pytest.mark.parametrize("fault", ["sheet", "field", "sparse", "entity", "badxml", "oversize", "order"])
def test_invalid_raw_structure_is_rejected(fault):
    raw = body()
    if fault == "sheet":
        raw = raw.replace(b'Name="Historical"', b'Name="Unknown"')
    elif fault == "field":
        raw = raw.replace(b"Non-FV NAV", b"Price")
    elif fault == "sparse":
        raw = raw.replace(b"<ss:Cell>", b'<ss:Cell ss:Index="2">', 1)
    elif fault == "entity":
        raw = raw.replace(b"<ss:Table>", b"<ss:Table><!DOCTYPE test>")
    elif fault == "badxml":
        raw = raw.replace(b"</ss:Row>", b"</wrong>", 1)
    elif fault == "oversize":
        raw = b"x" * (d.MAX_BYTES + 1)
    elif fault == "order":
        raw = raw.replace(b"Sep 11, 2026", b"Sep 16, 2026")
    with pytest.raises(ValueError):
        d.parse(raw)


def test_row_count_is_bounded(monkeypatch):
    monkeypatch.setattr(d, "MAX_ROWS", 1)
    d.xml_rows.cache_clear()
    with pytest.raises(ValueError, match="ROW_LIMIT"):
        d.parse(body())


@pytest.fixture
def ready(tmp_path, monkeypatch):
    calendar = b.calendar()
    monkeypatch.setattr(b, "ROOT", tmp_path)
    monkeypatch.setattr(b, "calendar", lambda: calendar)
    clock = [datetime(2026, 9, 16, 7, 10, tzinfo=b.ZONE)]
    monkeypatch.setattr(b, "now", lambda: clock[0])
    monkeypatch.setattr(
        b, "window", lambda at: {"status": "OPEN", "base_nav_date": "2026-09-15", "target_nav_date": "2026-09-16"}
    )
    b.save(tmp_path / "protocol.json", {"deadline_at": "2026-09-17T12:11:38+08:00"})
    calls = []
    monkeypatch.setattr(d, "fetch_fxi", lambda: (calls.append(1) or body(), {}))
    return clock, calls


def test_capture_shared_and_raw_tampering_rejected(ready):
    _, calls = ready
    value = d.capture(b.now())
    assert list(value["rows"]) == ["2026-09-11", "2026-09-14"]
    assert d.capture(b.now()) == value and calls == [1]
    path = d.root() / "2026-09-16/raw/0700.xls"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="LIVE_RAW_CHANGED"):
        d.load("2026-09-16")


def test_rehashed_mask_cannot_replace_raw_data(ready):
    value = d.capture(b.now())
    value["rows"]["2026-09-11"].update(available=False, unavailable_reason="EXPLICIT_MISSING", non_fv_nav=None)
    b.save(d.root() / "2026-09-16/input.json", value, replace=True)
    with pytest.raises(ValueError, match="LIVE_PARSED_INPUT_CHANGED"):
        d.load("2026-09-16")


def test_late_response_saved_but_not_eligible(ready, monkeypatch):
    clock, _ = ready

    def late():
        clock[0] = datetime(2026, 9, 16, 8, 30, tzinfo=b.ZONE)
        return body(), {}

    monkeypatch.setattr(d, "fetch_fxi", late)
    with pytest.raises(ValueError, match="LIVE_RESPONSE_LATE"):
        d.capture(b.now())
    assert (d.root() / "2026-09-16/raw/0700.xls").exists()
    assert not (d.root() / "2026-09-16/input.json").exists()


def test_request_failure_consumes_slot_and_three_is_limit(ready, monkeypatch):
    clock, calls = ready

    def fail():
        calls.append(1)
        raise ValueError("SOURCE_UNAVAILABLE")

    monkeypatch.setattr(d, "fetch_fxi", fail)
    for hour, minute in [(7, 0), (7, 30), (8, 0)]:
        clock[0] = datetime(2026, 9, 16, hour, minute, tzinfo=b.ZONE)
        with pytest.raises(ValueError, match="SOURCE_UNAVAILABLE"):
            d.capture(b.now())
        with pytest.raises(ValueError, match="SLOT_OR_BUDGET_EXHAUSTED"):
            d.capture(b.now())
    assert calls == [1, 1, 1]


def test_historical_snapshot_rehash_cannot_override_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "ROOT", tmp_path)
    original, folder = tmp_path / "fxi-data-feasibility-v1", tmp_path / "fxi-masked-feasibility-v2"
    original.mkdir()
    folder.mkdir()
    raw = body()
    (original / "response.xls").write_bytes(raw)
    original_plan = {"url": d.URL}
    b.save(original / "plan.json", original_plan)
    capture = {"plan_hash": b.digest(original_plan), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    b.save(original / "result.json", capture)
    (folder / "check.py").write_bytes(b"synthetic")
    plan = {
        "raw_sha256": capture["sha256"],
        "capture_hash": b.digest(capture),
        "script_sha256": hashlib.sha256(b"synthetic").hexdigest(),
    }
    b.save(folder / "plan.json", plan)
    snapshot = {"plan_hash": b.digest(plan), "points": d.parse(raw)}
    b.save(folder / "history.json", snapshot)
    result = {"plan_hash": b.digest(plan), "history_hash": b.digest(snapshot)}
    b.save(folder / "result.json", result)
    assert d.history() == snapshot["points"]
    snapshot["points"]["2026-09-11"]["nav_per_share"] += 1
    b.save(folder / "history.json", snapshot, replace=True)
    b.save(folder / "result.json", result | {"history_hash": b.digest(snapshot)}, replace=True)
    with pytest.raises(ValueError, match="HISTORICAL_PARSED_CHANGED"):
        d.history()


@pytest.mark.parametrize(
    "status,limit,error", [(200, 8000000, None), (301, 8000000, "HTTP_FAILED"), (200, 3, "RESPONSE_LIMIT")]
)
def test_public_client_url_redirect_and_response_limit(monkeypatch, status, limit, error):
    class Response:
        status_code = status
        headers = {}

        def iter_bytes(self, size):
            yield body()

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["follow_redirects"] is False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def stream(self, method, url):
            assert method == "GET" and url == d.URL
            return nullcontext(Response())

    monkeypatch.setattr(client.httpx, "Client", Client)
    monkeypatch.setattr(client, "MAX_BYTES", limit)
    if error:
        with pytest.raises(ValueError, match=error):
            client.fetch_fxi()
    else:
        assert client.fetch_fxi()[0] == body()
