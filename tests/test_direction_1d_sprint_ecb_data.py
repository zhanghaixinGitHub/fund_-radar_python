"""ECB参考汇率单位换算、应发布日期、原始ZIP、时点和请求预算。"""

import io
import zipfile
from contextlib import nullcontext
from datetime import datetime

import pytest
from app.integrations import ecb_sprint_fx as client
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_ecb_data as d


def zipped(csv, name="eurofxref-hist.csv", extra=False):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, csv)
        if extra:
            archive.writestr("unexpected.txt", "x")
    return stream.getvalue()


CSV = "Date,USD,CNY\n2026-09-16,9,90\n2026-09-15,1.25,9.0\n2026-09-14,1.2,8.4\n2026-09-11,1.1,7.7\n2026-09-10,1.1,7.7\n"


def body():
    return zipped(CSV)


def test_units_known_publication_dates_and_future_exclusion():
    points = d.parse(body())
    assert d.required("2026-09-15", "2026-09-16") == ["2026-09-14", "2026-09-15"]
    values = d.features("2026-09-15", "2026-09-16", points)
    assert values == pytest.approx([(7.2 / 7 - 1) * 100, (1.25 / 1.2 - 1) * 100])
    points["2026-09-16"]["cny_per_eur"] = 99999
    assert d.features("2026-09-15", "2026-09-16", points) == values
    assert d.parse(body())["2026-09-16"]["cny_per_eur"] == 90
    with pytest.raises(ValueError, match="NOT_ADJACENT"):
        d.required("2026-09-15", "2026-09-17")
    del points["2026-09-15"]
    with pytest.raises(ValueError, match="REQUIRED_DATE_MISSING"):
        d.features("2026-09-15", "2026-09-16", points)


def test_target_holidays_are_not_confused_with_other_ecb_holidays():
    dates = d.publication_dates()
    assert "2026-04-03" not in dates and "2026-04-06" not in dates
    assert "2026-12-24" in dates and "2026-12-31" in dates
    assert "2021-04-02" not in dates and "2025-04-21" not in dates
    assert d.required("2026-04-03", "2026-04-07") == ["2026-04-01", "2026-04-02"]


@pytest.mark.parametrize(
    "fault", ["field", "nan", "negative", "duplicate", "reverse", "holiday", "path", "extra", "badzip", "oversize"]
)
def test_invalid_rate_file_is_rejected(fault):
    csv = CSV
    if fault == "field":
        csv = csv.replace("USD", "usd")
    elif fault == "nan":
        csv = csv.replace("1.25,9.0", "nan,9.0")
    elif fault == "negative":
        csv = csv.replace("1.25,9.0", "-1.25,9.0")
    elif fault == "duplicate":
        csv += "2026-09-10,1.1,7.7\n"
    elif fault == "reverse":
        lines = csv.splitlines()
        csv = "\n".join([lines[0], *reversed(lines[1:])])
    elif fault == "holiday":
        csv = csv.replace("2026-09-11", "2026-09-12")
    raw = zipped(csv, name="../escape.csv" if fault == "path" else "eurofxref-hist.csv", extra=fault == "extra")
    if fault == "badzip":
        raw = b"not a zip"
    elif fault == "oversize":
        raw = b"x" * (d.MAX_BYTES + 1)
    with pytest.raises(ValueError):
        d.parse(raw)


def test_zip_uncompressed_size_is_bounded(monkeypatch):
    monkeypatch.setattr(d, "MAX_UNCOMPRESSED", 1)
    d.zip_rows.cache_clear()
    with pytest.raises(ValueError, match="ZIP_STRUCTURE"):
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
    monkeypatch.setattr(d, "fetch_ecb_fx", lambda: (calls.append(1) or body(), {"Content-Type": "application/zip"}))
    return clock, calls


def test_success_shared_and_changed_raw_rejected(ready):
    _, calls = ready
    value = d.capture(b.now())
    assert list(value["rows"]) == ["2026-09-14", "2026-09-15"]
    assert d.capture(b.now()) == value and calls == [1]
    (d.root() / "2026-09-16/raw/0700.zip").write_bytes(zipped(CSV.replace("1.25,9.0", "1.25,9.1")))
    with pytest.raises(ValueError, match="LIVE_RAW_CHANGED"):
        d.load("2026-09-16")


def test_rehashed_parsed_values_cannot_replace_raw(ready):
    value = d.capture(b.now())
    value["rows"]["2026-09-15"]["cny_per_eur"] = 9.1
    b.save(d.root() / "2026-09-16/input.json", value, replace=True)
    with pytest.raises(ValueError, match="LIVE_PARSED_INPUT_CHANGED"):
        d.load("2026-09-16")


def test_late_response_is_saved_but_ineligible(ready, monkeypatch):
    clock, _ = ready

    def late():
        clock[0] = datetime(2026, 9, 16, 8, 30, tzinfo=b.ZONE)
        return body(), {}

    monkeypatch.setattr(d, "fetch_ecb_fx", late)
    with pytest.raises(ValueError, match="LIVE_RESPONSE_LATE"):
        d.capture(b.now())
    assert (d.root() / "2026-09-16/raw/0700.json").exists()
    assert not (d.root() / "2026-09-16/input.json").exists()


def test_failed_slot_not_retried_and_three_request_limit(ready, monkeypatch):
    clock, calls = ready

    def fail():
        calls.append(1)
        raise ValueError("SOURCE_UNAVAILABLE")

    monkeypatch.setattr(d, "fetch_ecb_fx", fail)
    for hour, minute in [(7, 0), (7, 30), (8, 0)]:
        clock[0] = datetime(2026, 9, 16, hour, minute, tzinfo=b.ZONE)
        with pytest.raises(ValueError, match="SOURCE_UNAVAILABLE"):
            d.capture(b.now())
        with pytest.raises(ValueError, match="SLOT_OR_BUDGET_EXHAUSTED"):
            d.capture(b.now())
    assert calls == [1, 1, 1]


@pytest.mark.parametrize(
    "status,limit,error", [(200, 100, None), (301, 100, "HTTP_FAILED"), (200, 3, "RESPONSE_LIMIT")]
)
def test_client_fixed_endpoint_and_response_limits(monkeypatch, status, limit, error):
    class Response:
        status_code = status
        headers = {}

        def iter_bytes(self, size):
            yield b"12345"

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["follow_redirects"] is False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def stream(self, method, url):
            assert method == "GET" and url == "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
            return nullcontext(Response())

    monkeypatch.setattr(client.httpx, "Client", Client)
    monkeypatch.setattr(client, "MAX_BYTES", limit)
    if error:
        with pytest.raises(ValueError, match=error):
            client.fetch_ecb_fx()
    else:
        assert client.fetch_ecb_fx()[0] == b"12345"
