"""验证VIX取价锚点、原始文件校验、请求预算及真实接收时间；测试不请求外部服务。"""

from contextlib import nullcontext
from datetime import datetime

import pytest
from app.integrations import cboe_sprint_vix as client
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_vix_data as d


def body():
    return (
        b"DATE,OPEN,HIGH,LOW,CLOSE\n"
        b"09/10/2026,20,21,19,20\n"
        b"09/11/2026,21,23,20,22\n"
        b"09/12/2026,50,60,40,55\n"
        b"09/14/2026,22,26,21,24\n"
        b"09/15/2026,100,110,90,105\n"
    )


def test_anchor_precedes_base_close_and_later_values_do_not_enter():
    points = d.parse(body())
    assert "2026-09-12" not in points
    assert d.required("2026-09-15", "2026-09-16") == ["2026-09-11", "2026-09-14"]
    before = d.features("2026-09-15", "2026-09-16", points)
    assert before == pytest.approx([1.2, (24 / 22 - 1) * 100])
    points["2026-09-15"]["close"] = 99999
    assert d.features("2026-09-15", "2026-09-16", points) == before
    assert d.parse(body())["2026-09-15"]["close"] == 105  # 缓存不暴露可变行。
    with pytest.raises(ValueError, match="NOT_ADJACENT"):
        d.required("2026-09-15", "2026-09-17")
    del points["2026-09-14"]
    with pytest.raises(ValueError, match="CASH_CLOSE_MISSING"):
        d.features("2026-09-15", "2026-09-16", points)


@pytest.mark.parametrize("fault", ["header", "nan", "negative", "ohlc", "duplicate", "reverse", "oversize"])
def test_invalid_csv_is_rejected(fault):
    raw = body()
    if fault == "header":
        raw = raw.replace(b"DATE,", b"date,")
    elif fault == "nan":
        raw = raw.replace(b",24\n", b",nan\n")
    elif fault == "negative":
        raw = raw.replace(b",24\n", b",-24\n")
    elif fault == "ohlc":
        raw = raw.replace(b",24\n", b",27\n")
    elif fault == "duplicate":
        raw += b"09/15/2026,100,110,90,105\n"
    elif fault == "reverse":
        lines = raw.splitlines()
        raw = b"\n".join([lines[0], *reversed(lines[1:])])
    else:
        raw = b"x" * (d.MAX_BYTES + 1)
    with pytest.raises(ValueError):
        d.parse(raw)


@pytest.fixture
def ready(tmp_path, monkeypatch):
    calendar = b.calendar()
    events = d.overnight.sessions()
    monkeypatch.setattr(b, "ROOT", tmp_path)
    monkeypatch.setattr(b, "calendar", lambda: calendar)
    monkeypatch.setattr(d.overnight, "sessions", lambda: events)
    clock = [datetime(2026, 9, 16, 7, 10, tzinfo=b.ZONE)]
    monkeypatch.setattr(b, "now", lambda: clock[0])
    monkeypatch.setattr(
        b, "window", lambda at: {"status": "OPEN", "base_nav_date": "2026-09-15", "target_nav_date": "2026-09-16"}
    )
    b.save(tmp_path / "protocol.json", {"deadline_at": "2026-09-17T12:11:38+08:00"})
    calls = []
    monkeypatch.setattr(d, "fetch_vix", lambda: (calls.append(1) or body(), {"Content-Type": "text/csv"}))
    return clock, calls


def test_success_is_shared_without_redownload_and_raw_changes_fail(ready):
    _, calls = ready
    value = d.capture(b.now())
    assert list(value["rows"]) == ["2026-09-11", "2026-09-14"]
    assert d.capture(b.now()) == value and calls == [1]
    path = d.root() / "2026-09-16/raw/0700.csv"
    path.write_bytes(body().replace(b",24\n", b",25\n"))
    with pytest.raises(ValueError, match="VIX_LIVE_RAW_CHANGED"):
        d.load("2026-09-16")


def test_rehashed_parsed_value_cannot_replace_raw_source(ready):
    value = d.capture(b.now())
    value["rows"]["2026-09-14"]["close"] = 25
    b.save(d.root() / "2026-09-16/input.json", value, replace=True)
    with pytest.raises(ValueError, match="VIX_LIVE_PARSED_INPUT_CHANGED"):
        d.load("2026-09-16")


def test_late_response_is_saved_but_not_eligible(ready, monkeypatch):
    clock, _ = ready

    def late():
        clock[0] = datetime(2026, 9, 16, 8, 30, tzinfo=b.ZONE)
        return body(), {}

    monkeypatch.setattr(d, "fetch_vix", late)
    with pytest.raises(ValueError, match="VIX_LIVE_RESPONSE_LATE"):
        d.capture(b.now())
    assert (d.root() / "2026-09-16/raw/0700.json").exists()
    assert not (d.root() / "2026-09-16/input.json").exists()


def test_failed_slot_never_retries_and_total_is_three(ready, monkeypatch):
    clock, calls = ready

    def failed():
        calls.append(1)
        raise ValueError("SOURCE_UNAVAILABLE")

    monkeypatch.setattr(d, "fetch_vix", failed)
    for hour, minute in [(7, 0), (7, 30), (8, 0)]:
        clock[0] = datetime(2026, 9, 16, hour, minute, tzinfo=b.ZONE)
        with pytest.raises(ValueError, match="SOURCE_UNAVAILABLE"):
            d.capture(b.now())
        with pytest.raises(ValueError, match="SLOT_OR_BUDGET_EXHAUSTED"):
            d.capture(b.now())
    assert len(calls) == 3
    clock[0] = datetime(2026, 9, 16, 8, 30, tzinfo=b.ZONE)
    with pytest.raises(ValueError, match="WINDOW_CLOSED"):
        d.capture(b.now())
    assert len(calls) == 3


@pytest.mark.parametrize(
    "status,limit,expected", [(200, 100, None), (301, 100, "HTTP_FAILED"), (200, 3, "RESPONSE_LIMIT")]
)
def test_client_fixed_url_redirect_rejection_and_size_limit(monkeypatch, status, limit, expected):
    class Response:
        status_code = status
        headers = {"Content-Type": "text/csv"}

        def iter_bytes(self, size):
            yield b"content"

    class Client:
        def stream(self, method, url):
            assert method == "GET" and url == client.URL
            return nullcontext(Response())

    def factory(**kwargs):
        assert kwargs["follow_redirects"] is False
        return nullcontext(Client())

    monkeypatch.setattr(client.httpx, "Client", factory)
    monkeypatch.setattr(client, "MAX_BYTES", limit)
    if expected:
        with pytest.raises(ValueError, match=expected):
            client.fetch_vix()
    else:
        assert client.fetch_vix()[0] == b"content"
