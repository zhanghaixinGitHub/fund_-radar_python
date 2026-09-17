"""CNYA自己的下载证据、最新现金日期、预算和截止边界；原解析器测试另行复用。"""

import hashlib
from contextlib import nullcontext
from datetime import datetime

import pytest
from app.integrations import ishares_sprint_cnya as client
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_cnya_data as d

from test_direction_1d_sprint_fxi_overnight_data import body


@pytest.fixture
def ready(tmp_path, monkeypatch):
    calendar, sessions = b.calendar(), d.timing.overnight.sessions()
    monkeypatch.setattr(b, "ROOT", tmp_path)
    monkeypatch.setattr(b, "calendar", lambda: calendar)
    monkeypatch.setattr(d.timing.overnight, "sessions", lambda: sessions)
    clock = [datetime(2026, 9, 16, 7, 10, tzinfo=b.ZONE)]
    monkeypatch.setattr(b, "now", lambda: clock[0])
    monkeypatch.setattr(
        b, "window", lambda at: {"status": "OPEN", "base_nav_date": "2026-09-15", "target_nav_date": "2026-09-16"}
    )
    b.save(tmp_path / "protocol.json", {"deadline_at": "2026-09-17T12:11:38+08:00"})
    calls = []
    monkeypatch.setattr(d, "fetch_cnya", lambda: (calls.append(1) or body(), {}))
    return clock, calls


def test_latest_session_and_one_source_download_shared_by_all_funds(ready):
    _, calls = ready
    value = d.capture(b.now())
    assert list(value["rows"]) == ["2026-09-14", "2026-09-15"]
    assert d.features(value["base"], value["target"], value["rows"]) == pytest.approx([12, 2, 1])
    assert d.capture(b.now()) == value and calls == [1]
    assert "273318" in d.URL and d.URL == client.URL
    assert not (b.ROOT / "round-31").exists()


@pytest.mark.parametrize("fault", ["raw", "parsed", "request_product", "receipt_before_close"])
def test_rehashed_metadata_cannot_hide_wrong_original_input(ready, fault):
    value = d.capture(b.now())
    folder = d.root() / "2026-09-16"
    if fault == "raw":
        path = folder / "raw/0700.xls"
        path.write_bytes(path.read_bytes() + b" ")
    elif fault == "parsed":
        value["rows"]["2026-09-15"]["nav_per_share"] += 1
        b.save(folder / "input.json", value, replace=True)
    else:
        request, meta = b.read(folder / "requests/0700.json"), b.read(folder / "raw/0700.json")
        if fault == "request_product":
            request["url"] = d.URL.replace("273318", "239536")
        else:
            request["at"] = meta["received_at"] = "2026-09-16T03:00:00+08:00"
        b.save(folder / "requests/0700.json", request, replace=True)
        meta["request_hash"] = b.digest(request)
        b.save(folder / "raw/0700.json", meta, replace=True)
        value["raw_ref"]["hash"] = b.digest(meta)
        b.save(folder / "input.json", value, replace=True)
    with pytest.raises(ValueError):
        d.load("2026-09-16")


def test_missing_latest_date_consumes_only_one_slot(ready, monkeypatch):
    clock, calls = ready
    monkeypatch.setattr(d, "fetch_cnya", lambda: (calls.append(1) or body(omit=15), {}))
    for hour, minute in [(7, 0), (7, 30), (8, 0)]:
        clock[0] = datetime(2026, 9, 16, hour, minute, tzinfo=b.ZONE)
        with pytest.raises(ValueError, match="REQUIRED_ROW_ABSENT"):
            d.capture(b.now())
        with pytest.raises(ValueError, match="SLOT_OR_BUDGET_EXHAUSTED"):
            d.capture(b.now())
    assert calls == [1, 1, 1]


@pytest.mark.parametrize("reason", ["--", "0"])
def test_explicit_missing_value_remains_eligible_with_mask(ready, monkeypatch, reason):
    monkeypatch.setattr(d, "fetch_cnya", lambda: (body(reason), {}))
    value = d.capture(b.now())
    assert d.features(value["base"], value["target"], value["rows"]) == [0, 0, 0]


@pytest.mark.parametrize("late_at", ["response", "readback"])
def test_late_response_or_readback_is_not_eligible(ready, monkeypatch, late_at):
    clock, _ = ready
    load = d.load

    def late_fetch():
        clock[0] = datetime(2026, 9, 16, 8, 30, tzinfo=b.ZONE)
        return body(), {}

    def late_load(target):
        value = load(target)
        clock[0] = datetime(2026, 9, 16, 8, 30, tzinfo=b.ZONE)
        return value

    monkeypatch.setattr(
        d, "fetch_cnya" if late_at == "response" else "load", late_fetch if late_at == "response" else late_load
    )
    with pytest.raises(ValueError, match="LATE"):
        d.capture(b.now())


def test_history_rechecks_original_fields_even_when_outer_hashes_recomputed(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(b, "ROOT", tmp_path)
    folder = tmp_path / "cnya-data-feasibility-v1"
    folder.mkdir()
    raw = body()
    (folder / "response.xls").write_bytes(raw)
    (folder / "probe.py").write_bytes(b"synthetic")
    plan = {
        "url": d.URL,
        "script_sha256": hashlib.sha256(b"synthetic").hexdigest(),
        "client_sha256": hashlib.sha256(Path(client.__file__).read_bytes()).hexdigest(),
        "parser_sha256": hashlib.sha256(Path(d.parser.__file__).read_bytes()).hexdigest(),
        "timing_sha256": hashlib.sha256(Path(d.timing.__file__).read_bytes()).hexdigest(),
    }
    request = {"at": "2026-09-15T18:40:00+08:00", "url": d.URL, "plan_hash": b.digest(plan)}
    capture = {
        "at": request["at"],
        "url": d.URL,
        "plan_hash": b.digest(plan),
        "request_hash": b.digest(request),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    snapshot = {"plan_hash": b.digest(plan), "capture_hash": b.digest(capture), "points": d.parse(raw)}
    result = {
        "status": "FEASIBLE_NOT_TRAINED",
        "plan_hash": b.digest(plan),
        "capture_hash": b.digest(capture),
        "history_hash": b.digest(snapshot),
    }
    for name, value in (
        ("plan", plan),
        ("request", request),
        ("capture", capture),
        ("history", snapshot),
        ("result", result),
    ):
        b.save(folder / (name + ".json"), value)
    assert d.history() == snapshot["points"]
    snapshot["points"]["2026-09-15"]["nav_per_share"] += 1
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
            client.fetch_cnya()
    else:
        assert client.fetch_cnya()[0] == body()
