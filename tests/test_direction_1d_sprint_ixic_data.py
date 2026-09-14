"""验证新指数采集的代码/日期范围、缺失拒绝及不重复消耗请求预算。"""

import json
from datetime import date
from types import SimpleNamespace

import pytest
from app.integrations import tushare_sprint_ixic as client
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_ixic_data as d
from pydantic import SecretStr


def body():
    return {
        "code": 0,
        "data": {
            "fields": client.FIELDS,
            "items": [
                ["IXIC", "20260910", 100.0, 99.0],
                ["IXIC", "20260911", 101.0, 100.0],
            ],
        },
    }


def test_parse_requires_requested_index_dates_and_continuous_prices():
    payload = body()
    start, end = date(2026, 9, 10), date(2026, 9, 11)
    assert len(d.validate(json.dumps(payload).encode(), start, end)) == 2
    payload["data"]["items"][1][3] = 99.0
    with pytest.raises(ValueError, match="IXIC_PREVIOUS_CLOSE_CHANGED"):
        d.validate(json.dumps(payload).encode(), start, end)
    payload = body()
    payload["data"]["items"][0][0] = "SPX"
    with pytest.raises(ValueError, match="IXIC_UNEXPECTED_OR_DUPLICATE_DATE"):
        d.validate(json.dumps(payload).encode(), start, end)


def test_missing_day_is_not_carried_forward():
    payload = body()
    payload["data"]["items"].pop()
    with pytest.raises(ValueError, match="IXIC_CALENDAR_COVERAGE_INCOMPLETE"):
        d.validate(json.dumps(payload).encode(), date(2026, 9, 10), date(2026, 9, 11))


def test_nonofficial_endpoint_and_excessive_range_are_rejected(monkeypatch):
    monkeypatch.setattr(
        client,
        "get_settings",
        lambda: SimpleNamespace(tushare_api_url="http://api.tushare.pro", tushare_token=SecretStr("test")),
    )
    monkeypatch.setattr(client.httpx, "Client", lambda **kwargs: pytest.fail("unexpected network"))
    with pytest.raises(ValueError, match="IXIC_OFFICIAL_ENDPOINT_REQUIRED"):
        client.fetch_ixic(date(2026, 9, 10), date(2026, 9, 11))
    with pytest.raises(ValueError, match="IXIC_QUERY_RANGE_INVALID"):
        client.fetch_ixic(date(2021, 1, 1), date(2026, 9, 11))


@pytest.fixture
def ready(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "ROOT", tmp_path)
    monkeypatch.setattr(
        d, "plan", lambda: {"source_id": "test-source", "maximum_requests": 1, "minimum_spacing_seconds": 7}
    )
    monkeypatch.setattr(d.overnight, "source", lambda: {"source_id": "test-source", "rate_limit_per_minute": 60})
    monkeypatch.setattr(d.time, "sleep", lambda seconds: None)


def test_invalid_raw_is_retained_without_repeating_request(ready, monkeypatch):
    payload = body()
    payload["data"]["items"].pop()
    calls = []

    def fetch(*args):
        calls.append(1)
        return json.dumps(payload).encode()

    monkeypatch.setattr(d, "fetch_ixic", fetch)
    for _ in range(2):
        with pytest.raises(ValueError, match="IXIC_CALENDAR_COVERAGE_INCOMPLETE"):
            d.query("sample", date(2026, 9, 10), date(2026, 9, 11))
    assert len(calls) == 1
    assert (d.root() / "raw/sample.json").exists()
    assert not (d.root() / "data/sample.json").exists()


def test_success_is_reused_and_total_budget_is_enforced(ready, monkeypatch):
    calls = []

    def fetch(*args):
        calls.append(1)
        return json.dumps(body()).encode()

    monkeypatch.setattr(d, "fetch_ixic", fetch)
    a = d.query("sample", date(2026, 9, 10), date(2026, 9, 11))
    assert d.query("sample", date(2026, 9, 10), date(2026, 9, 11)) == a
    with pytest.raises(ValueError, match="IXIC_REQUEST_BUDGET_EXHAUSTED"):
        d.query("another", date(2026, 9, 10), date(2026, 9, 11))
    assert len(calls) == 1


def test_failed_request_is_not_automatically_repeated(ready, monkeypatch):
    def fail(*args):
        raise ValueError("IXIC_PROVIDER_BUSINESS_FAILED")

    monkeypatch.setattr(d, "fetch_ixic", fail)
    with pytest.raises(ValueError, match="IXIC_PROVIDER_BUSINESS_FAILED"):
        d.query("sample", date(2026, 9, 10), date(2026, 9, 11))
    monkeypatch.setattr(d, "fetch_ixic", lambda *args: pytest.fail("unexpected retry"))
    with pytest.raises(ValueError, match="IXIC_PREVIOUS_REQUEST_NO_SUCCESS_NO_AUTO_RETRY"):
        d.query("sample", date(2026, 9, 10), date(2026, 9, 11))
