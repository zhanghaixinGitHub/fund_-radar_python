"""成交额来源、日期、旧收盘和有限请求预算；全部客户端为替身，不调用真实接口。"""

from datetime import date, datetime
from types import SimpleNamespace

import pytest
from app.services import direction_1d_activity_data as data
from app.services.direction_1d_protocol import ZONE


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    days = tuple(date(y, 1, 4) for y in data.YEARS)
    monkeypatch.setattr(data, "calendar", lambda: (days, "synthetic"))
    monkeypatch.setattr(data, "sleep", lambda _: None)
    base = tmp_path / "base"
    (base / "base").mkdir(parents=True)
    data.write_new(base / "base/prices.json", {"prices": {data.INDEX: {str(d): "100" for d in days}}})
    data.write_new(base / "study.json", {"synthetic": True})
    source = {
        "source_id": "synthetic",
        "source_code": "TUSHARE_PRO_FUND",
        "enabled": True,
        "authorization_verified_at": datetime.now(ZONE).isoformat(),
        "authorized_api_names": ["index_basic", "index_daily"],
        "retention_days": 365,
        "rate_limit_per_minute": 200,
    }
    audit = tmp_path / "audit.json"
    data.write_new(audit, {"at": datetime.now(ZONE).isoformat(), "source": source, "local_index_amount_tables": []})
    plan = tmp_path / "plan.md"
    plan.write_text("合成测试", encoding="utf-8")
    root = tmp_path / "run"
    data.initialize(root, base, audit, plan)
    monkeypatch.setattr(
        data,
        "metadata",
        lambda _: {
            "source_id": "synthetic",
            "rate_limit_per_minute": 200,
            "catalog": {data.INDEX: {"display_name": "沪深300"}},
        },
    )
    return root, days


def fake_client(monkeypatch, *, bad_close=False, denied=False):
    calls = []

    class Fake:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def list_index_activity(self, code, *, start_date, end_date):
            calls.append((code, start_date, end_date))
            if denied:
                raise RuntimeError("permission denied")
            return [
                SimpleNamespace(
                    index_code=code,
                    trade_date=date(start_date.year, 1, 4),
                    close_price="101" if bad_close else "100",
                    amount="2000",
                )
            ]

    monkeypatch.setattr(data, "client", Fake)
    return calls


def test_four_calls_then_idempotent_no_provider_reentry(prepared, monkeypatch):
    root, _ = prepared
    calls = fake_client(monkeypatch)
    assert data.acquire(root)["new_api_calls"] == 4 and len(calls) == 4
    assert len(data.verify(root)["rows"]) == 4
    monkeypatch.setattr(data, "client", lambda: pytest.fail("完整回执不再请求"))
    assert data.acquire(root)["new_api_calls"] == 0


def test_first_year_old_close_mismatch_stops_remaining_calls(prepared, monkeypatch):
    root, _ = prepared
    calls = fake_client(monkeypatch, bad_close=True)
    with pytest.raises(ValueError, match="OLD_CLOSE_CHANGED"):
        data.acquire(root)
    assert len(calls) == 1 and (root / "2021.json").exists()
    assert not (root / "2022-reserved.json").exists()


def test_permission_failure_is_saved_without_retry(prepared, monkeypatch):
    root, _ = prepared
    calls = fake_client(monkeypatch, denied=True)
    with pytest.raises(ValueError, match="REQUEST_FAILED_OR_CHANGED"):
        data.acquire(root)
    assert len(calls) == 1
    failure = data.read(root / "2021.json")
    assert failure["permission_error"] and failure["error_type"] == "RuntimeError"
    assert "permission denied" not in (root / "2021.json").read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        data.acquire(root)
    assert len(calls) == 1


def test_uncertain_request_never_repeated(prepared, monkeypatch):
    root, _ = prepared
    data.write_new(root / "2021-reserved.json", {"uncertain": True})
    calls = fake_client(monkeypatch)
    with pytest.raises(ValueError, match="UNCERTAIN_REQUEST"):
        data.acquire(root)
    assert not calls


@pytest.mark.parametrize("amount", [None, "0", "-1", "NaN", "Infinity"])
def test_invalid_amount_cannot_pass_coverage(prepared, amount):
    _, days = prepared
    row = {"index_code": data.INDEX, "date": str(days[0]), "close": "100", "amount": amount}
    with pytest.raises(ValueError, match="VALUE_INVALID"):
        data.validate_rows([row], {str(days[0]): "100"}, 2021)


@pytest.mark.parametrize("case", ["missing", "duplicate", "identity", "protected_date"])
def test_invalid_identity_or_calendar_rejected(prepared, case):
    _, days = prepared
    row = {"index_code": data.INDEX, "date": str(days[0]), "close": "100", "amount": "1000"}
    rows = [dict(row)]
    if case == "missing":
        rows = []
    elif case == "duplicate":
        rows.append(dict(row))
    elif case == "identity":
        rows[0]["index_code"] = "000905.SH"
    else:
        rows[0]["date"] = "2025-01-04"
    with pytest.raises(ValueError, match="ACTIVITY_"):
        data.validate_rows(rows, {str(days[0]): "100"}, 2021)


def test_changed_response_invalidates_completed_seal(prepared, monkeypatch):
    root, _ = prepared
    fake_client(monkeypatch)
    data.acquire(root)
    path = root / "2021.json"
    path.write_text(path.read_text(encoding="utf-8").replace('"2000"', '"2001"'), encoding="utf-8")
    with pytest.raises(ValueError, match="FROZEN_RESPONSE_CHANGED"):
        data.verify(root)
