"""供应商调用预算、空结果停止和完成后只读复用；仅使用替身响应。"""

from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace

import pytest
from scripts import direction_1d_sector_acquire as acquire


def prepare(tmp_path, monkeypatch):
    identities = {c: {"index_code": c} for c in ("A", "B")}
    acquire.write_new(tmp_path / "identity.json", {"catalog": identities})
    plan = {
        "index_codes": ["A", "B"],
        "identity_files": {"identity.json": acquire.file_hash(tmp_path / "identity.json")},
    }
    acquire.write_new(tmp_path / "daily-plan.json", plan)
    acquire.write_new(tmp_path / "daily-plan-receipt.json", {"hash": acquire.digest(plan)})
    acquire.write_new(tmp_path / "budget.json", {"index_daily_calls": 8})
    monkeypatch.setattr(acquire, "metadata", lambda codes: {"rate_limit_per_minute": 200})
    monkeypatch.setattr(acquire, "sleep", lambda _: None)


def test_empty_index_stops_and_finished_archive_never_requeries(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch)
    calls = []

    class Client:
        def list_index_daily(self, code, *, start_date, end_date):
            calls.append((code, start_date.year))
            return (
                []
                if code == "B"
                else [SimpleNamespace(index_code=code, trade_date=date(start_date.year, 1, 4), close_price=100)]
            )

    @contextmanager
    def client():
        yield Client()

    monkeypatch.setattr(acquire, "client", client)
    result = acquire.acquire(tmp_path)
    assert result["calls"] == 5 and calls[-1] == ("B", 2021)
    assert len(list(tmp_path.glob("daily-*-reserved.json"))) == 5
    assert acquire.acquire(tmp_path)["new_api_calls"] == 0 and len(calls) == 5


def test_uncertain_reserved_call_not_repeated(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch)
    acquire.write_new(tmp_path / "daily-A-2021-reserved.json", {})

    @contextmanager
    def client():
        yield SimpleNamespace(list_index_daily=lambda *a, **kw: pytest.fail("不能重放不确定请求"))

    monkeypatch.setattr(acquire, "client", client)
    with pytest.raises(ValueError, match="UNCERTAIN_REQUEST"):
        acquire.acquire(tmp_path)


def test_changed_identity_prevents_any_client_call(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch)
    (tmp_path / "identity.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(acquire, "client", lambda: pytest.fail("身份变化时不能查询"))
    with pytest.raises(ValueError, match="IDENTITY_CHANGED"):
        acquire.acquire(tmp_path)
