"""一日历史分页、真实读回、失败和动态持仓的隔离回归。"""

import json
from types import SimpleNamespace

import httpx
import pytest
from app.services import fund_daily_outcomes as service
from app.services.fund_exposure_common import read


def backend(tmp_path, monkeypatch, pages, live=None, scopes=None):
    from app.commands import fund_prediction_followthrough as scope

    scopes = scopes or [["006038"], ["006038"]]
    monkeypatch.setattr(scope, "owner_scope", lambda *args: scopes.pop(0))
    seen = []

    def handler(request):
        seen.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=live or dict(total=2, completed=2, pending=0, failed=0, remaining=0))
        page = pages.pop(0)
        page["dueBefore"] = str(service.now().date())
        return httpx.Response(200, json=page)

    b = SimpleNamespace(
        directory=tmp_path, owner="test-owner", settings=SimpleNamespace(core_service_base_url="http://local")
    )
    b.client = httpx.Client(transport=httpx.MockTransport(handler))
    b.get = lambda url: b.client.get(url).json()
    return b, seen


def page(cursor=None, status="SUCCEEDED", created=1):
    return dict(
        checked=1,
        completed=int(status == "SUCCEEDED"),
        created=created,
        pending=int(status == "PENDING_DATA"),
        failed=int(status == "FAILED"),
        nextCursor=cursor,
        items=[dict(forecastId="id", fundCode="006038", status=status)],
    )


def test_two_pages_keep_cutoff_and_read_back_real_owner_scope(tmp_path, monkeypatch):
    b, seen = backend(tmp_path, monkeypatch, [page("next"), page(created=0)])
    with b.client:
        result = service.reconcile(b)
    assert result["status"] == "SUCCEEDED"
    assert result["summary"] == dict(total=2, checked=2, completed=2, created=1, pending=0, failed=0, remaining=0)
    assert json.loads(seen[1].content)["after"] == "next"
    assert "/users/test-owner/outcomes/status" in str(seen[-1].url)
    assert read(tmp_path / "one-day-outcomes.json")["summary"]["created"] == 1


def test_missing_data_and_readback_gap_cannot_pass(tmp_path, monkeypatch):
    b, _ = backend(
        tmp_path,
        monkeypatch,
        [page(status="PENDING_DATA", created=0)],
        dict(total=2, completed=0, pending=1, failed=0, remaining=1),
    )
    with b.client:
        result = service.reconcile(b)
    assert result["status"] == "PARTIAL_SUCCESS"
    assert result["summary"]["pending"] == result["summary"]["remaining"] == 1


def test_repeated_cursor_and_foreign_fund_are_rejected(tmp_path, monkeypatch):
    b, _ = backend(tmp_path, monkeypatch, [page("same"), page("same")])
    with b.client, pytest.raises(ValueError, match="CURSOR_REPEATED"):
        service.reconcile(b)
    foreign = page()
    foreign["items"][0]["fundCode"] = "002112"
    b, _ = backend(tmp_path, monkeypatch, [foreign])
    with b.client, pytest.raises(ValueError, match="PAGE_INVALID"):
        service.reconcile(b)


def test_holdings_changed_during_reconcile_is_not_success(tmp_path, monkeypatch):
    b, _ = backend(tmp_path, monkeypatch, [page()], scopes=[["006038"], []])
    with b.client, pytest.raises(ValueError, match="HOLDINGS_CHANGED"):
        service.reconcile(b)


def test_daily_material_barrier_and_failure_receipt(tmp_path):
    from app.services.fund_daily_prediction import DailyPredictionRun
    from tests.test_fund_daily_prediction import Backend, reviews

    b = Backend(["006038"])
    calls = []
    b.one_day_outcomes = lambda: calls.append(True) or {"status": "PARTIAL_SUCCESS"}
    b.terminal = False
    run = DailyPredictionRun(tmp_path, b)
    assert run.advance()["status"] == "WAITING_MATERIALS" and calls == []
    b.terminal = True
    reviews(tmp_path, b.codes)
    assert run.advance()["status"] == "ONE_DAY_OUTCOMES_PENDING" and calls == [True]


def test_daily_check_requires_one_day_receipt_and_independent_readback():
    from app.commands.fund_daily_check import assess
    from tests.test_daily_holdings_check import completed

    state, guard, codes = completed()
    state.pop("one_day_outcomes")
    result = assess(state, guard, codes, [], "2026-11-11T09:50:00+08:00")
    assert {"ONE_DAY_OUTCOME_RECEIPT_MISSING", "ONE_DAY_OUTCOME_READBACK_MISSING"} <= set(result["issues"])


def test_explicit_archive_backfill_keeps_separate_scope_receipt_and_endpoints(tmp_path, monkeypatch):
    b, seen = backend(
        tmp_path, monkeypatch, [page()], dict(total=1, completed=1, pending=0, failed=0, remaining=0), scopes=[]
    )
    original = b.get
    b.get = lambda url: ["006038"] if url.endswith("/fund-codes") else original(url)
    with b.client:
        result = service.reconcile(b, all_archived=True)
    assert result["scope_kind"] == "OWNER_ARCHIVE"
    assert all("/outcomes/archive/" in str(request.url) for request in seen)
    assert (tmp_path / "one-day-archive-outcomes.json").exists()
    assert not (tmp_path / "one-day-outcomes.json").exists()
