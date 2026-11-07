from app.services import fund_prediction_outcome_recovery as service


def test_same_job_recovers_dynamic_ids_once_and_preserves_native_summary(tmp_path, monkeypatch):
    ids = ["first", "second", "third"]
    native = {"jobId": "job", "summary": {"pending": 3, "dueBefore": "2026-10-30"}}
    child = {"job_id": "job", "started_at": "2026-10-30T00:30:00+08:00"}
    calls = []
    monkeypatch.setattr(service, "pending_ids", lambda *a: {"total":len(ids), "ids":ids})
    monkeypatch.setattr(service, "verify_outcomes", lambda **kw: calls.append(kw) or {"completed": 3, "pending": 0})
    path = tmp_path / "recovery.json"
    result = service.recover(path, native, child)
    assert result["original_summary"] == native and native["summary"]["pending"] == 3
    assert calls == [{"limit": 3, "prediction_ids": ids}]
    assert service.recover(path, native, child) == result and len(calls) == 1


def test_recovery_budget_is_explicit_and_empty_or_complete_does_not_invent_checks(tmp_path, monkeypatch):
    native = {"summary": {"pending": 501, "dueBefore": "2026-10-30"}}
    child = {"job_id": "job", "started_at": "2026-10-30T00:30:00+08:00"}
    calls = []
    monkeypatch.setattr(service, "pending_ids", lambda *a: {"total":501, "ids":list(map(str, range(500)))})
    monkeypatch.setattr(service, "verify_outcomes", lambda **kw: calls.append(kw) or {"pending": 500})
    result = service.recover(tmp_path / "budget.json", native, child)
    assert len(calls[0]["prediction_ids"]) == 500 and result["remaining_beyond_budget"] == 1
    assert service.recover(tmp_path / "none.json", {"summary": {"pending": 0}}, child) is None
