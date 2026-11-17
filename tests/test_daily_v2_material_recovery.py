"""第二版每日任务必须实际补资料，且保留真实候选路径与失败。"""

import pytest
from app.services.fund_daily_runtime import LocalDailyBackend


def test_v2_recovers_before_requesting_review(monkeypatch, tmp_path):
    backend = object.__new__(LocalDailyBackend)
    backend.collector = object()
    calls = []
    window = {"base_nav_date": "2026-11-13", "target_nav_date": "2026-11-16"}
    candidate = {"status": "SUPPLEMENTED_REVIEW_REQUIRED", "materials_path": "new-candidate.json",
                 "market_attempts": [{"status": "ACQUIRED"}]}
    monkeypatch.setattr("app.services.fund_daily_prediction_v2.adoption", lambda: {})

    def recover(code, output, current_window, *, collector):
        calls.append((code, output, current_window, collector))
        return candidate

    monkeypatch.setattr("app.services.fund_prediction_recovery.recover", recover)
    inspections = []

    def inspect(code, path):
        inspections.append((code, path))
        return {"reviewed": False, "gaps": ["仍须真实阅读"]}

    monkeypatch.setattr("app.services.fund_workflow_materials.inspect_materials", inspect)
    result = backend.recover_materials("001632", tmp_path, window)
    assert calls == [("001632", tmp_path, window, backend.collector)]
    assert result["materials_path"] == "new-candidate.json"
    assert result["market_attempts"] == candidate["market_attempts"]
    assert result["status"] == "SUPPLEMENTED_REVIEW_REQUIRED"
    assert result["analysis_version"] == "COMPREHENSIVE_REVIEW_V2"
    assert inspections == [("001632", "new-candidate.json")]
    assert result["material_inspection"]["reviewed"] is False


def test_v2_does_not_report_old_materials_as_recovered(monkeypatch, tmp_path):
    backend = object.__new__(LocalDailyBackend)
    backend.collector = object()
    monkeypatch.setattr("app.services.fund_daily_prediction_v2.adoption", lambda: {})

    def unavailable(*args, **kwargs):
        raise ValueError("SOURCE_UNAVAILABLE")

    monkeypatch.setattr("app.services.fund_prediction_recovery.recover", unavailable)
    with pytest.raises(ValueError, match="SOURCE_UNAVAILABLE"):
        backend.recover_materials("001632", tmp_path, {})
