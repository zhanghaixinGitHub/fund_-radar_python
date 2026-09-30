"""002112 一键输入维护的日期、互斥和真实完成边界；所有来源均使用替身。"""

from contextlib import contextmanager
from datetime import datetime, timedelta

import pytest
from app.services import fund_exposure_runtime as runtime
from app.services.direction_1d_protocol import ZONE
from app.services.fund_exposure_common import read, save


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    at = datetime(2026, 9, 30, 10, tzinfo=ZONE)
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setattr(runtime, "now", lambda: at)
    save(tmp_path / "runtime-control.json", {"enabled": False, "until": "2026-12-31T15:00:00+08:00"})
    save(tmp_path / "runtime-state.json", {"next_attempt_at": (at + timedelta(hours=2)).isoformat()})
    calls = []

    @contextmanager
    def lock():
        calls.append("locked")
        try:
            yield True
        finally:
            calls.append("released")

    def source(name, result):
        def run(**kwargs):
            calls.append(name)
            return result
        return run

    monkeypatch.setattr(runtime, "execution_lock", lock)
    monkeypatch.setattr(runtime, "acquire", source("reports", {"errors": []}))
    monkeypatch.setattr(runtime, "acquire_quotes", source("quotes", {"errors": []}))
    monkeypatch.setattr(runtime, "acquire_nav", source("nav", {"rows": 61}))
    monkeypatch.setattr(runtime, "capture", source("capture", {"status": "INPUT_CAPTURED_CANDIDATE_BLOCKED"}))
    return tmp_path, calls


def test_manual_sync_ignores_old_timer_flags_but_keeps_lock_and_progress(prepared):
    root, calls = prepared
    control = (root / "runtime-control.json").read_bytes()
    progress = []
    result = runtime.synchronize_inputs(progress_reporter=lambda *args: progress.append(args))
    assert result["status"] == "SUCCEEDED" and result["created"] == 1
    assert calls == ["locked", "reports", "quotes", "nav", "capture", "released"]
    assert [row[0] for row in progress] == [0, 1, 2, 3, 4]
    assert all(row[1:3] == (4, "002112") for row in progress)
    assert (root / "runtime-control.json").read_bytes() == control
    assert "next_attempt_at" not in read(root / "runtime-state.json")


@pytest.mark.parametrize("capture_status", ["WAITING_NAV", "WAITING_QUOTES", "DEADLINE_PASSED", "UNKNOWN"])
def test_missing_or_late_inputs_never_count_as_complete(prepared, monkeypatch, capture_status):
    monkeypatch.setattr(runtime, "capture", lambda: {"status": capture_status})
    result = runtime.synchronize_inputs(progress_reporter=lambda *_: None)
    assert result["status"] == "PARTIAL_SUCCESS"
    assert result["created"] == result["skipped"] == 0
    assert capture_status not in result["message"]


def test_existing_snapshot_is_reused_but_source_failure_remains_partial(prepared, monkeypatch):
    monkeypatch.setattr(runtime, "capture", lambda: {"status": "ALREADY_CAPTURED"})
    result = runtime.synchronize_inputs(progress_reporter=lambda *_: None)
    assert result["status"] == "SUCCEEDED" and result["skipped"] == 1
    monkeypatch.setattr(runtime, "acquire_quotes", lambda **_: {"errors": ["synthetic-secret"]})

    def fail(**kwargs):
        raise ValueError("synthetic-secret")

    monkeypatch.setattr(runtime, "acquire_nav", fail)
    result = runtime.synchronize_inputs(progress_reporter=lambda *_: None)
    assert result["status"] == "PARTIAL_SUCCESS" and result["skipped"] == 1
    assert "行情" in result["message"] and "官网净值" in result["message"]
    assert "synthetic-secret" not in result["message"]


def test_expiry_and_missing_setup_do_not_trigger_sources(prepared):
    root, calls = prepared
    save(root / "runtime-control.json", {"enabled": True, "until": "2026-09-29T15:00:00+08:00"}, replace=True)
    assert runtime.synchronize_inputs(progress_reporter=lambda *_: None)["status"] == "FAILED"
    assert calls == ["locked", "released"]
    (root / "runtime-control.json").unlink()
    assert runtime.synchronize_inputs(progress_reporter=lambda *_: None)["status"] == "FAILED"
    assert calls == ["locked", "released"]


def test_busy_lock_never_runs_sources(prepared, monkeypatch):
    _, calls = prepared

    @contextmanager
    def busy():
        yield False

    monkeypatch.setattr(runtime, "execution_lock", busy)
    result = runtime.synchronize_inputs(progress_reporter=lambda *_: None)
    assert result["status"] == "FAILED" and "其他任务" in result["message"]
    assert calls == []


def test_failed_round_cannot_reuse_previous_success_or_schedule_retry(prepared, monkeypatch):
    root, calls = prepared
    save(root / "runtime-state.json", {
        "official_nav": {"rows": 61}, "capture": {"status": "INPUT_CAPTURED_CANDIDATE_BLOCKED"},
    }, replace=True)
    monkeypatch.setattr(runtime, "acquire", lambda **_: {"errors": ["test failure"]})
    result = runtime.synchronize_inputs(progress_reporter=lambda *_: None)
    state = read(root / "runtime-state.json")
    assert result["status"] == "FAILED" and result["created"] == 0
    assert not {"capture", "official_nav", "next_attempt_at"} & state.keys()
    assert calls == ["locked", "released"]
