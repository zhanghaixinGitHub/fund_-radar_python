from pathlib import Path

import pytest
from app.services.fund_daily_review import prepare_daily_review
from app.services.fund_exposure_common import read
from app.services.fund_materials_store import versioned_save


def test_prepare_then_evaluate_actual_file_and_only_then_link_for_publication(tmp_path, monkeypatch):
    from app.commands import fund_prediction_v2_sources as sources
    from app.services import prediction_trial_v2 as trial

    request = {
        "owner_hash": "owner", "fund_code": "001632", "window": {}, "policy_hash": "rules",
        "materials_input_hash": "material", "materials_hash": "full-material", "nav_input_hash": "nav-1",
    }
    path, reference = tmp_path / "request.json", tmp_path / "reviewed-trial-ref.json"
    versioned_save(path, request)
    packet = {"path": str(path), "review_path": str(reference)}
    calls = []
    monkeypatch.setattr(trial, "storage_check", lambda since: {"count": 0})

    def freeze(directory, **kwargs):
        calls.append("sources")
        result = {"frozen_count": 1, "selection_path": str(directory / "selection.json")}
        versioned_save(directory / "freeze-summary.json", result)
        return result

    def prepare(directory, selection):
        calls.append("prepare")
        versioned_save(directory / "manifest.json", {"prepared": True})

    def evaluate(directory, reviews):
        calls.append("evaluate")
        if Path(reviews).read_text() != "actual-reviewed-input":
            raise ValueError("REVIEW_INVALID")
        return {"review_count": 4}

    monkeypatch.setattr(sources, "freeze_sources", freeze)
    monkeypatch.setattr(trial, "prepare_trial", prepare)
    monkeypatch.setattr(trial, "evaluate_trial", evaluate)
    result = prepare_daily_review(tmp_path, packet)
    assert calls == ["sources", "prepare"] and not reference.exists()
    Path(result["reviews_file"]).write_text("invalid")
    with pytest.raises(ValueError, match="REVIEW_INVALID"):
        prepare_daily_review(tmp_path, packet)
    assert not reference.exists()
    Path(result["reviews_file"]).write_text("actual-reviewed-input")
    final = prepare_daily_review(tmp_path, packet)
    assert read(reference)["trial_directory"] == final["trial_directory"]
    # 同材料晚到净值改变，必须产生新的准备输入，不能沿用缺净值的旧包。
    versioned_save(path, {**request, "nav_input_hash": "nav-2"})
    changed = prepare_daily_review(tmp_path, packet)
    assert changed["trial_directory"] != result["trial_directory"]
    assert changed["evaluation"] is None


def test_storage_failure_stops_before_any_source_write(tmp_path, monkeypatch):
    from app.services import prediction_trial_v2 as trial

    path = tmp_path / "request.json"
    versioned_save(path, {k: "x" for k in (
        "owner_hash", "fund_code", "window", "policy_hash", "materials_input_hash", "materials_hash"
    )})

    def stop(since):
        raise ValueError("STORAGE_ERROR")

    monkeypatch.setattr(trial, "storage_check", stop)
    with pytest.raises(ValueError, match="STORAGE_ERROR"):
        prepare_daily_review(tmp_path, {"path": str(path)})
    assert not (tmp_path / "analysis").exists()


def test_transient_failed_freeze_is_retried_without_approving_any_review(tmp_path, monkeypatch):
    from app.commands import fund_prediction_v2_sources as sources
    from app.services import prediction_trial_v2 as trial

    path = tmp_path / "request.json"
    versioned_save(path, {k: "same" for k in (
        "owner_hash", "fund_code", "window", "policy_hash", "materials_input_hash", "materials_hash"
    )})
    count = []
    monkeypatch.setattr(trial, "storage_check", lambda since: {"count": 0})
    def freeze(directory, **kwargs):
        count.append(1)
        result = {"frozen_count": int(len(count) > 1), "selection_path": str(directory / "selection.json")}
        versioned_save(directory / "freeze-summary.json", result)
        return result
    def prepare(directory, selection):
        versioned_save(directory / "manifest.json", {"prepared": True})
    monkeypatch.setattr(sources, "freeze_sources", freeze)
    monkeypatch.setattr(trial, "prepare_trial", prepare)
    packet = {"path": str(path), "review_path": str(tmp_path / "reference.json")}
    assert prepare_daily_review(tmp_path, packet)["preparation_status"] == "WAITING_SOURCE_REPAIR"
    assert prepare_daily_review(tmp_path, packet)["preparation_status"] == "READ_AND_JUDGE"
    assert len(count) == 2 and not (tmp_path / "reference.json").exists()
