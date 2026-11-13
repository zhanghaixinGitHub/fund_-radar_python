"""每日第二版只接真实新窗口，条件研判和事后研究另列。"""

from datetime import datetime
from pathlib import Path

import pytest
from app.services import fund_daily_prediction_v2 as v2
from app.services import fund_daily_runtime as runtime
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import read
from app.services.fund_materials_store import versioned_save

WINDOW = {"base_nav_date": "2026-11-12", "target_nav_date": "2026-11-13"}


def record(horizon="T1", **updates):
    return {
        "id": "record-1",
        "fund_code": "002112",
        "horizon": horizon,
        "base_date": "2026-11-12",
        "target_date": "2026-11-13",
        "end_date": "2026-11-13",
        "analysis_version": "COMPREHENSIVE_REVIEW_V2",
        "policy_hash": digest(v2.policy_v2()),
        "owner_verified": True,
        "snapshot_verified": True,
        "materials_input_hash": "current-material",
        "record_kind": "DATED_FORECAST",
        "ex_ante_eligible": True,
        "decision": "UP",
        **updates,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"target_date": "2026-11-12"},
        {"base_date": "2026-11-11"},
        {"materials_input_hash": "old-material"},
        {"owner_verified": False},
        {"snapshot_verified": False},
        {"analysis_version": "V1"},
        {"record_kind": "RETROSPECTIVE", "ex_ante_eligible": False},
    ],
)
def test_old_research_wrong_owner_or_material_cannot_complete_new_window(change):
    result = v2.verify_outputs(
        "002112", WINDOW, [record(**change)], materials_input_hash="current-material", materials_ready=True
    )
    row = result["horizons"][0]
    assert row["status"] == "ACTION_REQUIRED"
    assert row["generationStatus"] == "WAITING_ANALYSIS"
    assert row["eligible_for_accuracy"] is False


def test_conditional_and_abstain_save_are_not_dated_direction_success():
    rows = [
        record(),
        record("T5_V1", decision="ABSTAIN"),
        record("M6_V1", record_kind="CONDITIONAL_OUTLOOK", end_date=None, ex_ante_eligible=False),
    ]
    result = v2.verify_outputs("002112", WINDOW, rows, materials_input_hash="current-material")
    assert [r["status"] for r in result["horizons"]] == [
        "SAVED_AND_READ_BACK",
        "REVIEWED_ABSTAIN",
        "ACTION_REQUIRED",
        "CONDITIONAL_SAVED",
    ]
    assert [r["counts_as_dated_forecast"] for r in result["horizons"]] == [True, False, False, False]


def test_active_v2_saved_records_never_reads_old_tables_or_env(monkeypatch):
    from app.commands import fund_prediction_followthrough as command

    expected = [record()]
    monkeypatch.setattr(v2, "adoption", lambda: {"version": "COMPREHENSIVE_REVIEW_V2"})
    monkeypatch.setattr(v2, "saved_records", lambda owner, codes, target: expected)
    monkeypatch.setattr(command, "dotenv_values", lambda *_: pytest.fail("不能读取旧归档"))
    assert command.saved_records("owner", ["002112"], "2026-11-13", "unused") == expected


def test_active_runtime_freeze_skips_old_v1_even_with_legacy_long_term_gap(tmp_path, monkeypatch):
    backend = object.__new__(runtime.LocalDailyBackend)
    backend.owner, backend.window = "owner", WINDOW
    handoff = {"materials_path": "current", "reviewed": False, "gaps": ["旧长期盈利字段缺失"]}
    monkeypatch.setattr(v2, "adoption", lambda: {"version": "COMPREHENSIVE_REVIEW_V2"})
    monkeypatch.setattr(runtime, "registered_materials", lambda *a, **k: handoff)
    calls = []
    monkeypatch.setattr(v2, "freeze_request", lambda *a: calls.append(a) or {"generationStatus": "WAITING_ANALYSIS"})
    assert backend.freeze("002112", "T1", tmp_path)["generationStatus"] == "WAITING_ANALYSIS"
    assert calls[0] == (tmp_path, "owner", "002112", "T1", WINDOW, handoff)


def test_v1_approval_and_review_file_do_not_become_v2_approval(tmp_path, monkeypatch):
    backend = object.__new__(runtime.LocalDailyBackend)
    monkeypatch.setattr(v2, "adoption", lambda: {"version": "COMPREHENSIVE_REVIEW_V2"})
    with pytest.raises(ValueError, match="V2_STRICT_REVIEWED_TRIAL_REFERENCE_REQUIRED"):
        backend.approve("002112", {"note": "旧说明"})
    versioned_save(tmp_path / "review.json", {"decision": "UP"})
    with pytest.raises(ValueError, match="V2_STRICT_REVIEWED_TRIAL_REFERENCE_REQUIRED"):
        v2.import_reviewed_trial(tmp_path, "owner", "002112", "T1", WINDOW, "current-material")


def test_freeze_request_is_idempotent_and_keeps_cycle_rules(tmp_path, monkeypatch):
    from app.services import prediction_review_v2_requests as requests

    material = {"fund_code": "002112", "documents": [], "facts": {}}
    source = tmp_path / "material.json"
    versioned_save(source, material)
    monkeypatch.setattr(v2, "adoption", lambda: {"policyHash": digest(v2.policy_v2())})
    monkeypatch.setattr(v2, "planning_window", lambda at: WINDOW)
    calls = []
    monkeypatch.setattr(
        requests,
        "request_batch",
        lambda codes, **kwargs: calls.append((codes, kwargs)) or {"taskId": "pending", "status": "WAITING_ANALYSIS"},
    )
    handoff = {"materials_path": str(source), "materials_hash": digest(material), "gaps": ["缺项待本周期核对"]}
    first = v2.freeze_request(tmp_path, "owner", "002112", "T1", WINDOW, handoff)
    again = v2.freeze_request(tmp_path, "owner", "002112", "T1", WINDOW, handoff)
    assert first["input_hash"] == again["input_hash"]
    request = read(Path(first["path"]))
    assert request["period_rule"]["id"] == "PREDICTION_V2_T1_SPLIT_1"
    assert request["format"] == "FUND_PREDICTION_V2_DAILY_ANALYSIS_REQUEST"
    assert request["legacy_material_gaps"] == handoff["gaps"]
    assert "decision" not in request
    assert calls == [(["002112"], {"horizons": ["T1"]})] * 2


def test_import_ref_wrong_window_rejected_before_publication(tmp_path, monkeypatch):
    from app.services import prediction_review_v2_publication as publication
    from app.services import prediction_trial_v2 as trial

    directory = tmp_path / "strict-trial"
    versioned_save(tmp_path / "reviewed-trial-ref.json", {"trial_directory": str(directory)})
    versioned_save(
        directory / "manifest.json",
        {"inventory": {"owner_hash": digest("owner")}, "selection": [{"fund_code": "002112"}]},
    )
    monkeypatch.setattr(trial, "local_path", lambda p: Path(p))
    monkeypatch.setattr(
        trial,
        "checked_input",
        lambda *args: ({"periods": {"T1": {"base_date": "2026-11-11", "start_date": "2026-11-12"}}}, {}),
    )
    monkeypatch.setattr(publication, "publish_trial", lambda *a, **k: pytest.fail("不得发布旧窗口"))
    with pytest.raises(ValueError, match="V2_DAILY_TRIAL_WINDOW_MISMATCH"):
        v2.import_reviewed_trial(tmp_path, "owner", "002112", "T1", WINDOW, "current-material")


def test_report_does_not_count_conditional_as_all_predictions_generated(tmp_path, monkeypatch):
    from app.services import fund_workflow_materials as workflow
    from app.services.fund_prediction_snapshot_compat import materials_input_identity

    material = {"fund_code": "002112"}
    source = tmp_path / "material.json"
    versioned_save(source, material)
    monkeypatch.setattr(workflow, "registered_materials", lambda *a, **k: {"materials_path": str(source)})
    rows = [record(h, materials_input_hash=materials_input_identity(material)) for h in ("T1", "T5_V1", "T20_V1")]
    rows.append(
        record(
            "M6_V1",
            materials_input_hash=materials_input_identity(material),
            end_date=None,
            record_kind="CONDITIONAL_OUTLOOK",
            ex_ante_eligible=False,
        )
    )
    report = v2.followthrough_report(
        "owner", ["002112"], WINDOW, rows, at=datetime.fromisoformat("2026-11-13T09:50:00+08:00")
    )
    assert report["all_reviews_saved"] is True
    assert report["all_predictions_generated"] is False
    assert report["conditional_saved_count"] == 1
    assert report["dated_prediction_count"] == 3
