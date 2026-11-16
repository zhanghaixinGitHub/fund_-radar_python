"""变化待办的真实状态转换、摘要与持仓边界。"""

from copy import deepcopy
from datetime import datetime, timedelta

import pytest
from app.services import prediction_review_watch as watch
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import read
from app.services.fund_materials_store import versioned_save
from app.services.fund_prediction_snapshot_compat import materials_input_identity

AT = datetime.fromisoformat("2026-11-13T18:00:00+08:00")
MATERIAL = {
    "fund_code": "005284",
    "report": {"endDate": "2026-09-30"},
    "documents": [{"id": "terms", "body": "到期现金108元，含末息"}],
    "market": {"close": 114.2},
    "facts": {"earnings": "主要经营仍有分歧"},
}
PERIOD = {
    "horizonId": "T20_V1",
    "analysisAt": "2026-11-13T17:33:00+08:00",
    "decision": "ABSTAIN",
    "baseNavDate": "2026-11-11",
    "startDate": "2026-11-12",
    "endDate": "2026-12-09",
    "nominalEndDate": None,
    "conditions": ["经营与主要券共同转弱则重审"],
}


def transition(material=None, baseline=None, previous=None, period=None, at=AT):
    material = material or deepcopy(MATERIAL)
    return watch.watch_transition(
        "owner",
        "005284",
        period or PERIOD,
        baseline if baseline is not None else materials_input_identity(MATERIAL),
        {"materials_hash": digest(material), "materials_path": "D:/material.json"},
        material,
        at=at,
        previous=previous,
    )


def test_unchanged_review_is_complete_without_direction_and_observation_is_idempotent():
    first, changed = transition()
    assert changed and first["status"] == "WATCHING"
    assert first["review_complete"] and not first["direction_available"]
    assert not first["review_required"] and not first["completes_new_daily_window"]
    second, changed = transition(previous=first, at=AT + timedelta(days=1))
    assert not changed and second is first


@pytest.mark.parametrize(
    "section,value",
    [
        ("documents", [{"id": "terms", "body": "已公告调整转股价格"}]),
        ("market", {"close": 110.0}),
        ("report", {"endDate": "2026-12-31"}),
        ("facts", {"earnings": "最新经营指标改变"}),
    ],
)
def test_new_economic_input_opens_one_original_window_work_item(section, value):
    material = deepcopy(MATERIAL)
    material[section] = value
    state, changed = transition(material)
    assert changed and state["status"] == "REVIEW_REQUIRED"
    assert state["original_period"]["baseNavDate"] == "2026-11-11"
    assert state["next_action"] == "CONTINUE_ORIGINAL_WINDOW_REVIEW"
    again, changed = transition(material, previous=state, at=AT + timedelta(hours=1))
    assert not changed and again["work_item_id"] == state["work_item_id"]


def test_reapproval_time_alone_does_not_create_new_economic_evidence():
    first, _ = transition()
    material = {**MATERIAL, "workflow_reviewed_at": AT.isoformat(), "workflow_review_note": "复用原资料"}
    second, changed = transition(material, previous=first)
    assert not changed and second is first


def test_new_verified_review_acknowledges_input_but_first_observation_cannot():
    changed = {**MATERIAL, "market": {"close": 110}}
    pending, _ = transition(changed)
    review = {**PERIOD, "analysisAt": "2026-11-13T18:01:00+08:00"}
    closed, _ = transition(changed, baseline=materials_input_identity(changed), previous=pending, period=review)
    assert closed["status"] == "WATCHING" and closed["review_complete"]
    unknown, _ = watch.watch_transition(
        "owner",
        "005284",
        PERIOD,
        None,
        {"materials_path": "D:/material.json", "materials_hash": digest(MATERIAL)},
        MATERIAL,
        at=AT,
    )
    assert unknown["status"] == "WAITING_REVIEW_BASELINE" and unknown["review_required"]


def test_expired_window_stops_and_nominal_date_does_not_become_real_end():
    state, _ = transition(at=datetime.fromisoformat("2026-12-09T15:00:00+08:00"))
    assert state["status"] == "WINDOW_ENDED" and not state["review_required"]
    state, _ = transition(
        period={**PERIOD, "horizonId": "M6_V1", "endDate": None, "nominalEndDate": "2027-05-11"},
        at=datetime.fromisoformat("2027-05-11T09:00:00+08:00"),
    )
    assert state["status"] == "WAITING_OFFICIAL_END_DATE"
    assert state["original_period"]["endDate"] is None


def test_material_identity_mismatch_never_becomes_no_change():
    with pytest.raises(ValueError, match="CURRENT_MATERIAL_CHANGED"):
        watch.watch_transition(
            "owner",
            "005284",
            PERIOD,
            "old",
            {"materials_hash": "wrong", "materials_path": "D:/material.json"},
            MATERIAL,
            at=AT,
        )


def test_continuation_review_binds_new_material_and_rejects_later_file_change(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(watch, "PROJECT", tmp_path)
    source = tmp_path / "material.json"
    material = {**MATERIAL, "market": {"close": 110}}
    versioned_save(source, material)
    packet = {"reviewed_materials": {"005284": {"materials_path": str(source), "materials_hash": digest(material)}}}
    source_hash = digest(packet)
    archive = {
        "cases": [{"case": {"fund_code": "005284", "horizon": "T20_V1"}, "source_hash": source_hash}],
        "source_packets": {source_hash: {"canonical_json": json.dumps(packet)}},
    }
    assert watch.reviewed_material_identity(archive, "005284", "T20_V1") == materials_input_identity(material)
    versioned_save(source, MATERIAL)
    with pytest.raises(ValueError, match="WATCH_REVIEWED_MATERIAL_CHANGED"):
        watch.reviewed_material_identity(archive, "005284", "T20_V1")


def test_real_file_observation_reuses_pending_without_write_and_removed_holding_is_excluded(tmp_path, monkeypatch):
    from app.commands import fund_prediction_followthrough as command
    from app.services import fund_workflow_materials as workflow

    material = {**MATERIAL, "market": {"close": 110}}
    source = tmp_path / "material.json"
    versioned_save(source, material)
    held = ["005284"]
    monkeypatch.setattr(command, "owner_scope", lambda *_: held)
    monkeypatch.setattr(
        watch, "stored_reviews", lambda owner, codes: [("005284", [PERIOD], {})] if "005284" in codes else []
    )
    monkeypatch.setattr(watch, "reviewed_material_identity", lambda *_: materials_input_identity(MATERIAL))
    monkeypatch.setattr(
        workflow,
        "registered_materials",
        lambda *a, **kw: {"materials_path": str(source), "materials_hash": digest(material)},
    )
    monkeypatch.setattr(watch, "storage_check", lambda *_: None)
    first = watch.watch_report("owner", ["005284"], at=AT, root=tmp_path)
    assert first["reassessment_pending_count"] == 1 and first["reviewed_abstain_count"] == 1
    path = tmp_path / digest("owner") / "005284/T20_V1.json"
    saved, modified = read(path), path.stat().st_mtime_ns
    again = watch.watch_report("owner", ["005284"], at=AT + timedelta(minutes=1), root=tmp_path)
    assert again["items"] == first["items"] and path.stat().st_mtime_ns == modified
    monkeypatch.setattr(workflow, "registered_materials", lambda *a, **kw: None)
    failed = watch.watch_report("owner", ["005284"], at=AT, root=tmp_path)
    assert failed["failure_count"] == 1 and failed["status"] == "CHECK_FAILED"
    assert read(path) == saved
    held.clear()
    assert watch.watch_report("owner", ["005284"], at=AT, root=tmp_path)["items"] == []
    assert path.exists()
