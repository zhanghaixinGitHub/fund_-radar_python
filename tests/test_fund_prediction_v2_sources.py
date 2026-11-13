"""第二版资料冻结的隔离验收；不访问真实本人接口或业务库。"""

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pytest
from app.commands import fund_prediction_v2_sources as command
from app.services import prediction_trial_v2 as trial
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import read
from app.services.fund_materials_store import versioned_save
from app.services.prediction_contract import PredictionFailure

AT = datetime.fromisoformat("2026-11-12T17:00:00+08:00")
WINDOW = {"base_nav_date": "2026-11-12", "target_nav_date": "2026-11-13", "status": "OPEN"}


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    material = {"fund_code": "002112", "documents": [], "facts": {}}
    source = tmp_path / "materials.json"
    versioned_save(source, material)
    inventory = {
        "owner_hash": digest("owner"),
        "scope": ["002112"],
        "profiles": [{"fund_code": "002112", "fund_type": "EQUITY"}],
    }
    data = {
        "fund_code": "002112",
        "window": WINDOW,
        "as_of": AT.isoformat(),
        "materials_path": str(source),
        "materials_hash": digest(material),
        "documents": [],
        "facts": {},
        "time_context": {"baseline_complete": True},
    }
    handoff = {
        "materials_path": str(source),
        "materials_hash": digest(material),
        "reviewed": False,
        "gaps": ["旧长期字段待核，本命令不代替第二版审阅"],
    }
    monkeypatch.setattr(command, "local_path", lambda p: Path(p))
    monkeypatch.setattr(trial, "local_path", lambda p: Path(p))
    monkeypatch.setattr(command, "window", lambda at: deepcopy(WINDOW))
    monkeypatch.setattr(command, "registered_materials", lambda *a, **k: deepcopy(handoff))
    monkeypatch.setattr(command, "prepare", lambda at, code, path: deepcopy(data))
    monkeypatch.setattr(command, "review_calendar", lambda *args: object())

    def dates(calendar, at, horizon):
        return {
            "horizonId": horizon,
            "baseNavDate": "2026-11-12",
            "startDate": "2026-11-13",
            "endDate": None if horizon == "M6_V1" else "2026-11-19",
            "nominalEndDate": "2027-05-12" if horizon == "M6_V1" else None,
            "endDateStatus": "PENDING_OFFICIAL_CALENDAR" if horizon == "M6_V1" else "RESOLVED",
            "calendarId": "verified-test-calendar",
            "calendarHash": "verified-test-hash",
        }

    monkeypatch.setattr(command, "target_dates", dates)
    return inventory, data, handoff


def test_freeze_selection_can_directly_prepare_without_v1_review(tmp_path, isolated):
    inventory, _, _ = isolated
    result = command.freeze_sources(tmp_path / "sources", holdings_provider=lambda: inventory, clock=AT)
    selection = json.loads(Path(result["selection_path"]).read_text(encoding="utf-8"))
    assert result["directions_generated"] == 0
    assert result["source_ready_count"] == 1
    assert selection[0]["source_kind"] == "FROZEN"
    source = read(Path(selection[0]["source_path"]))
    assert source["format"] == "CODEX_FUND_INPUT_V1"
    assert source["input_hash"] == digest(source["input"])
    trial.prepare_trial(tmp_path / "trial", Path(result["selection_path"]), holdings=inventory, clock=AT)
    packet = read(tmp_path / "trial/inputs/002112.json")
    assert packet["periods"]["T5_V1"]["status"] == "RESOLVED"
    assert packet["periods"]["M6_V1"]["status"] == "PENDING_OFFICIAL_CALENDAR"
    assert packet["periods"]["M6_V1"]["end_date"] is None
    assert packet["periods"]["M6_V1"]["nominal_end_date"] == "2027-05-12"
    assert "decision" not in source


def test_baseline_gap_is_not_promoted_to_source_ready(tmp_path, isolated):
    inventory, data, _ = isolated
    data["time_context"] = {"baseline_complete": False, "nav": {"status": "MISSING"}}
    result = command.freeze_sources(tmp_path / "sources", holdings_provider=lambda: inventory, clock=AT)
    assert result["source_ready_count"] == 0
    assert result["funds"][0]["status"] == "FROZEN_WITH_BASELINE_GAPS"
    trial.prepare_trial(tmp_path / "trial", Path(result["selection_path"]), holdings=inventory, clock=AT)
    packet = read(tmp_path / "trial/inputs/002112.json")
    assert packet["source_ready"] is False
    assert packet["periods"]["T1"]["status"] == "UNVERIFIED"


def test_calendar_failure_is_preserved_not_filled_with_weekdays(tmp_path, isolated, monkeypatch):
    inventory, _, _ = isolated

    def missing(*args):
        raise PredictionFailure("CALENDAR_POLICY_MISSING", "CALENDAR", "产品估值条款仍待核验")

    monkeypatch.setattr(command, "review_calendar", missing)
    result = command.freeze_sources(tmp_path / "sources", holdings_provider=lambda: inventory, clock=AT)
    for horizon in command.HORIZONS:
        packet = read(tmp_path / "sources/funds/002112" / horizon / "input.json")
        dates = packet["input"]["dates"]
        assert dates["endDateStatus"] == "UNVERIFIED"
        assert dates["endDate"] is None
        assert dates["missingEvidence"]["code"] == "CALENDAR_POLICY_MISSING"
    assert result["funds"][0]["periods"]["M6_V1"]["nominal_end_date"] == "2027-05-12"


def test_prepare_failure_returns_concrete_metadata_and_no_fake_selection(tmp_path, isolated, monkeypatch):
    inventory, _, _ = isolated
    monkeypatch.setattr(
        command, "prepare", lambda *a: (_ for _ in ()).throw(ValueError("COMPREHENSIVE_CALENDAR_POLICY_REQUIRED"))
    )
    result = command.freeze_sources(tmp_path / "sources", holdings_provider=lambda: inventory, clock=AT)
    assert result["frozen_count"] == 0
    assert result["status"] == "WAITING_MATERIALS"
    assert result["funds"][0]["failure"]["code"] == "COMPREHENSIVE_CALENDAR_POLICY_REQUIRED"
    assert json.loads(Path(result["selection_path"]).read_text(encoding="utf-8")) == []
    assert not (tmp_path / "sources/funds/002112/T1/input.json").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"owner_hash": "other"},
        {"policy_hash": "old"},
        {"window": {"base_nav_date": "2026-11-11", "target_nav_date": "2026-11-12"}},
        {"materials_hash": "old-material"},
    ],
)
def test_request_wrong_owner_window_policy_or_material_is_rejected(tmp_path, isolated, change):
    inventory, _, handoff = isolated
    request = {
        "format": "FUND_PREDICTION_V2_DAILY_ANALYSIS_REQUEST",
        "fund_code": "002112",
        "owner_hash": inventory["owner_hash"],
        "window": WINDOW,
        "policy_hash": digest(command.policy_v2()),
        "materials_hash": handoff["materials_hash"],
        "materials_input_hash": handoff["materials_hash"],
        **change,
    }
    path = tmp_path / "request.json"
    versioned_save(path, request)
    if "materials_hash" not in change:
        with pytest.raises(ValueError, match="V2_SOURCE_REQUEST_OWNER_POLICY_OR_WINDOW_MISMATCH"):
            command.freeze_sources(
                tmp_path / "sources", analysis_requests=[path], holdings_provider=lambda: inventory, clock=AT
            )
    else:
        result = command.freeze_sources(
            tmp_path / "sources", analysis_requests=[path], holdings_provider=lambda: inventory, clock=AT
        )
        assert result["funds"][0]["failure"]["code"] == "V2_SOURCE_REQUEST_MATERIALS_CHANGED"


def test_removed_holding_never_produces_selection(tmp_path, isolated):
    inventory, _, _ = isolated
    states = iter([inventory, {**inventory, "scope": []}])
    with pytest.raises(ValueError, match="V2_SOURCE_HOLDINGS_CHANGED_DURING_FREEZE"):
        command.freeze_sources(tmp_path / "sources", holdings_provider=lambda: next(states), clock=AT)
    assert not (tmp_path / "sources/selection.json").exists()
