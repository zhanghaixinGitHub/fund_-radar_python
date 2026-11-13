"""正式采用的身份、过期展示、追加存储和逐基金读取边界。"""

import importlib.util
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from app.services import prediction_review_v2_publication as service
from app.services import prediction_trial_v2 as trial
from app.services.direction_1d_protocol import canonical, digest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from tests.test_prediction_review_v2 import AT, NOTE, sample


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical({"hash": digest(payload), "payload": payload}), encoding="utf-8")


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    assert digest(value["payload"]) == value["hash"]
    return value["payload"]


@pytest.fixture
def strict_trial(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "SOURCE_ROOT", tmp_path)
    monkeypatch.setattr(trial, "PROJECT", tmp_path)
    monkeypatch.setattr(trial, "read", read)
    monkeypatch.setattr(trial, "save", save)
    directory = tmp_path / ".local-runs/trial"
    packet, data, _ = sample()
    material = {"fund_code": "001234", "facts": {"public": "真实公开资料"}, "number": -0.0}
    materials_path = tmp_path / ".local-runs/material.json"
    save(materials_path, material)
    data.update(
        as_of=packet["source_cutoff"],
        window={"base_nav_date": "2026-11-11", "target_nav_date": "2026-11-12"},
        time_context={"baseline_complete": True},
        materials_path=str(materials_path),
        materials_hash=digest(material),
    )
    source_path = tmp_path / ".local-runs/source/input.json"
    save(source_path, {"format": "CODEX_FUND_INPUT_V1", "input": data, "input_hash": digest(data)})
    inventory = {
        "owner_hash": digest("test-owner"),
        "scope": ["001234"],
        "profiles": [{"fund_code": "001234", "fund_type": "STOCK"}],
    }
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            [
                {
                    "fund_code": "001234",
                    "source_kind": "FROZEN",
                    "source_path": str(source_path),
                    "sample_label": "行业基金",
                    "reason": NOTE,
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest = trial.prepare_trial(directory, selection, holdings=inventory, clock=AT)
    packet = read(directory / "inputs/001234.json")
    for h, end in {"T1": "2026-11-12", "T5_V1": "2026-11-18", "T20_V1": "2026-12-09", "M6_V1": None}.items():
        packet["periods"][h].update(
            status="PENDING_OFFICIAL_CALENDAR" if end is None else "RESOLVED",
            base_date="2026-11-11",
            start_date="2026-11-12",
            end_date=end,
        )
        if end is None:
            packet["periods"][h]["nominal_end_date"] = "2027-05-11"
    save(directory / "inputs/001234.json", packet)
    manifest["selection"][0]["packet_hash"] = digest(packet)
    save(directory / "manifest.json", manifest)
    reviews = []
    for h in service.HORIZONS:
        _, _, review = sample(h)
        review["packet_hash"] = digest(packet)
        if h == "M6_V1":
            review["output_kind"] = "CONDITIONAL_OUTLOOK"
        reviews.append(review)
    reviews_path = tmp_path / "reviews.json"
    reviews_path.write_text(json.dumps(reviews, ensure_ascii=False), encoding="utf-8")
    summary = trial.evaluate_trial(directory, reviews_path, holdings_provider=lambda: inventory, clock=AT)
    monkeypatch.setattr(service, "current_holdings", lambda: deepcopy(inventory))
    return directory, inventory, summary


def test_strict_trial_revalidates_and_keeps_real_times_without_private_fields(strict_trial):
    directory, inventory, _ = strict_trial
    prepared = service.prepare_trial_release(directory, holdings=inventory, clock=AT)
    assert prepared["summary"]["records"] == 4
    view = prepared["views"]["001234"]
    assert all(r["analysisAt"] == AT.isoformat() for r in view["periods"])
    assert view["periods"][-1]["kind"] == "CONDITIONAL_OUTLOOK"
    assert view["periods"][-1]["endDate"] is None
    assert view["periods"][-1]["nominalEndDate"] == "2027-05-11"
    assert "owner_hash" not in canonical(view) and "materials_hash" not in canonical(view)
    assert "test-owner" not in canonical(view) and ".local-runs" not in canonical(view)
    assert prepared["archive"]["cases"][0]["workflow_identity"]["ex_ante_eligible"] is True
    later = service.prepare_trial_release(
        directory, holdings=inventory, clock=datetime.fromisoformat("2026-11-12T19:00:00+08:00")
    )
    assert later["contentHash"] == prepared["contentHash"]
    public = service.public_view(
        view, "2026-11-12T17:00:00+08:00", clock=datetime.fromisoformat("2026-11-12T19:00:00+08:00")
    )
    assert public["periods"][0]["kind"] == "RETROSPECTIVE"
    assert public["periods"][0]["analysisAt"] == AT.isoformat()
    assert public["periods"][0]["excludedFromAccuracy"] is True
    assert public["periods"][1]["kind"] == "DATED_ANALYSIS"
    assert view["periods"][0]["kind"] == "DATED_ANALYSIS"


def test_blocked_period_is_saved_as_public_waiting_without_hiding_other_periods(strict_trial):
    directory, inventory, summary = strict_trial
    reviews = []
    for row in summary["results"]:
        record = read(Path(row["result_path"]))
        review = record["review"]
        if row["horizon"] == "M6_V1":
            review.update(
                decision="BLOCKED",
                output_kind="DATED_ANALYSIS",
                dominant_factors=[],
                abstention_reasons=["DATA_MISSING"],
                recovery_conditions=[NOTE],
            )
        reviews.append(review)
    path = directory / "blocked-reviews.json"
    path.write_text(json.dumps(reviews, ensure_ascii=False), encoding="utf-8")
    trial.evaluate_trial(
        directory, path, holdings_provider=lambda: inventory, clock=datetime.fromisoformat("2026-11-12T11:31:00+08:00")
    )
    prepared = service.prepare_trial_release(directory, holdings=inventory, clock=AT.replace(minute=32))
    periods = prepared["views"]["001234"]["periods"]
    assert len(periods) == 4 and all(r["decision"] == "UP" for r in periods[:3])
    assert periods[-1]["decision"] == "ABSTAIN" and periods[-1]["awaitingReason"]
    assert periods[-1]["kind"] == "CONDITIONAL_OUTLOOK" and periods[-1]["excludedFromAccuracy"] is True
    stored = next(v for v in prepared["archive"]["cases"] if v["case"]["horizon"] == "M6_V1")
    assert stored["original_record"]["decision"] == "BLOCKED"


@pytest.mark.parametrize("problem", ["owner", "holding", "result", "review", "summary"])
def test_changed_identity_or_evaluation_is_rejected(strict_trial, problem):
    directory, inventory, summary = strict_trial
    if problem == "owner":
        inventory["owner_hash"] = "someone-else"
    elif problem == "holding":
        inventory["scope"] = []
    elif problem == "result":
        path = Path(summary["results"][0]["result_path"])
        result = read(path)
        result["decision"] = "DOWN"
        save(path, result)
    elif problem == "review":
        result = read(Path(summary["results"][0]["result_path"]))
        save(directory / "reviews" / (result["review_hash"] + ".json"), {})
    else:
        path = next((directory / "summaries").glob("*.json"))
        value = read(path)
        value["results"].pop()
        save(path, value)
    with pytest.raises(ValueError, match="V2_PUBLICATION"):
        service.prepare_trial_release(directory, holdings=inventory, clock=AT)


@pytest.fixture
def release_db(monkeypatch):
    original = service.get_engine()
    if original.url.host not in {"localhost", "127.0.0.1", "::1"}:
        pytest.skip("只用本机隔离schema")
    schema = "review_v2_test_" + uuid4().hex
    with original.begin() as connection:
        connection.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_engine(original.url, connect_args={"options": f"-c search_path={schema}"}, hide_parameters=True)
    path = Path(__file__).parents[1] / "alembic/versions/20261112_44_prediction_review_v2_release.py"
    spec = importlib.util.spec_from_file_location("review_v2_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    try:
        with engine.begin() as connection:
            monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(connection)))
            migration.upgrade()
        monkeypatch.setattr(service, "get_engine", lambda: engine)
        from app.services import prediction_review_v2_requests

        monkeypatch.setattr(prediction_review_v2_requests, "get_engine", lambda: engine)
        monkeypatch.setattr(service, "storage_check", lambda _: {"count": 0})
        yield engine
    finally:
        engine.dispose()
        with original.begin() as connection:
            connection.execute(text(f"DROP SCHEMA {schema} CASCADE"))


def test_append_only_idempotence_and_per_fund_latest_keeps_other_funds(strict_trial, release_db):
    directory, inventory, _ = strict_trial
    assert service.adoption_status() is None
    prepared = service.prepare_trial_release(directory, holdings=inventory, clock=AT)
    first = service._publish(prepared, adopt=True)
    repeated = service._publish(prepared, adopt=True)
    assert first["releaseId"] == repeated["releaseId"] and repeated["reused"] is True
    assert service.adoption_status()["releaseId"] == first["releaseId"]
    assert service.current_review("001234")["periods"][0]["kind"] == "RETROSPECTIVE"
    # 模拟下一轮只采用另一只基金；原基金仍能找到自己的最新版本。
    other = deepcopy(prepared)
    other["views"] = {"009999": {**other["views"]["001234"], "fundCode": "009999"}}
    other["archive"]["owner_audit"]["current_fund_codes"].append("009999")
    inventory["scope"].append("009999")
    other["contentHash"] = digest(other["archive"])
    second = service._publish(other, adopt=True)
    assert second["releaseId"] != first["releaseId"]
    assert service.current_review("001234")["fundCode"] == "001234"
    assert service.current_review("009999")["fundCode"] == "009999"
    assert service.current_review("008888") is None
    internal = service.daily_records("test-owner", ["001234"], "2026-11-12")
    assert len(internal) == 4
    assert all(r["owner_verified"] and r["snapshot_verified"] for r in internal)
    assert all(r["materials_input_hash"] and r["materials_hash"] for r in internal)
    assert all(not r["ex_ante_eligible"] for r in internal)
    assert service.daily_records("another-owner", ["001234"], "2026-11-12") == []
    with release_db.begin() as connection, pytest.raises(Exception, match="append only"):
        connection.execute(text("UPDATE prediction_review_v2_release SET version='changed'"))


def test_request_completion_runs_only_after_successful_commit(strict_trial, release_db, monkeypatch):
    from app.services import prediction_review_v2_requests

    directory, inventory, _ = strict_trial
    prepared = service.prepare_trial_release(directory, holdings=inventory, clock=AT)
    called = []

    def completed(packet, release_id, adopted_at):
        # 使用另一连接看到这条结果，证明正式事务已经提交。
        with release_db.connect() as connection:
            row = (
                connection.execute(
                    text("SELECT archive,adopted_at FROM prediction_review_v2_release WHERE release_id=:id"),
                    {"id": release_id},
                )
                .mappings()
                .one()
            )
        assert digest(row["archive"]) == packet["contentHash"]
        assert row["adopted_at"] == adopted_at
        called.append(str(release_id))
        return 0

    monkeypatch.setattr(prediction_review_v2_requests, "complete_from_release", completed, raising=False)
    assert service._publish(prepared, adopt=False)["status"] == "PREVIEW"
    assert called == []
    invalid = {**prepared, "contentHash": "f" * 64}
    with pytest.raises(ValueError, match="DATABASE_READBACK_MISMATCH"):
        service._publish(invalid, adopt=True)
    assert called == []
    result = service._publish(prepared, adopt=True)
    assert called == [result["releaseId"]]


def test_routes_are_service_protected_and_use_new_queue(monkeypatch):
    from app.api.dependencies import require_service_token
    from app.api.routes import prediction_review_v2 as routes

    app = FastAPI()
    app.include_router(routes.router, prefix="/internal/v1/multi-predictions")
    monkeypatch.setattr(routes, "current_review", lambda code: {"fundCode": code})
    monkeypatch.setitem(
        sys.modules,
        "app.services.prediction_review_v2_requests",
        SimpleNamespace(request_review=lambda code: {"status": "WAITING_FOR_NAV", "fundCode": code}),
    )
    with TestClient(app) as client:
        assert client.get("/internal/v1/multi-predictions/funds/001234/review-v2").status_code == 403
        app.dependency_overrides[require_service_token] = lambda: None
        assert client.get("/internal/v1/multi-predictions/funds/001234/review-v2").json() == {"fundCode": "001234"}
        assert client.get("/internal/v1/multi-predictions/funds/invalid/review-v2").status_code == 422
        queued = client.post("/internal/v1/multi-predictions/funds/001234/review-v2/generate")
        assert queued.json() == {"status": "WAITING_FOR_NAV", "fundCode": "001234"}
        from app.services.prediction_contract import PredictionFailure

        def rejected(code):
            raise PredictionFailure("FUND_OUTSIDE_HOLDINGS", "REQUEST", "本基金不在当前持仓中。")

        monkeypatch.setattr(sys.modules["app.services.prediction_review_v2_requests"], "request_review", rejected)
        failed = client.post("/internal/v1/multi-predictions/funds/001234/review-v2/generate")
        assert failed.status_code == 422
        assert failed.json()["detail"]["code"] == "FUND_OUTSIDE_HOLDINGS"


@pytest.mark.parametrize("duplicate", [False, True])
def test_research_adoption_requires_exactly_36_unique_cases(tmp_path, monkeypatch, duplicate):
    rows = [{"fund_code": f"{code:06d}", "horizon": h} for code in range(9) for h in service.HORIZONS]
    if duplicate:
        rows[-1] = rows[0]
    else:
        rows.pop()
    parent, original = {}, {"results": []}
    manifest = {
        "parent_ledger_path": str(tmp_path / "parent"),
        "original_summary_path": str(tmp_path / "original"),
        "parent_ledger_hash": digest(parent),
        "original_summary_hash": digest(original),
    }
    ledger = {
        "parent_manifest_hash": digest(manifest),
        "latest_36": rows,
        "verification": {
            "old_36_results_unchanged": True,
            "previous_23_ledger_unchanged": True,
            "exact_reading_spans_checked": 1,
            "passed_related_tests": 1,
            "storage_check": {"count": 0},
        },
    }
    values = {"ledger": ledger, "case-manifest.json": manifest, "parent": parent, "original": original}
    monkeypatch.setattr(service, "_load", lambda path, _: values[Path(path).name])
    with pytest.raises(ValueError, match="36_UNIQUE_CASES_REQUIRED"):
        service.prepare_release(tmp_path / "ledger")
