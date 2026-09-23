"""真实PostgreSQL隔离schema检验租约、去重、CAS和冻结恢复；收益为受控夹具。"""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from app.services import auto_model_selection as auto
from app.services import auto_model_store as store
from app.services import prediction_models as models
from app.services.auto_model_contract import auto_policy
from app.services.prediction_contract import PredictionFailure, fingerprint
from sqlalchemy import text
from tests.test_prediction_database import database  # noqa: F401
from tests.test_prediction_models import three_package as package


@pytest.fixture
def auto_database(database, monkeypatch):  # noqa: F811
    from app.db import session

    monkeypatch.setattr(session, "get_engine", lambda: database)
    monkeypatch.setattr(auto, "get_engine", lambda: database)
    monkeypatch.setattr(store, "get_engine", lambda: database)
    monkeypatch.setattr(auto, "data_version", lambda codes: {"nav": [{"n": 100}], "outcomeWatermark": "none"})
    monkeypatch.setattr(auto._executor, "submit", lambda *args: None)
    return database


def test_cancel_and_resume_preserves_original_spec(auto_database):
    from uuid import UUID

    from app.api.routes.multi_predictions import auto_cancel, auto_resume

    cycle = create_cycle()["cycleId"]
    original = store.cycle_state(cycle)["spec"]
    assert auto_cancel(UUID(cycle))["status"] == "CANCELLED"
    resumed = auto_resume(UUID(cycle))
    assert resumed["status"] == "INTERRUPTED" and not resumed["cancel_requested"]
    assert resumed["spec"] == original


def test_waiting_data_and_mature_watermark_no_repeated_research(auto_database, monkeypatch):
    monkeypatch.setattr(auto, "data_version", lambda codes: {"nav": [], "outcomeWatermark": "none"})
    wait = create_cycle()
    assert wait["status"] == "WAITING_DATA" and create_cycle()["cycleId"] == wait["cycleId"]
    monkeypatch.setattr(auto, "data_version", lambda codes: {"nav": [{"n": 100}], "outcomeWatermark": "none"})
    queued = create_cycle()
    assert queued["status"] == "QUEUED" and queued["cycleId"] != wait["cycleId"]
    with auto_database.begin() as c:
        c.execute(
            text("UPDATE prediction_auto_cycle SET status='COMPLETED',checkpoint=:point WHERE cycle_id=:id"),
            {"id": queued["cycleId"], "point": '{"runId":"frozen-original"}'},
        )
    assert create_cycle()["cycleId"] == queued["cycleId"]
    monkeypatch.setattr(auto, "data_version", lambda codes: {"nav": [{"n": 100}], "outcomeWatermark": "new-answer"})
    evaluation = create_cycle()
    assert evaluation["trigger"] == "MATURED_OR_REVISED"
    assert store.cycle_state(evaluation["cycleId"])["spec"]["reuseRunId"] == "frozen-original"


def test_same_timestamp_cycle_pagination_does_not_drop_rows(auto_database, monkeypatch):
    from app.services import auto_model_summary as summary

    monkeypatch.setattr(summary, "get_engine", lambda: auto_database)
    cycle = create_cycle()["cycleId"]
    with auto_database.begin() as c:
        c.execute(
            text("""INSERT INTO prediction_auto_cycle(cycle_id,request_key,scope_hash,data_hash,protocol_hash,
          status,trigger_reason,spec,created_at)
          SELECT :id,:key,scope_hash,data_hash,protocol_hash,'COMPLETED',trigger_reason,
          spec,created_at FROM prediction_auto_cycle WHERE cycle_id=:old"""),
            {"id": uuid4(), "key": "z" * 64, "old": cycle},
        )
    first = summary.technical_cycles(1)["items"][0]
    second = summary.technical_cycles(1, datetime.fromisoformat(first["createdAt"]), first["cycleId"])["items"][0]
    assert first["cycleId"] != second["cycleId"]


def create_cycle():
    return auto.check_auto(["006730", "001632", "006730"], now=datetime(2026, 9, 23, 7, tzinfo=UTC))


def test_check_is_idempotent_and_scope_deduplicated(auto_database):
    first, second = create_cycle(), create_cycle()
    assert first["cycleId"] == second["cycleId"] and first["fundCount"] == 2
    with auto_database.connect() as c:
        assert c.execute(text("SELECT count(*) FROM prediction_auto_cycle")).scalar() == 1
    state = store.cycle_state(first["cycleId"])
    assert state["spec"]["shards"] == [["001632", "006730"]]
    assert state["trigger_reason"] == "INITIAL"


def test_old_implementation_requeues_only_after_lease_released(auto_database):
    first = create_cycle()["cycleId"]
    with auto_database.begin() as c:
        c.execute(
            text("""UPDATE prediction_auto_cycle SET protocol_hash=:hash,
          lease_until=clock_timestamp()+interval '1 hour' WHERE cycle_id=:id"""),
            {"id": first, "hash": "0" * 64},
        )
    assert create_cycle()["cycleId"] == first
    with auto_database.begin() as c:
        # 原请求键也来自旧协议，生产中它和新实现请求键不同。
        c.execute(
            text("UPDATE prediction_auto_cycle SET lease_until=NULL,request_key=:key WHERE cycle_id=:id"),
            {"id": first, "key": "old-implementation"},
        )
    second = create_cycle()["cycleId"]
    assert second != first and store.cycle_state(first)["status"] == "FAILED"


def test_cancelled_worker_cannot_publish_late_checkpoint(auto_database):
    from uuid import UUID

    from app.api.routes.multi_predictions import auto_cancel

    cycle, owner = create_cycle()["cycleId"], uuid4()
    with auto_database.begin() as c:
        c.execute(
            text("UPDATE prediction_auto_cycle SET lease_owner=:owner WHERE cycle_id=:id"),
            {"id": cycle, "owner": owner},
        )
    auto_cancel(UUID(cycle))
    with pytest.raises(PredictionFailure, match="租约"):
        store.transition(cycle, owner, "REPLAYING")
    assert store.cycle_state(cycle)["status"] == "CANCELLED"


def test_training_checkpoint_and_deciding_crash_resume(auto_database, monkeypatch):
    cycle = create_cycle()["cycleId"]
    original = store.cycle_state(cycle)["spec"]
    with auto_database.begin() as c:
        c.execute(
            text("""UPDATE prediction_auto_cycle SET status='INTERRUPTED',
          checkpoint='{"runId":"saved-run"}'::jsonb WHERE cycle_id=:id"""),
            {"id": cycle},
        )
    monkeypatch.setattr(auto, "research_status", lambda key: {"run_id": key, "status": "SUCCEEDED", "result": {}})
    monkeypatch.setattr(auto, "create_research", lambda *a, **kw: pytest.fail("不能重复创建研究"))
    monkeypatch.setattr(auto, "run_research", lambda *a, **kw: pytest.fail("不能重训已完成阶段"))
    monkeypatch.setattr(auto, "build_bundles", lambda *a: [{"id": "CURRENT", "modelRefs": []}])
    monkeypatch.setattr(auto, "mature_evaluation", lambda *a: {"comparisons": {}})
    monkeypatch.setattr(auto, "notify_replay_ready", lambda *a: None)
    auto.run_auto(cycle)
    resumed = store.cycle_state(cycle)
    assert resumed["status"] == "REPLAYING" and resumed["spec"] == original
    assert resumed["checkpoint"]["runId"] == "saved-run"
    for code in original["fundCodes"]:
        store.frozen_input(cycle, "result:" + code, lambda: {"status": "FAILED", "error": "fixture"})
    with auto_database.begin() as c:
        c.execute(
            text("UPDATE prediction_auto_cycle SET status='DECIDING',lease_until=NULL WHERE cycle_id=:id"),
            {"id": cycle},
        )
    claim = auto.claim_replay()
    assert len(claim["fundCodes"]) == 1
    result = auto.save_replay(cycle, claim["leaseOwner"], claim["fundCodes"][0], {"status": "FAILED"})
    assert result["status"] == "COMPLETED" and result["decision"] == "KEEP_CURRENT"


def test_frozen_checkpoint_never_reloads_changed_data(auto_database):
    cycle = create_cycle()["cycleId"]
    first = store.frozen_input(cycle, "fund:006730", lambda: {"version": 1, "asof": datetime.now(UTC)})
    recovered = store.frozen_input(cycle, "fund:006730", lambda: pytest.fail("必须复用冻结输入"))
    assert first == recovered


def test_old_owner_cannot_update_new_lease(auto_database):
    cycle = create_cycle()["cycleId"]
    owner = uuid4()
    with auto_database.begin() as c:
        c.execute(
            text("UPDATE prediction_auto_cycle SET lease_owner=:owner WHERE cycle_id=:id"),
            {"owner": owner, "id": cycle},
        )
    with pytest.raises(PredictionFailure, match="租约已被接管"):
        store.transition(cycle, uuid4(), "COMPLETED")
    store.transition(cycle, owner, "TRAINING", checkpoint={"runId": "fixture"})
    assert store.cycle_state(cycle)["checkpoint"] == {"runId": "fixture"}


def refs(routes):
    return [
        {
            "horizonId": horizon,
            "modelId": routes[models.route_key(horizon)]["model_id"],
            "modelHash": models.load_model(routes[models.route_key(horizon)]["model_id"])["modelHash"],
        }
        for horizon in ["T5_V1", "T20_V1", "M6_V1"]
    ]


def test_atomic_release_conflict_rolls_back_all_routes(auto_database):
    cycle = create_cycle()["cycleId"]
    before = models.freeze_routes()
    replacement = models.register_model(package())
    bundle = {"modelRefs": refs(before)}
    bundle["modelRefs"][0].update(modelId=replacement["modelId"], modelHash=replacement["modelHash"])
    models.activate(before[models.route_key("T20_V1")]["model_id"], reason={"fixture": True}, expected_revision=1)
    with pytest.raises(PredictionFailure, match="路由已被其他任务更新"):
        store.publish_bundle(
            cycle,
            bundle,
            before,
            {
                "policyHash": fingerprint(auto_policy()),
                "executionPolicy": auto_policy()["executionPolicy"],
                "decision": "ACTIVATE",
            },
        )
    after = models.freeze_routes()
    assert after[models.route_key("T5_V1")]["model_id"] == before[models.route_key("T5_V1")]["model_id"]
    with auto_database.connect() as c:
        assert c.execute(text("SELECT count(*) FROM prediction_model_release")).scalar() == 0


def test_complete_release_is_idempotent_but_not_actual_use(auto_database):
    cycle = create_cycle()["cycleId"]
    before = models.freeze_routes()
    reason = {
        "policyHash": fingerprint(auto_policy()),
        "executionPolicy": auto_policy()["executionPolicy"],
        "decision": "ACTIVATE",
    }
    bundle = {"modelRefs": refs(before)}
    first = store.publish_bundle(cycle, bundle, before, reason)
    second = store.publish_bundle(cycle, bundle, before, reason)
    assert first["release_id"] == second["release_id"]
    assert {route["release_id"] for route in models.freeze_routes().values()} == {first["release_id"]}
    assert {route["revision"] for route in models.freeze_routes().values()} == {2}
    with pytest.raises(PredictionFailure, match="尚未取得完整周期"):
        store.actual_use_receipt(
            first["release_id"], {"predictionIds": [uuid4()], "apiReadback": True, "adviceReferenced": True}
        )


def test_transient_retry_is_bounded_and_keeps_checkpoint(auto_database):
    cycle = create_cycle()["cycleId"]
    owner = uuid4()
    with auto_database.begin() as c:
        c.execute(
            text("UPDATE prediction_auto_cycle SET lease_owner=:owner WHERE cycle_id=:id"),
            {"owner": owner, "id": cycle},
        )
    for expected in range(1, 4):
        auto.retry_transient(cycle, owner, {"frozen": True}, {"code": "DB_CONNECTION"})
        state = store.cycle_state(cycle)
        assert state["retries"] == expected and state["status"] == "INTERRUPTED" and state["next_attempt_at"]
    auto.retry_transient(cycle, owner, {"frozen": True}, {"code": "DB_CONNECTION"})
    assert store.cycle_state(cycle)["status"] == "FAILED"


def test_replay_tie_negative_improvement_drawdown_and_failures():
    bundles = [{"id": "CURRENT"}, {"id": "CANDIDATE", "predictionEligible": True}]
    state = {"spec": {"fundCodes": ["001632", "006730"], "policy": auto_policy()}, "checkpoint": {"bundles": bundles}}

    def metric(value, drawdown=0.1):
        return dict(netReturn=value, maxDrawdown=drawdown, fees=10, turnover=1, cashOnlySessions=3)

    def result(old, new, drawdown=0.1):
        return {
            code: {
                "status": "SUCCEEDED",
                "family": "shared",
                "comparisons": {"CURRENT": metric(old), "CANDIDATE": metric(new, drawdown)},
            }
            for code in state["spec"]["fundCodes"]
        }

    assert auto.decide_bundle(state, result(0.1, 0.1))["decision"] == "KEEP_CURRENT"
    assert auto.decide_bundle(state, result(-0.1, -0.05))["decision"] == "ACTIVATE"
    assert auto.decide_bundle(state, result(0.1, 0.2, 0.13))["decision"] == "KEEP_CURRENT"
    incomplete = result(0.1, 0.2)
    del incomplete["001632"]["comparisons"]["CANDIDATE"]
    assert auto.decide_bundle(state, incomplete)["decision"] == "KEEP_CURRENT"


def test_effect_counts_unique_funds_latest_answer_and_pending_separately(auto_database, monkeypatch):
    from datetime import date, timedelta

    from app.repositories import prediction_store
    from app.services import auto_model_summary

    monkeypatch.setattr(prediction_store, "get_engine", lambda: auto_database)
    monkeypatch.setattr(auto_model_summary, "get_engine", lambda: auto_database)
    model = models.freeze_routes()[models.route_key("T5_V1")]["model_id"]
    ids = []
    with auto_database.begin() as c:
        for number, horizon in enumerate(["T5_V1", "T20_V1", "M6_V1", "T5_V1"]):
            identity = uuid4()
            ids.append(identity)
            prediction_store.save_prediction(
                c,
                {
                    "predictionId": str(identity),
                    "fundCode": "001632",
                    "horizonId": horizon,
                    "mode": "LIVE",
                    "modelId": model,
                    "activationRevision": 1,
                    "direction": "UP",
                    "role": "PRIMARY",
                    "generatedAt": "2026-09-15T01:00:00+00:00",
                    "startDate": "2026-09-15",
                    "targetDefinitionId": "TOTAL_RETURN",
                    "endDate": str(date.today() + timedelta(days=90 if number == 2 else -1)),
                },
                "effect-" + str(number),
            )
        for correct in (True, False):
            c.execute(
                text("""INSERT INTO prediction_outcome(outcome_id,prediction_id,content_hash,payload)
              VALUES(:id,:p,:hash,CAST(:payload AS jsonb))"""),
                {
                    "id": uuid4(),
                    "p": ids[0],
                    "hash": fingerprint(correct),
                    "payload": prediction_store.encode({"correct": correct}),
                },
            )
        c.execute(
            text("INSERT INTO prediction_check_state(prediction_id,status,payload) VALUES(:id,'FAILED','{}')"),
            {"id": ids[3]},
        )
    result = auto_model_summary.effect_summary(["001632", "001632"])
    rows = {v["horizon_id"]: v for v in result["horizons"]}
    assert result["fundCount"] == result["followedFundCount"] == 1
    assert rows["T5_V1"]["matured"] == 1 and rows["T5_V1"]["correct"] == 0
    assert rows["T5_V1"]["check_failed"] == 1
    assert rows["T20_V1"]["pending_answers"] == 1 and rows["T20_V1"]["unmatured"] == 0
    assert rows["M6_V1"]["unmatured"] == 1
    assert auto_model_summary.effect_summary(["999999"])["fundCount"] == 0


def test_internal_route_token_origin_schema_and_domain_errors(auto_database, monkeypatch):
    from types import SimpleNamespace

    from app.api import dependencies
    from app.api.routes.multi_predictions import router
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from pydantic import SecretStr

    monkeypatch.setattr(dependencies, "get_settings", lambda: SimpleNamespace(ai_service_token=SecretStr("test-only")))
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    header = {"X-Service-Token": "test-only"}
    assert client.get("/auto/cycles").status_code == 403
    assert client.get("/auto/cycles", headers=header | {"Origin": "http://localhost"}).status_code == 403
    assert client.post("/auto/check", headers=header, json={"fundCodes": [], "userId": "other"}).status_code == 422
    missing = client.get("/auto/cycles/" + str(uuid4()), headers=header)
    assert missing.status_code == 404 and missing.json()["detail"]["code"] == "AUTO_CYCLE_NOT_FOUND"


def test_release_inference_saved_identity_and_verified_receipt(auto_database, monkeypatch):
    """时间和特征是受控夹具；模型文件、适配器、不可变保存与回执校验走真实实现。"""
    from app.repositories import prediction_store
    from app.services.prediction_contract import prediction_policy
    from app.services.prediction_generation import prediction_payload
    from tests.test_prediction_contract import calendar

    monkeypatch.setattr(prediction_store, "get_engine", lambda: auto_database)
    cycle = create_cycle()["cycleId"]
    before = models.freeze_routes()
    release = store.publish_bundle(
        cycle,
        {"modelRefs": refs(before)},
        before,
        {
            "decision": "ACTIVATE",
            "policyHash": fingerprint(auto_policy()),
            "executionPolicy": auto_policy()["executionPolicy"],
        },
    )
    routes = models.freeze_routes()
    instant = datetime.fromisoformat("2026-09-22T10:00:00+08:00")
    features = {
        "features": {"momentum": 0.1, "actualLookbackReturns": 60},
        "knowledgeCutoff": instant.isoformat(),
        "dataAsOf": "2026-09-21",
        "featureHash": "fixture-feature",
        "missingOptionalFactors": [],
        "calendarAssumption": "受控日历",
    }
    ids = []
    with auto_database.begin() as c:
        for horizon in prediction_policy()["horizons"]:
            payload = prediction_payload(
                {"calendar": calendar(), "fund": {"fund_code": "001632"}},
                features,
                horizon,
                instant,
                routes[models.route_key(horizon["horizon_id"])],
            )
            assert payload["releaseId"] == str(release["release_id"]) and payload["direction"] == "UP"
            saved, created = prediction_store.save_prediction(c, payload, "release:" + horizon["horizon_id"])
            assert created and saved["modelHash"] == payload["modelHash"]
            ids.append(saved["predictionId"])
    assert len(prediction_store.history("001632")) == 3
    receipt = {"predictionIds": ids, "apiReadback": True, "adviceReferenced": False, "referenceHash": fingerprint(ids)}
    with pytest.raises(PredictionFailure, match="尚未取得完整周期"):
        store.actual_use_receipt(release["release_id"], receipt)
    receipt["adviceReferenced"] = True
    assert store.actual_use_receipt(release["release_id"], receipt)["status"] == "ACTUAL_USE_CONFIRMED"


@pytest.mark.parametrize("failure_code", ["MODEL_PACKAGE_INCOMPATIBLE", "INFERENCE_ERROR"])
def test_technical_fault_quarantine_never_claims_requested_release(auto_database, monkeypatch, failure_code):
    before = models.freeze_routes()
    key = models.route_key("T5_V1")
    registered = models.register_model(package())
    models.activate(registered["modelId"], reason={"fixture": True}, expected_revision=1)
    route = models.freeze_routes()[key] | {"release_id": uuid4()}
    original = models.infer_package

    def faulty(manifest, values):
        if manifest["adapter"] not in models.BASELINE_ADAPTERS:
            raise PredictionFailure(failure_code, "INFERENCE", "受控包故障", retryable=False)
        return original(manifest, values)

    monkeypatch.setattr(models, "infer_package", faulty)
    features = dict.fromkeys(models.FEATURES, 0.1) | {"momentum": 0.1}
    answer = models.infer_route(route, features, "T5_V1")
    assert answer["modelId"] == before[key]["model_id"] and answer["releaseId"] is None
    assert answer["requestedReleaseId"] == str(route["release_id"])
    again = models.infer_route(route, features, "T5_V1")
    assert again["fallbackReason"][0]["code"] == "MODEL_QUARANTINED"
    with pytest.raises(PredictionFailure):
        models.infer_route(route | {"strictModel": True}, features, "T5_V1")
    # 隔离期满且原包恢复后允许重试，不能在隔离期内每个基金反复触碰坏包。
    monkeypatch.setattr(models, "infer_package", original)
    with auto_database.begin() as c:
        c.execute(text("UPDATE prediction_model_quarantine SET retry_after=clock_timestamp()-interval '1 second'"))
    restored = models.infer_route(route, features, "T5_V1")
    assert restored["modelId"] == registered["modelId"] and restored["fallbackReason"] is None
