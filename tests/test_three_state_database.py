"""本机隔离库的三态切换、不可变输入和旧规则核验；不写真实基金记录。"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from app.repositories import prediction_store as store
from app.services import auto_model_summary
from app.services import prediction_generation as generation
from app.services import prediction_models as models
from app.services.prediction_contract import PredictionFailure, legacy_policy, policy_scope, prediction_policy
from app.services.prediction_direction import direction_fields
from app.services.prediction_task_inputs import task_input
from sqlalchemy import text
from tests.test_prediction_contract import calendar
from tests.test_prediction_database import database  # noqa: F401


def test_same_period_switch_is_immutable_and_outcome_uses_saved_threshold(database, monkeypatch):  # noqa: F811
    for module in (store, generation, auto_model_summary):
        monkeypatch.setattr(module, "get_engine", lambda: database)
    now = datetime.now(UTC)
    model = models.freeze_routes()[models.route_key("T5_V1")]["model_id"]
    base = dict(
        predictionId=str(uuid4()),
        fundCode="123456",
        horizonId="T5_V1",
        mode="LIVE",
        role="PRIMARY",
        modelId=model,
        activationRevision=1,
        generatedAt="2026-09-15T01:00:00+00:00",
        startDate="2026-09-15",
        endDate="2026-09-16",
        direction="FLAT",
        targetDefinitionId=prediction_policy()["target_definition_id"],
        predictionPolicySnapshot=deepcopy(prediction_policy()),
        **direction_fields("T5_V1"),
    )
    old = base | dict(
        predictionId=str(uuid4()), direction="UP", targetDefinitionId=legacy_policy()["target_definition_id"]
    )
    for key in (
        "predictionPolicySnapshot",
        "directionPolicyId",
        "directionPolicyHash",
        "directionPolicySnapshot",
        "flatThreshold",
    ):
        old.pop(key, None)
    with database.begin() as c:
        store.save_prediction(c, old, generation.prediction_period_key(old))

    def save(_):
        with database.begin() as c:
            return store.save_prediction(
                c, base | {"predictionId": str(uuid4())}, generation.prediction_period_key(base)
            )

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(save, range(3)))
    assert sum(created for _, created in results) == 1
    primary = results[0][0]
    assert all(value == primary for value, _ in results)
    shadow = base | dict(predictionId=str(uuid4()), role="SHADOW", direction="DOWN")
    with database.begin() as c:
        store.save_prediction(c, shadow, generation.prediction_period_key(base) + ":SHADOW:test")
    assert [p["predictionId"] for p in generation.current_predictions("123456")["predictions"]] == [
        primary["predictionId"]
    ]
    values = {date(2026, 9, 15): Decimal("1"), date(2026, 9, 16): Decimal("1.002")}
    monkeypatch.setattr(
        generation,
        "read_fund_data",
        lambda *a, **kw: dict(
            calendar=calendar(),
            navs=[dict(nav_date=d, unit_nav=v, updated_at=now) for d, v in values.items()],
            dividends=[],
            dividendWatermark=now,
        ),
    )
    changed = deepcopy(prediction_policy())
    changed["direction"]["thresholds"]["T5_V1"] = "0.001"
    with policy_scope(changed):
        generation.verify_outcomes()
        assert not generation.current_predictions("123456")["predictions"]
    result = {row["payload"]["predictionId"]: row for row in store.history("123456")}
    assert result[primary["predictionId"]]["outcomes"][0]["actualDirection"] == "FLAT"
    assert result[old["predictionId"]]["outcomes"][0]["actualDirection"] == "UP"
    assert result[old["predictionId"]]["payload"] == old
    generation.verify_outcomes()
    effects = auto_model_summary.effect_summary(["123456"])
    assert effects["fundCount"] == 1 and sum(h["records"] for h in effects["horizons"]) == 2
    tri = next(h for h in effects["horizons"] if h["direction_policy_hash"])
    assert tri["flat_actual"] == tri["flat_correct"] == tri["matured"] == 1


def test_frozen_live_success_failure_and_retry_policy(database, monkeypatch):  # noqa: F811
    monkeypatch.setattr(generation, "get_engine", lambda: database)
    monkeypatch.setattr(generation._executor, "submit", lambda *args: None)
    task = generation.create_task(["123456"], request_key="frozen-source")
    saved = task_input("LIVE", task["taskId"], "123456", lambda: {"amount": Decimal("1.001"), "day": date(2026, 9, 21)})
    assert task_input("LIVE", task["taskId"], "123456", lambda: pytest.fail("恢复不应重读来源")) == saved

    def missing():
        raise PredictionFailure("SOURCE_MISSING", "SOURCE", "原缺口")

    failure = task_input("LIVE", task["taskId"], "654321", missing)
    assert task_input("LIVE", task["taskId"], "654321", lambda: {}) == failure
    with database.begin() as c:
        c.execute(
            text("UPDATE prediction_generation_item SET status='FAILED' WHERE task_id=:id"), {"id": task["taskId"]}
        )
    old_policy = deepcopy(prediction_policy())
    changed = deepcopy(old_policy)
    changed["direction"]["thresholds"]["T5_V1"] = "0.2"
    with policy_scope(changed):
        retried = generation.retry_failed(task["taskId"])
    with database.connect() as c:
        policy = c.execute(
            text("SELECT payload->'predictionPolicy' FROM prediction_generation_task WHERE task_id=:id"),
            {"id": retried["taskId"]},
        ).scalar()
    assert policy == old_policy and retried["taskId"] != task["taskId"]
    with database.begin() as c, pytest.raises(Exception, match="append only"):
        c.execute(text("UPDATE prediction_task_input SET payload='{}'"))
