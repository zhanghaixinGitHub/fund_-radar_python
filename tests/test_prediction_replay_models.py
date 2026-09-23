"""模型对照的时间、身份与完整性边界；受控测试不代替真实数据库回放。"""

from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import pytest
from app.services import prediction_replay_inputs as inputs
from app.services import prediction_replay_models as models
from app.services.prediction_contract import PredictionFailure, fingerprint
from app.services.prediction_models import FEATURES, baseline_package, route_key
from tests.test_prediction_models import package


@pytest.fixture
def registry(monkeypatch):
    routes, packages = {}, {}
    for horizon in ("T5_V1", "T20_V1", "M6_V1"):
        base = baseline_package(horizon)
        trained = package() | {"horizonId": horizon, "recipeVersion": "TOTAL_RETURN_LOGISTIC_V1"}
        for key, manifest in ((horizon + "base", base), (horizon + "trained", trained)):
            packages[key] = {"modelId": key, "modelHash": fingerprint(manifest), "manifest": manifest}
        routes[route_key(horizon)] = {"model_id": horizon + "base", "shadow_ids": [horizon + "trained"], "revision": 2}
    monkeypatch.setattr(models, "freeze_routes", lambda: routes)
    monkeypatch.setattr(models, "load_model", lambda key: packages[key])
    return routes, packages


def test_full_bundles_keep_actual_fingerprints_and_current_is_not_duplicated(registry):
    bundles, excluded = models.replay_model_bundles(date(2026, 1, 1))
    assert [bundle["role"] for bundle in bundles] == ["CURRENT", "CANDIDATE"]
    assert not excluded
    for bundle in bundles:
        assert len(bundle["modelRefs"]) == 3
        for ref in bundle["modelRefs"]:
            assert ref["modelHash"] == fingerprint(registry[1][ref["modelId"]]["manifest"])


def test_future_training_cannot_join_and_missing_period_is_not_filled_with_baseline(registry):
    registry[1]["M6_V1trained"]["manifest"]["labelEndMax"] = "2026-01-01T00:00:00+08:00"
    bundles, excluded = models.replay_model_bundles(date(2026, 1, 1))
    assert len(bundles) == 1 and bundles[0]["role"] == "CURRENT"
    assert any("标签" in item["reason"] for item in excluded)
    assert any("完整" in item["reason"] for item in excluded)


def test_current_components_do_not_invent_additional_candidates(registry):
    for route in registry[0].values():
        route["shadow_ids"] = []
    registry[0][route_key("T5_V1")]["model_id"] = "T5_V1trained"
    bundles, excluded = models.replay_model_bundles(date(2026, 1, 1))
    assert len(bundles) == 1 and bundles[0]["role"] == "CURRENT"
    assert not excluded


def test_failed_package_does_not_prevent_other_complete_bundle(registry, monkeypatch):
    def load(key):
        if key == "T5_V1base":
            raise PredictionFailure("MODEL_HASH_MISMATCH", "MODEL_LOAD", "模型文件与登记指纹不一致")
        return registry[1][key]

    monkeypatch.setattr(models, "load_model", load)
    bundles, excluded = models.replay_model_bundles(date(2026, 1, 1))
    assert len(bundles) == 1 and bundles[0]["role"] == "CANDIDATE"
    assert any("指纹" in item["reason"] for item in excluded)


def replay_fixture(monkeypatch):
    days = [date(2026, 1, 5), date(2026, 1, 6)]
    data = {
        "fund": {"fund_master_id": "fixture-family"},
        "calendar": SimpleNamespace(sessions=days, source_hash="calendar"),
        "navs": [{"nav_date": d, "unit_nav": Decimal(1)} for d in days],
        "dividends": [],
    }
    monkeypatch.setattr(inputs, "read_fund_data", lambda *args, **kwargs: data)
    monkeypatch.setattr(inputs, "cash_events", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        inputs,
        "build_features",
        lambda *args, **kwargs: {
            "features": dict.fromkeys(FEATURES, 0.0) | {"momentum": 1, "trendRiskFactor": 0, "currentDrawdown": 0},
            "dataAsOf": "2026-01-02",
        },
    )


def test_each_bundle_really_infers_its_own_model_on_identical_days(registry, monkeypatch):
    replay_fixture(monkeypatch)
    result = inputs.replay_inputs("006730", date(2026, 1, 1), date(2026, 1, 7))
    current, candidate = result["modelComparisons"]
    assert len(current["frames"]) == len(candidate["frames"]) == 2
    assert {s["direction"] for s in current["frames"][0]["input"]["predictions"]} == {"UP"}
    assert {s["direction"] for s in candidate["frames"][0]["input"]["predictions"]} == {"NON_UP"}
    assert all(s["modelId"].endswith("trained") for s in candidate["frames"][0]["input"]["predictions"])


def test_one_inference_failure_excludes_whole_bundle_without_cherry_picking_dates(registry, monkeypatch):
    replay_fixture(monkeypatch)
    original = inputs.infer_package

    def infer(manifest, features):
        if manifest["adapter"] == "LOGISTIC_STANDARDIZED_V1":
            raise PredictionFailure("INFERENCE_ERROR", "INFERENCE", "候选推理失败")
        return original(manifest, features)

    monkeypatch.setattr(inputs, "infer_package", infer)
    result = inputs.replay_inputs("006730", date(2026, 1, 1), date(2026, 1, 7))
    assert len(result["modelComparisons"]) == 1
    assert result["modelComparisons"][0]["role"] == "CURRENT"
    assert any("候选推理失败" in item["reason"] for item in result["excludedModels"])
