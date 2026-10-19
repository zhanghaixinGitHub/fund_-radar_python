"""第二版质量计算的隔离算例，不使用真实评级，不把测试校准包用作生产依据。"""

from copy import deepcopy
from decimal import Decimal

import pytest
from app.services.fund_rating_inputs import digest
from app.services.fund_rating_quality_v2 import (
    HIGHER_BETTER,
    KINDS,
    assess,
    calculate_candidate,
    candidate_policy,
    metric_names,
)


def arithmetic(packet, calibration):
    """测试样本只检验算术，不通过原件、校准或独立验证门禁。"""
    return calculate_candidate(
        {k: v["value"] for k, v in packet["metrics"].items()},
        candidate_policy(packet["kind"]),
        calibration["anchors"],
        packet.get("material_event_pending", False),
    )


def sample(kind="ACTIVE_EQUITY", value=95):
    config = candidate_policy(kind)
    calibration = {
        "version": config["version"],
        "kind": kind,
        "policy_hash": digest(config),
        "cutoff": "2026-08-31",
        "frozen_at": "2026-09-02T10:00:00+08:00",
        "product_ids": [f"TEST_PRODUCT_{i}" for i in range(30)],
        "validation_product_ids": ["TEST_HOLDOUT"],
        "source_hashes": ["TEST_ONLY"],
        "independent_review": "TEST_FIXTURE_NOT_PRODUCTION",
        "anchors": {
            m: [[0, 0], [100, 100]] if m in HIGHER_BETTER else [[0, 100], [100, 0]] for m in metric_names(config)
        },
    }
    calibration["anchors"].update(
        {name: spec["points"] for name, spec in config["rule_defined_anchors"]["metrics"].items()}
    )
    calibration["hash"] = digest(calibration)
    packet = {
        "kind": kind,
        "as_of_date": "2026-09-30",
        "frozen_at": "2026-10-16T10:00:00+08:00",
        "history_months": 36,
        "benchmark_verified": True,
        "classification_verified": True,
        "sources": {"TEST_ONLY": {"published_on": "2026-09-30", "received_at": "2026-10-01T10:00:00+08:00"}},
        "metrics": {
            m: {
                "value": value if m in HIGHER_BETTER else 100 - value,
                "source_hashes": ["TEST_ONLY"],
                "observed_on": "2026-09-30",
            }
            for m in metric_names(config)
        },
    }
    packet["metrics"]["governance_state"]["value"] = 2 if value >= 70 else 1 if value >= 30 else 0
    packet["metrics"]["holder_pressure_band"]["value"] = 0 if value >= 70 else 1 if value >= 30 else 2
    packet["metrics"]["key_person_coverage"]["value"] = str(Decimal(value) / 100)
    return packet, calibration


@pytest.mark.parametrize("kind", KINDS)
def test_eight_dimensions_count_once_with_type_specific_weights(kind):
    # 状态项只有0/1/2等明确语义，统一满分算例用各自真实上端，不能把“95”当治理状态。
    packet, calibration = sample(kind, value=100)
    config = candidate_policy(kind)
    assert sum(config["weights"].values()) == 100
    assert len(config["weights"]) == 8
    assert "calmar" not in metric_names(config)
    result = arithmetic(packet, calibration)
    assert result["status"] == "ARITHMETIC_ONLY_NOT_VALIDATED" and result["grade"] == "EXCELLENT"
    assert len(result["dimensions"]) == 8
    assert "100" == str(float(result["score"])).removesuffix(".0")


def test_missing_metrics_or_calibration_never_redistribute_or_produce_grade():
    packet, calibration = sample()
    del packet["metrics"]["cost_1y"]
    result = assess(packet, calibration)
    assert result["grade"] is None and result["score"] is None
    assert "MISSING:cost_1y" in result["reasons"]
    assert "INDEPENDENT_CALIBRATION_REQUIRED" in assess(packet, None)["reasons"]


def test_management_tenure_alone_cannot_grant_full_dimension_and_weakness_caps_total():
    packet, calibration = sample(value=100)
    packet["metrics"]["governance_state"]["value"] = 0
    packet["metrics"]["key_person_coverage"]["value"] = 0
    result = arithmetic(packet, calibration)
    assert float(result["dimensions"]["management"]) == 20
    assert float(result["score"]) == 88
    assert result["raw_grade"] == "GOOD" and result["grade"] == "AVERAGE"
    assert result["grade_cap_reasons"] == ["KEY_DIMENSION_WEAK:management"]


def test_critical_minimum_and_pending_event_prevent_excellent():
    packet, calibration = sample(value=100)
    for name in ("extra_drawdown", "extra_downside"):
        packet["metrics"][name]["value"] = 40
    assert arithmetic(packet, calibration)["grade"] == "GOOD"
    packet["material_event_pending"] = True
    result = arithmetic(packet, calibration)
    assert result["status"] == "REVIEW_REQUIRED" and result["score"] is None


@pytest.mark.parametrize("change", ["overlap", "future_source", "future_observation", "changed_hash"])
def test_rejects_leakage_and_unverified_calibration(change):
    packet, calibration = sample()
    if change == "overlap":
        calibration["validation_product_ids"] = [calibration["product_ids"][0]]
        calibration.pop("hash")
        calibration["hash"] = digest(calibration)
    elif change == "future_source":
        packet["sources"]["TEST_ONLY"]["received_at"] = "2026-10-17T10:00:00+08:00"
    elif change == "future_observation":
        packet["metrics"]["cost_1y"]["observed_on"] = "2026-10-01"
    else:
        calibration = deepcopy(calibration)
        calibration["anchors"]["cost_1y"][0][1] = 20
    with pytest.raises(ValueError):
        assess(packet, calibration)
