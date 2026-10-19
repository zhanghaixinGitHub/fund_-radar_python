"""校准与诊断算法的隔离算例，不代表真实产品样本或独立验收。"""

from datetime import date, datetime
from decimal import Decimal

import pytest
from app.services.fund_quality_v2_research import (
    correlations,
    diagnostics,
    empirical_anchors,
    experiment,
    partial_observations,
    representatives,
)
from app.services.fund_rating_quality_v2 import HIGHER_BETTER, candidate_policy, metric_names


def packet(product, code, month="2026-09-30", value=50, complete=True):
    result = {
        "kind": "ACTIVE_EQUITY",
        "product_id": product,
        "fund_code": code,
        "as_of_date": month,
        "raw_metrics_complete": complete,
        "hash": "TEST_ONLY",
        "frozen_at": "2026-10-16T00:00:00+08:00",
        "metrics": {
            m: {"value": str(value if m in HIGHER_BETTER else 100 - value)}
            for m in metric_names(candidate_policy("ACTIVE_EQUITY"))
        },
    }
    result["metrics"]["governance_state"]["value"] = str(min(2, int(value) // 30))
    result["metrics"]["holder_pressure_band"]["value"] = str(2 - min(2, int(value) // 30))
    result["metrics"]["key_person_coverage"]["value"] = str(Decimal(value) / 100)
    return result


def test_product_representative_does_not_count_a_c_or_months_twice():
    a = packet("TEST_PRODUCT", "000001", "2026-08-31")
    b = packet("TEST_PRODUCT", "000002")
    c = packet("TEST_PRODUCT", "000001")
    assert representatives([a, b, c]) == [c]


def test_empirical_anchors_use_products_and_do_not_invent_constant_endpoints():
    packets = [packet("TEST_" + str(i), str(i).zfill(6), value=i * 20) for i in range(1, 4)]
    anchors = empirical_anchors(packets, "ACTIVE_EQUITY")
    assert [Decimal(p[1]) for p in anchors["relative_return_1y"]] == sorted(
        Decimal(p[1]) for p in anchors["relative_return_1y"]
    )
    assert empirical_anchors(packets + [packets[0]], "ACTIVE_EQUITY") == anchors
    with pytest.raises(ValueError, match="CONSTANT_CALIBRATION_METRIC"):
        empirical_anchors([packets[0]], "ACTIVE_EQUITY")


def test_diagnostics_preserve_sensitivity_caps_and_effective_contributions():
    packets = [packet("TEST_" + str(i), str(i).zfill(6), value=i * 20) for i in range(1, 4)]
    result = diagnostics(packets, empirical_anchors(packets, "ACTIVE_EQUITY"), "ACTIVE_EQUITY")
    month = result["months"][0]
    assert month["product_count"] == 3
    assert all(len(r["weight_sensitivity"]) == 16 for r in month["results"])
    assert abs(sum(month["variance_contribution"].values()) - 1) < 1e-12
    assert all("grade_cap_reasons" in r["baseline"] for r in month["results"])
    assert result["independent_acceptance"] == "NOT_PERFORMED"


def test_rule_states_use_frozen_semantics_when_constant_but_continuous_data_are_not_fabricated():
    packets = [packet("TEST_" + str(i), str(i).zfill(6), value=i * 20) for i in range(1, 4)]
    for item in packets:
        item["metrics"]["key_person_coverage"]["value"] = "1"
        item["metrics"]["governance_state"]["value"] = "2"
        item["metrics"]["holder_pressure_band"]["value"] = "0"
    anchors = empirical_anchors(packets, "ACTIVE_EQUITY")
    assert anchors["key_person_coverage"] == [["0", "0"], ["1", "100"]]
    assert len(packets) == 3 and {p["metrics"]["key_person_coverage"]["value"] for p in packets} == {"1"}
    for item in packets:
        item["metrics"]["cost_1y"]["value"] = "0.01"
    with pytest.raises(ValueError, match="CONSTANT_CALIBRATION_METRIC:cost_1y"):
        empirical_anchors(packets, "ACTIVE_EQUITY")


def test_constant_correlations_are_missing_and_small_samples_are_not_proof():
    assert correlations([{"a": "1"}, {"a": "2"}])["reason"] == "FEWER_THAN_THREE_PRODUCTS"
    pairs = correlations([{"a": str(i), "b": "1"} for i in range(3)])["pairs"]
    assert pairs[0]["pearson"] is None


def test_partial_observations_deduplicate_each_pair_without_becoming_calibration():
    packets = [packet("TEST_" + str(i), str(i).zfill(6), value=i * 20, complete=False) for i in range(1, 4)]
    duplicate_share = packet("TEST_1", "999999", value=99, complete=False)
    result = partial_observations([*packets, duplicate_share])
    group = result["groups"][0]
    assert group["products"] == 3
    assert group["metric_product_counts"]["cost_1y"] == 3
    assert all(pair["product_count"] == 3 for pair in group["pairwise"])
    assert result["status"] == "PARTIAL_RAW_OBSERVATIONS_ONLY" and "anchors" not in result


def test_experiment_keeps_missing_inputs_and_does_not_manufacture_calibration(monkeypatch, tmp_path):
    monkeypatch.setattr("app.services.fund_quality_v2_materials.verify_packet", lambda *args: None)
    p = packet("TEST_ONLY", "000001", complete=False)
    result = experiment(
        [p],
        {"TEST_ONLY": "validation"},
        date(2026, 8, 31),
        tmp_path,
        datetime.fromisoformat("2026-10-16T00:00:00+08:00"),
    )
    assert result["groups"][0]["calibration_products"] == 0
    assert result["groups"][0]["validation_products"] == 0
    assert "calibration" not in result["groups"][0]
    assert result["gaps"][0]["reason"] == "RAW_METRICS_INCOMPLETE"
