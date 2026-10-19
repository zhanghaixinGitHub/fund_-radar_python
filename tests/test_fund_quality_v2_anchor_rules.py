"""规则端点不伪造样本；以下是反例测试，不属于真实校准或验证产品。"""

from decimal import Decimal

import pytest
from app.services.fund_quality_v2_anchor_rules import rule_anchor_specs, rule_points


def test_complete_role_coverage_can_be_constant_without_manufacturing_missing_role_samples():
    observed = [Decimal(1)] * 6
    before = list(observed)
    assert rule_points("key_person_coverage", observed) == [["0", "0"], ["1", "100"]]
    assert observed == before and len(observed) == 6 and set(observed) == {Decimal(1)}
    assert rule_anchor_specs()["kind"] == "RULE_DEFINED_NOT_EMPIRICAL_SAMPLE_ENDPOINTS"


@pytest.mark.parametrize(
    "name,value",
    [
        ("key_person_coverage", "1.01"),
        ("key_person_coverage", "NaN"),
        ("governance_state", "1.5"),
        ("holder_pressure_band", "3"),
    ],
)
def test_raw_values_outside_declared_domain_are_not_clipped_or_promoted_to_a_valid_state(name, value):
    with pytest.raises(ValueError, match="RULE_METRIC_DOMAIN"):
        rule_points(name, [Decimal(value)])


def test_callers_cannot_mutate_frozen_future_rule_configuration():
    config = rule_anchor_specs()
    config["metrics"]["governance_state"]["points"][0][1] = "100"
    assert rule_anchor_specs()["metrics"]["governance_state"]["points"][0] == ["0", "0"]
