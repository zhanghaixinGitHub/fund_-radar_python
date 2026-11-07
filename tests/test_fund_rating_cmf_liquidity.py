"""未知集中标签不能补写事实；只有全部结构条件同分且基础证据齐全才识别点值。"""

from copy import deepcopy
from decimal import Decimal

import pytest
from app.services.fund_rating_cmf_liquidity import (
    COMPATIBLE_TYPES,
    FOUNDATION_GATES,
    PROTOCOL,
    invariant_holders_score,
)
from app.services.fund_rating_r5_rules import holders_score


@pytest.fixture
def holder_node():
    return {
        "structure_type": None,
        "structure_condition_protocol": PROTOCOL,
        "compatible_structure_types": deepcopy(COMPATIBLE_TYPES),
        "structure_basis": "OPEN_END_CASH_LEGAL_PRODUCT_AND_COMPLETE_REPORTED_CATEGORY_TABLE",
        "foundation_gate_checks": {key: True for key in FOUNDATION_GATES},
        "structure_closed": True,
        "share_flow_closed": True,
        "redemption_mechanism_verified": True,
        "actual_execution_verified": True,
        "confirmed_pressure_effects": [],
    }


def test_unknown_actual_type_stays_unknown_after_exhaustive_equal_outcomes(holder_node):
    before = deepcopy(holder_node)
    score, detail = holders_score(holder_node)
    assert score == 75 and detail["actual_structure_type"] is None
    assert {row["point"] for row in detail["structure_condition_outcomes"]} == {"75"}
    assert holder_node == before


@pytest.mark.parametrize(
    "key", ["structure_closed", "share_flow_closed", "redemption_mechanism_verified", "actual_execution_verified"]
)
def test_same_score_never_waives_missing_basic_evidence(holder_node, key):
    holder_node[key] = False
    with pytest.raises(ValueError, match="BASIC_EVIDENCE_MISSING"):
        holders_score(holder_node)


@pytest.mark.parametrize("change", ["one_case_only", "invent_actual_label", "unknown_effects", "unknown_foundation"])
def test_uncertain_conditions_or_actual_effects_do_not_pass(holder_node, change):
    if change == "one_case_only":
        holder_node["compatible_structure_types"] = ["DISPERSED"]
    elif change == "invent_actual_label":
        holder_node["structure_type"] = "DISPERSED"
    elif change == "unknown_effects":
        holder_node["confirmed_pressure_effects"] = None
    else:
        holder_node["foundation_gate_checks"]["current_manager_and_trustee_execution"] = False
    with pytest.raises(ValueError):
        holders_score(holder_node)


def test_unequal_condition_scores_cannot_pick_favorable_case_or_midpoint(holder_node):
    def evaluator(case):
        value, details = holders_score(case)
        return value - (Decimal(25) if case["structure_type"] == "CONCENTRATED_OPEN_END" else 0), details

    with pytest.raises(ValueError, match="NOT_POINT_IDENTIFIED"):
        invariant_holders_score(holder_node, evaluator=evaluator)


def test_confirmed_pressure_is_retained_in_each_condition(holder_node):
    holder_node["confirmed_pressure_effects"] = [
        {
            "effect_confirmed": True,
            "source_refs": ["isolated-original"],
            "classification": "LIMITED_ACTUAL_PRESSURE",
            "only_static_cash_proxy": False,
        }
    ]
    assert holders_score(holder_node)[0] == 50
