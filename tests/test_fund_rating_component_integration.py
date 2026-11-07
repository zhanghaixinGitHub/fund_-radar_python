"""新增原件组件必须绑定原份额；缺完整原始八维包时不能靠分项数量晋级。"""

from copy import deepcopy

import pytest
from app.services import (
    fund_rating_active_stability,
    fund_rating_bosera_qdii_performance,
    fund_rating_enhanced_performance,
    fund_rating_gold_components,
    fund_rating_passive_execution,
    fund_rating_passive_performance,
)
from app.services import fund_rating_reviewed_snapshot as snapshots


def bundle():
    """纯隔离输入仅验证组件边界，不冒充真实基金资料或实际评分。"""
    points = {
        "return": "11",
        "risk": "22",
        "execution": "75",
        "stability": "33",
        "management": "44",
        "holdings": "55",
        "fee": "66",
        "liquidity": None,
    }
    own = {
        "fund_code": "160119",
        "as_of_date": "2026-09-30",
        "holding_date": "2026-06-30",
        "legal_name": "隔离测试产品",
        "primary_ref": "original",
        "sources": {},
    }
    value = {
        "fund_code": "160119",
        "as_of_date": "2026-09-30",
        "holding_date": "2026-06-30",
        "family": "INDEX",
        "share_class": "A",
        "identity": {"legal_name": "隔离测试产品", "manager_name": "隔离管理人"},
        "original_report_archive": {"sha256": "original"},
        "reviewed_document": {
            "code": "160119",
            "as_of": "2026-09-30",
            "outer_weights_percent": {key: "12.5" for key in snapshots.KEYS},
            "dimensions": points,
        },
        "passive_execution_calculation": {"inputs": own},
        "passive_performance_calculation": {
            "inputs": {
                "fund_code": "160119",
                "as_of_date": "2026-09-30",
                "share_class": "A",
                "target_execution_inputs": {k: v for k, v in own.items() if k != "sources"},
            }
        },
    }
    return value


@pytest.mark.parametrize("damage", [None, "foreign_share", "foreign_primary", "forged_point", "eight_without_full_raw"])
def test_gold_components_replay_original_scope_and_keep_total_missing(monkeypatch, tmp_path, damage):
    value = bundle()
    value.pop("passive_execution_calculation")
    value.pop("passive_performance_calculation")
    value["fund_code"] = value["reviewed_document"]["code"] = "000216"
    raw = {
        "fund_code": "000216",
        "as_of_date": value["as_of_date"],
        "holding_date": value["holding_date"],
        "share_class": "A",
        "legal_name": value["identity"]["legal_name"],
        "manager_name": value["identity"]["manager_name"],
        "primary_ref": "original",
    }
    value["gold_components_calculation"] = {"inputs": raw}
    monkeypatch.setattr(snapshots, "archive_root", lambda: tmp_path)

    def replay(inputs, *, root):
        assert root == tmp_path
        return {"dimensions": {"return": "11", "stability": "33"}}

    monkeypatch.setattr(fund_rating_gold_components, "calculate", replay)
    if damage == "foreign_share":
        raw["share_class"] = "C"
    elif damage == "foreign_primary":
        raw["primary_ref"] = "another-product"
    elif damage == "forged_point":
        value["reviewed_document"]["dimensions"]["return"] = "99"
    elif damage == "eight_without_full_raw":
        value["reviewed_document"]["dimensions"]["liquidity"] = "50"
    if damage in {"foreign_share", "foreign_primary", "forged_point"}:
        with pytest.raises(ValueError, match="R5_GOLD_COMPONENT_"):
            snapshots.calculate(value)
    else:
        result = snapshots.calculate(value)
        assert result["status"] == "PARTIAL" and result["score"] is None
        assert result["point_dimensions"] == (8 if damage else 7)


@pytest.mark.parametrize(
    "damage", [None, "wrong_fund", "changed_execution_basis", "forged_component_score", "eight_without_full_raw"]
)
def test_performance_component_default_archive_and_binding(monkeypatch, tmp_path, damage):
    value = bundle()
    monkeypatch.setattr(snapshots, "archive_root", lambda: tmp_path)
    monkeypatch.setattr(fund_rating_passive_execution, "calculate", lambda raw, root: {"score": "75"})

    def replay(raw, *, root):
        assert root == tmp_path  # 正式verify_result不传root，也必须使用真实归档根。
        return {"dimensions": {"return": "11", "risk": "22", "stability": "33"}}

    monkeypatch.setattr(fund_rating_passive_performance, "calculate", replay)
    if damage == "wrong_fund":
        value["passive_performance_calculation"]["inputs"]["fund_code"] = "000001"
    elif damage == "changed_execution_basis":
        value["passive_performance_calculation"]["inputs"]["target_execution_inputs"]["primary_ref"] = "other"
    elif damage == "forged_component_score":
        value["reviewed_document"]["dimensions"]["risk"] = "99"
    elif damage == "eight_without_full_raw":
        value["reviewed_document"]["dimensions"]["liquidity"] = "50"
    if damage in {"wrong_fund", "changed_execution_basis", "forged_component_score"}:
        with pytest.raises(ValueError, match="R5_PASSIVE_PERFORMANCE_"):
            snapshots.calculate(value)
    else:
        result = snapshots.calculate(value)
        assert result["status"] == "PARTIAL" and result["score"] is None
        assert result["point_dimensions"] == (8 if damage else 7)


@pytest.mark.parametrize("damage", [None, "foreign_primary", "wrong_manager", "score_changed"])
def test_active_stability_keeps_primary_identity_and_replays_point(monkeypatch, tmp_path, damage):
    value = deepcopy(bundle())
    value.pop("passive_execution_calculation")
    value.pop("passive_performance_calculation")
    value["family"] = "ACTIVE_EQUITY"
    raw = {
        "fund_code": value["fund_code"],
        "as_of_date": value["as_of_date"],
        "holding_date": value["holding_date"],
        "share_class": "A",
        "legal_name": value["identity"]["legal_name"],
        "manager_name": value["identity"]["manager_name"],
        "refs": {"half": "original"},
    }
    value["stability_calculation"] = {"inputs": raw}
    monkeypatch.setattr(snapshots, "archive_root", lambda: tmp_path)

    def replay(inputs, *, root):
        assert root == tmp_path
        return {"score": "33"}

    monkeypatch.setattr(fund_rating_active_stability, "calculate", replay)
    if damage == "foreign_primary":
        raw["refs"]["half"] = "other"
    elif damage == "wrong_manager":
        raw["manager_name"] = "其他管理人"
    elif damage == "score_changed":
        value["reviewed_document"]["dimensions"]["stability"] = "99"
    if damage:
        with pytest.raises(ValueError, match="R5_STABILITY_"):
            snapshots.calculate(value)
    else:
        result = snapshots.calculate(value)
        assert result["score"] is None and result["dimensions"]["stability"]["candidate"] == "33"


@pytest.mark.parametrize("damage", [None, "foreign_share", "foreign_primary", "forged_point", "eight_without_full_raw"])
def test_enhanced_component_replays_three_points_with_same_product(monkeypatch, tmp_path, damage):
    """原件编译与快照之间不能串份额、串主报告或接收伪造分数。"""
    value = deepcopy(bundle())
    value.pop("passive_execution_calculation")
    value.pop("passive_performance_calculation")
    value["family"] = "ACTIVE_EQUITY"
    value["fund_code"] = value["reviewed_document"]["code"] = "014156"
    value["share_class"] = "C"
    raw = {
        "fund_code": "014156", "share_class": "C",
        "as_of_date": value["as_of_date"], "holding_date": value["holding_date"],
        "legal_name": value["identity"]["legal_name"],
        "manager_name": value["identity"]["manager_name"], "primary_ref": "original",
    }
    value["enhanced_performance_calculation"] = {"inputs": raw}
    monkeypatch.setattr(snapshots, "archive_root", lambda: tmp_path)

    def replay(inputs, *, root):
        assert root == tmp_path
        return {"dimensions": {"return": "11", "risk": "22", "stability": "33"}}

    monkeypatch.setattr(fund_rating_enhanced_performance, "calculate", replay)
    if damage == "foreign_share":
        raw["share_class"] = "A"
    elif damage == "foreign_primary":
        raw["primary_ref"] = "foreign-report"
    elif damage == "forged_point":
        value["reviewed_document"]["dimensions"]["risk"] = "99"
    elif damage == "eight_without_full_raw":
        value["reviewed_document"]["dimensions"]["liquidity"] = "50"
    if damage in {"foreign_share", "foreign_primary", "forged_point"}:
        with pytest.raises(ValueError, match="ENHANCED_CURRENT_BINDING|R5_ENHANCED_PERFORMANCE_"):
            snapshots.calculate(value)
    else:
        checked = snapshots.calculate(value)
        assert checked["status"] == "PARTIAL" and checked["score"] is None
        assert checked["point_dimensions"] == (8 if damage else 7)


@pytest.mark.parametrize("damage", [None, "old_Y_share", "foreign_primary", "forged_point", "eight_without_full_raw"])
def test_bosera_component_uses_corrected_C_identity_and_preserves_unknown_total(monkeypatch, tmp_path, damage):
    value = deepcopy(bundle())
    value.pop("passive_execution_calculation")
    value.pop("passive_performance_calculation")
    value["fund_code"] = value["reviewed_document"]["code"] = "018853"
    value["share_class"] = "C"
    raw = {
        "fund_code": "018853", "share_class": "C", "currency": "CNY",
        "as_of_date": value["as_of_date"], "holding_date": value["holding_date"],
        "legal_name": value["identity"]["legal_name"], "manager_name": value["identity"]["manager_name"],
        "primary_report_hash": "original",
    }
    value["bosera_performance_calculation"] = {"inputs": raw}
    monkeypatch.setattr(snapshots, "archive_root", lambda: tmp_path)

    def replay(inputs, *, root):
        assert root == tmp_path
        return {"component_scores": {"return": "11", "stability": "33"}}

    monkeypatch.setattr(fund_rating_bosera_qdii_performance, "calculate", replay)
    if damage == "old_Y_share":
        value["share_class"] = "Y"
    elif damage == "foreign_primary":
        raw["primary_report_hash"] = "another-report"
    elif damage == "forged_point":
        value["reviewed_document"]["dimensions"]["stability"] = "99"
    elif damage == "eight_without_full_raw":
        value["reviewed_document"]["dimensions"]["liquidity"] = "50"
    if damage in {"old_Y_share", "foreign_primary", "forged_point"}:
        with pytest.raises(ValueError, match="BOSERA_QDII_"):
            snapshots.calculate(value)
    else:
        checked = snapshots.calculate(value)
        assert checked["status"] == "PARTIAL" and checked["score"] is None
        assert checked["point_dimensions"] == (8 if damage else 7)


def test_conditional_unknown_survival_is_not_a_confirmed_failure_or_a_pass():
    events = [{"applicability": "UNKNOWN_APPLICABILITY", "status": "CONDITIONAL_UNVERIFIED",
               "severity": "MATERIAL", "anniversary_net_asset_cny": None}]
    assert fund_rating_bosera_qdii_performance.confirmed_material_events(events) == []
    assert events[0]["anniversary_net_asset_cny"] is None
    confirmed = [{**events[0], "applicability": "FUND_CONFIRMED", "status": "UNRESOLVED"}]
    assert fund_rating_bosera_qdii_performance.confirmed_material_events(confirmed) == confirmed
