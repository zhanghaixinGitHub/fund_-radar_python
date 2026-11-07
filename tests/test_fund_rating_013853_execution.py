"""013853执行分项的绑定、未知成本及实际事项门禁；原件重放另经限槽验收。"""

from copy import deepcopy

import pytest
from app.services import fund_rating_013853_execution as component


@pytest.fixture
def binding():
    """构造快照接线契约；无需测试数据库或把研究目录加入生产依赖。"""
    inputs = {
        "fund_code": "013853",
        "share_class": "A",
        "as_of_date": "2026-09-30",
        "holding_date": "2026-06-30",
        "legal_name": "大成匠心卓越三年持有期混合型证券投资基金",
        "manager_name": "大成基金管理有限公司",
        "primary_ref": "f" * 64,
        "sources": {"f" * 64: {"kind": "pdf"}},
    }
    return {
        "fund_code": "013853",
        "family": "ACTIVE_EQUITY",
        "share_class": "A",
        "as_of_date": "2026-09-30",
        "holding_date": "2026-06-30",
        "identity": {key: inputs[key] for key in ("legal_name", "manager_name")},
        "original_report_archive": {"sha256": "f" * 64},
        "fund_013853_execution_calculation": {"inputs": inputs},
        "verified_at": "2026-10-28T01:35:48.230391+00:00",
        "reviewed_document": {
            "dimensions": {"management": {"candidate": "65"}, "fee": {"applicability": "NOT_APPLICABLE"}}
        },
    }, inputs


def test_post_prepare_binding_preserves_identity_verification_and_fee(binding):
    bundle, inputs = binding
    bundle["reviewed_document"]["dimensions"]["execution"] = dict.fromkeys(("lower", "upper", "candidate"), "75")
    before = deepcopy(bundle)
    component.check_binding(bundle, inputs)
    assert bundle == before


@pytest.mark.parametrize(
    "key,value",
    [
        ("fund_code", "000041"),
        ("family", "QDII_EQUITY"),
        ("share_class", "C"),
        ("as_of_date", "2026-10-30"),
        ("holding_date", "2025-12-31"),
    ],
)
def test_binding_rejects_other_product_share_family_or_time(binding, key, value):
    bundle, inputs = binding
    bundle[key] = value
    with pytest.raises(ValueError, match="SNAPSHOT_PRODUCT_SHARE_DATES"):
        component.check_binding(bundle, inputs)


@pytest.mark.parametrize("key", ["legal_name", "manager_name"])
def test_binding_rejects_other_legal_identity(binding, key):
    bundle, inputs = binding
    bundle["identity"][key] = "其他主体"
    with pytest.raises(ValueError, match="SNAPSHOT_LEGAL_IDENTITY"):
        component.check_binding(bundle, inputs)


def test_binding_rejects_missing_original_primary_reference(binding):
    bundle, inputs = binding
    inputs["sources"].clear()
    with pytest.raises(ValueError, match="SNAPSHOT_PRIMARY_REPORT"):
        component.check_binding(bundle, inputs)


def test_binding_rejects_changed_typed_input(binding):
    bundle, inputs = binding
    bundle["fund_013853_execution_calculation"]["inputs"] = {}
    with pytest.raises(ValueError, match="SNAPSHOT_TYPED_INPUT_BINDING"):
        component.check_binding(bundle, inputs)


@pytest.mark.parametrize(
    "key",
    [
        "r5_calculation",
        "active_execution_calculation",
        "passive_components_calculation",
        "passive_execution_calculation",
        "japan_active_execution_calculation",
    ],
)
def test_binding_rejects_conflicting_execution_engines(binding, key):
    bundle, inputs = binding
    bundle[key] = {"inputs": {}}
    with pytest.raises(ValueError, match="SNAPSHOT_CONFLICTING_EXECUTION"):
        component.check_binding(bundle, inputs)


@pytest.mark.parametrize("bound", ["lower", "upper", "candidate"])
@pytest.mark.parametrize("value", [None, "74.99", "NaN", "unparsed"])
def test_snapshot_rechecks_all_three_bounds(binding, monkeypatch, tmp_path, bound, value):
    bundle, _ = binding
    monkeypatch.setattr(component, "calculate", lambda *args, **kwargs: {"score": "75"})
    point = dict.fromkeys(("lower", "upper", "candidate"), "75")
    point[bound] = value
    with pytest.raises(ValueError, match="SNAPSHOT_"):
        component.replay_snapshot(bundle, {"execution": point}, root=tmp_path)


def test_snapshot_replay_keeps_prior_fields(binding, monkeypatch, tmp_path):
    bundle, _ = binding
    before = deepcopy(bundle)
    answer = {"score": "75", "complete_score": None, "grade": None}
    monkeypatch.setattr(component, "calculate", lambda *args, **kwargs: answer)
    assert (
        component.replay_snapshot(
            bundle, {"execution": dict.fromkeys(("lower", "upper", "candidate"), "75")}, root=tmp_path
        )
        == answer
    )
    assert bundle == before


@pytest.fixture
def transfer_table():
    """转增无成本只保留未知，不影响由数量乘期末估值形成的公允金额。"""
    return (
        "金额单位：人民币元 证券 代码 认购价格 估值 数量 成本总额 股\n"
        "301538 示例 2024-06-01 6个月\n- 12.00 10 - 120.00 转增\n"
    ), [{"stock_code": "301538.SZ", "quantity_shares": "100", "market_value_cny": "1200"}]


def test_transfer_unknown_cost_is_not_zero(transfer_table):
    text, positions = transfer_table
    (row,) = component.parse_stock_valuation_rows(text, positions)
    assert row["amount"] == "120.00"
    assert row["cost"] is row["issue_price"] is None
    assert row["cost_unknown_not_zero"] is True


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("PRICE", "VALUATION_CONFLICT"),
        ("NO_TRANSFER", "COST_SCOPE"),
        ("ONE_DASH", "COST_SCOPE"),
        ("NO_UNIT", "HEADER"),
        ("UNKNOWN_CODE", "FINANCIAL_UNIQUE"),
        ("QUANTITY_BOUND", "COMPLETE_HOLDING_BOUND"),
        ("VALUE_BOUND", "COMPLETE_HOLDING_BOUND"),
    ],
)
def test_transfer_rejects_invalid_disclosure_or_holding_boundary(transfer_table, mutation, error):
    text, positions = transfer_table
    if mutation == "PRICE":
        text = text.replace("120.00", "120.01")
    elif mutation == "NO_TRANSFER":
        text = text.replace("转增", "未知")
    elif mutation == "ONE_DASH":
        text = text.replace("10 - ", "10 0.00 ")
    elif mutation == "NO_UNIT":
        text = text.replace("人民币元", "未知币种")
    elif mutation == "UNKNOWN_CODE":
        positions = []
    else:
        positions[0]["quantity_shares" if mutation == "QUANTITY_BOUND" else "market_value_cny"] = "1"
    with pytest.raises(ValueError, match=error):
        component.parse_stock_valuation_rows(text, positions)
