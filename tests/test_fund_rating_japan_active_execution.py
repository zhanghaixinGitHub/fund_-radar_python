"""日本主动执行分项的快照绑定单测；不连接数据库或读取私有研究目录。"""

from copy import deepcopy

import pytest
from app.services import fund_rating_japan_active_execution as component


@pytest.fixture
def binding():
    """只构造接线契约；原件计算另由两槽中的24源真实回放验收。"""
    inputs = {
        "fund_code": "019449",
        "share_class": "C",
        "as_of_date": "2026-09-30",
        "holding_date": "2026-06-30",
        "legal_name": "摩根日本精选股票型证券投资基金(QDII)",
        "manager_name": "摩根基金管理(中国)有限公司",
        "primary_ref": "f" * 64,
        "full_reports": {"2026-06-30": "f" * 64},
    }
    bundle = {
        "fund_code": "019449",
        "family": "QDII_EQUITY",
        "share_class": "C",
        "as_of_date": "2026-09-30",
        "holding_date": "2026-06-30",
        "identity": {"legal_name": inputs["legal_name"], "manager_name": inputs["manager_name"]},
        "original_report_archive": {"sha256": "f" * 64},
        "japan_active_execution_calculation": {"inputs": inputs},
        "verified_at": "2026-10-29T19:00:00+00:00",
        "reviewed_document": {"dimensions": {"management": {"candidate": "65"}, "fee": {"candidate": "55"}}},
    }
    return bundle, inputs


def test_binding_accepts_post_prepare_snapshot_without_changing_other_fields(binding):
    bundle, inputs = binding
    bundle["reviewed_document"]["dimensions"]["execution"] = {"lower": "75", "upper": "75", "candidate": "75"}
    before = deepcopy(bundle)
    component.check_binding(bundle, inputs)
    assert bundle == before


@pytest.mark.parametrize(
    "key,value",
    [
        ("fund_code", "OTHER"),
        ("family", "ACTIVE_EQUITY"),
        ("share_class", "A"),
        ("as_of_date", "2026-06-30"),
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


def test_binding_rejects_other_original_report(binding):
    bundle, inputs = binding
    bundle["original_report_archive"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SNAPSHOT_PRIMARY_REPORT"):
        component.check_binding(bundle, inputs)


def test_binding_rejects_changed_typed_raw_inputs(binding):
    bundle, inputs = binding
    bundle["japan_active_execution_calculation"]["inputs"] = {"fund_code": "OTHER"}
    with pytest.raises(ValueError, match="SNAPSHOT_TYPED_INPUT_BINDING"):
        component.check_binding(bundle, inputs)


@pytest.mark.parametrize(
    "key",
    [
        "r5_calculation",
        "active_execution_calculation",
        "passive_components_calculation",
        "passive_execution_calculation",
    ],
)
def test_binding_rejects_conflicting_execution_engines(binding, key):
    bundle, inputs = binding
    bundle[key] = {"inputs": {}}
    with pytest.raises(ValueError, match="SNAPSHOT_CONFLICTING_EXECUTION"):
        component.check_binding(bundle, inputs)


@pytest.mark.parametrize("bound", ["lower", "upper", "candidate"])
@pytest.mark.parametrize("value", [None, "74.99", "NaN", "unparsed"])
def test_snapshot_rechecks_every_bound(binding, monkeypatch, tmp_path, bound, value):
    bundle, _ = binding
    monkeypatch.setattr(component, "calculate", lambda *args, **kwargs: {"score": "75"})
    point = {"lower": "75", "upper": "75", "candidate": "75"}
    point[bound] = value
    with pytest.raises(ValueError, match="SNAPSHOT_"):
        component.replay_snapshot(bundle, {"execution": point}, root=tmp_path)


def test_snapshot_replay_does_not_replace_verified_at_or_prior_dimensions(binding, monkeypatch, tmp_path):
    bundle, _ = binding
    before = deepcopy(bundle)
    answer = {"score": "75", "complete_score": None, "grade": None}
    monkeypatch.setattr(component, "calculate", lambda *args, **kwargs: answer)
    assert (
        component.replay_snapshot(
            bundle, {"execution": {"lower": "75", "upper": "75", "candidate": "75"}}, root=tmp_path
        )
        == answer
    )
    assert bundle == before


def test_same_jpx_issuer_class_codes_are_merged_before_ten_percent_test():
    rows = component.merge_issuers(
        [{"security": "8035", "amount_cny": "6"}, {"security": "80350", "amount_cny": "5"}],
        component.D("100"),
    )
    assert len(rows) == 1
    assert component.D(rows[0]["ratio_to_NAV"]) == component.D("0.11")
