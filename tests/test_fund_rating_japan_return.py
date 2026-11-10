"""019449日本收益组件的快照绑定与单维重放边界。"""
from copy import deepcopy

import pytest

from app.services import fund_rating_japan_return as component


@pytest.fixture
def binding():
    inputs = {
        "fund_code": "019449",
        "share_class": "C",
        "as_of_date": "2026-09-30",
        "holding_date": "2026-06-30",
        "legal_name": "摩根日本精选股票型证券投资基金(QDII)",
        "manager_name": "摩根基金管理(中国)有限公司",
        "primary_ref": "f" * 64,
        "execution_inputs": {"fund_code": "019449"},
    }
    bundle = {
        "basis": "PUBLIC_REVIEWED_QUALITY_SNAPSHOT_V3",
        "family": "QDII_EQUITY",
        "fund_code": "019449",
        "share_class": "C",
        "as_of_date": "2026-09-30",
        "holding_date": "2026-06-30",
        "identity": {"legal_name": inputs["legal_name"], "manager_name": inputs["manager_name"]},
        "original_report_archive": {"sha256": "f" * 64},
        "japan_active_execution_calculation": {"inputs": deepcopy(inputs["execution_inputs"])},
    }
    return bundle, inputs


def test_binding_preserves_product_period_and_execution(binding):
    bundle, inputs = binding
    before = deepcopy(bundle)
    component.check_binding(bundle, inputs)
    assert bundle == before


@pytest.mark.parametrize("key,value", [("fund_code", "007832"), ("family", "ACTIVE_EQUITY"),
                                       ("share_class", "A"), ("as_of_date", "2026-11-06")])
def test_binding_rejects_other_scope(binding, key, value):
    bundle, inputs = binding
    bundle[key] = value
    with pytest.raises(ValueError, match="SNAPSHOT_BINDING"):
        component.check_binding(bundle, inputs)


def test_binding_rejects_changed_execution_input(binding):
    bundle, inputs = binding
    inputs["execution_inputs"]["fund_code"] = "OTHER"
    with pytest.raises(ValueError, match="EXECUTION_INPUT_BINDING"):
        component.check_binding(bundle, inputs)


@pytest.mark.parametrize("bound", ["lower", "upper", "candidate"])
def test_snapshot_replay_checks_every_return_bound(binding, monkeypatch, tmp_path, bound):
    bundle, inputs = binding
    bundle[component.FIELD] = {"inputs": inputs}
    monkeypatch.setattr(component, "calculate", lambda *args, **kwargs: {"dimensions": {"return": "80"}})
    point = {"lower": "80", "upper": "80", "candidate": "80"}
    point[bound] = None
    with pytest.raises(ValueError, match="SNAPSHOT_RECALCULATION"):
        component.replay_snapshot(bundle, {"return": point}, root=tmp_path)


def test_snapshot_replay_does_not_mutate_other_dimensions(binding, monkeypatch, tmp_path):
    bundle, inputs = binding
    bundle[component.FIELD] = {"inputs": inputs}
    before = deepcopy(bundle)
    monkeypatch.setattr(component, "calculate", lambda *args, **kwargs: {"dimensions": {"return": "80"}})
    dimensions = {"return": {"lower": "80", "upper": "80", "candidate": "80"}, "risk": None}
    assert component.replay_snapshot(bundle, dimensions, root=tmp_path)["dimensions"] == {"return": "80"}
    assert bundle == before and dimensions["risk"] is None
