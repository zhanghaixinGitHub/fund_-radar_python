"""个体特征公式及前向统计身份；未成熟或别的基金统计不得借当前来源进入预测。"""

from copy import deepcopy

import pytest
from app.services import direction_1d_sprint_fund_moneyflow_data as d


def prior():
    return {
        "beta": [0.2, -0.1],
        "count": 126,
        "available_fraction": 1.0,
        "max_t": "2026-09-11",
        "max_u": "2026-09-14",
        "max_mature": "2026-09-15",
    }


def point():
    return {"date": "2026-09-16", "available": True, "large_imbalance": 0.1, "net_fraction": -0.2}


def test_current_feature_contains_fund_identity_and_expected_product():
    value = d.feature("fund", "2026-09-17", prior(), point())
    d.validate_feature(value, point())
    assert value["code"] == "fund" and value["available"]
    assert value["extra"] == [0.2, -0.1, 0.2 * 0.1, -0.1 * -0.2, 1.0]


def test_changed_derived_product_is_rejected():
    value = d.feature("fund", "2026-09-17", prior(), point())
    value["extra"][2] = 0.9
    with pytest.raises(ValueError, match="FEATURE_FORMULA_CHANGED"):
        d.validate_feature(value, point())


def test_missing_flow_remains_missing():
    value = d.feature("fund", "2026-09-17", prior(), {"available": False})
    d.validate_feature(value, {"available": False})
    assert value["available"] is False and value["extra"] is None


@pytest.fixture
def current(monkeypatch, tmp_path):
    monkeypatch.setattr(d.b, "ROOT", tmp_path)
    value = {"cutoff": "2026-09-16", "contexts": {"fund": prior()}}
    d.b.save(d.root() / "context-reference.json", {"current_contexts_hash": d.b.digest(value)})
    monkeypatch.setattr(d.parent.data, "extend", lambda market, t, u, points: market | {"stock_moneyflow": points[t]})
    return value


def test_frozen_current_can_be_used_only_at_or_after_cutoff(current):
    value = d.extend(
        {"original": 1}, "2026-09-16", "2026-09-17", {"2026-09-16": point()}, code="fund", frozen_current=current
    )
    assert value["original"] == 1 and value["fund_moneyflow"]["code"] == "fund"
    with pytest.raises(ValueError, match="CURRENT_CONTEXT_USED_TOO_EARLY"):
        d.extend({}, "2026-09-12", "2026-09-15", {}, code="fund", frozen_current=current)


def test_current_stats_cannot_be_substituted(current):
    changed = deepcopy(current)
    changed["contexts"]["fund"]["beta"] = [0.0, 0.0]
    with pytest.raises(ValueError, match="CURRENT_CONTEXTS_CHANGED"):
        d.extend({}, "2026-09-16", "2026-09-17", {}, code="fund", frozen_current=changed)


def test_unknown_fund_not_filled_with_group_default(current):
    with pytest.raises(ValueError, match="UNKNOWN_FUND"):
        d.extend({}, "2026-09-16", "2026-09-17", {}, code="unknown", frozen_current=current)


def test_original_question_identity_excludes_only_new_features():
    row = {"code": "fund", "y": 1, "market": {"original": 2, "stock_moneyflow": {}, "fund_moneyflow": {}}}
    assert d.original_row(row) == {"code": "fund", "y": 1, "market": {"original": 2}}
