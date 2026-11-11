"""通用接线只验证边界；真实类型公式仍须独立原件和样本验证。"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from app.services import fund_market_review
from app.services import fund_rating_typed_disclosure as typed


@pytest.fixture
def sample(monkeypatch):
    digest = "a" * 64
    inputs = {
        "fund_code": "021740",
        "share_class": "C",
        "as_of_date": "2026-09-30",
        "holding_date": "2026-06-30",
        "legal_name": "隔离名称",
        "manager_name": "隔离管理人",
        "primary_ref": digest,
        "sources": {digest: {"sha256": digest}},
    }
    bundle = {
        **{key: inputs[key] for key in ("fund_code", "share_class", "as_of_date", "holding_date")},
        "family": "INDEX",
        "identity": {key: inputs[key] for key in ("legal_name", "manager_name")},
        "original_report_archive": {"sha256": digest},
        typed.FIELD: {"profile": "GOLD", "revision": typed.REVISION, "inputs": inputs, "input_archive": {}},
    }
    ends = [
        "2025-09-30",
        "2025-10-31",
        "2025-11-30",
        "2025-12-31",
        "2026-01-31",
        "2026-02-28",
        "2026-03-31",
        "2026-04-30",
        "2026-05-31",
        "2026-06-30",
        "2026-07-31",
        "2026-08-31",
        "2026-09-30",
    ]
    result = {
        "dimensions": dict.fromkeys(typed.KEYS.values(), "75"),
        "blocked": False,
        "facts": {"window": {"from": ends[0], "through": ends[-1], "month_ends": ends, "valuation_points": 240}},
    }
    spec = {**typed.PROFILES["GOLD"], "enabled": True}
    monkeypatch.setitem(typed.PROFILES, "GOLD", spec)
    monkeypatch.setattr(typed.importlib, "import_module", lambda _: SimpleNamespace(calculate=lambda *a, **kw: result))
    weights = {key: typed.policy(spec["basis"])["weights"][internal] for key, internal in typed.KEYS.items()}
    dimensions = {key: {"lower": "75", "upper": "75", "candidate": "75"} for key in typed.KEYS}
    return bundle, result, dimensions, weights


def test_unvalidated_type_cannot_be_enabled_by_frozen_payload(sample, monkeypatch, tmp_path):
    bundle, _, _, _ = sample
    monkeypatch.setitem(typed.PROFILES, "GOLD", {**typed.PROFILES["GOLD"], "enabled": False})
    bundle[typed.FIELD]["enabled"] = True
    with pytest.raises(ValueError, match="NOT_VALIDATED"):
        typed.calculate(bundle, root=tmp_path)


def test_verified_partial_release_cannot_produce_total_even_with_eight_points(sample, monkeypatch, tmp_path):
    bundle, _, dimensions, weights = sample
    monkeypatch.setitem(
        typed.PROFILES, "GOLD", {**typed.PROFILES["GOLD"], "enabled": False, "partial_enabled": True}
    )
    result = typed.calculate(bundle, root=tmp_path)
    assert result["score"] is None
    assert result["partial_release_only"] is True
    assert result["complete_type_validated"] is False
    assert typed.replay_snapshot(bundle, dimensions, weights, root=tmp_path) is None


def test_partial_release_still_rejects_frozen_points_and_live_material_events(sample, monkeypatch, tmp_path):
    bundle, _, dimensions, weights = sample
    monkeypatch.setitem(
        typed.PROFILES, "GOLD", {**typed.PROFILES["GOLD"], "enabled": False, "partial_enabled": True}
    )
    dimensions["risk"]["candidate"] = "100"
    with pytest.raises(ValueError, match="POINT_MISMATCH"):
        typed.replay_snapshot(bundle, dimensions, weights, root=tmp_path)
    monkeypatch.setattr(fund_market_review, "rating_review_required", lambda _: True)
    with pytest.raises(ValueError, match="EVENT_REVIEW_REQUIRED"):
        typed.publication_check([bundle], root=tmp_path)


@pytest.mark.parametrize(
    "key", ["fund_code", "share_class", "as_of_date", "holding_date", "legal_name", "manager_name", "primary_ref"]
)
def test_wrong_identity_or_original_source_is_rejected(sample, key, tmp_path):
    bundle, _, _, _ = sample
    bundle[typed.FIELD]["inputs"][key] = "wrong"
    with pytest.raises(ValueError, match="IDENTITY_BINDING"):
        typed.calculate(bundle, root=tmp_path)


def test_other_method_cannot_be_smuggled_into_single_type_result(sample, tmp_path):
    bundle, _, _, _ = sample
    bundle["r5_calculation"] = {"old": "method"}
    with pytest.raises(ValueError, match="CONFLICTING_METHOD"):
        typed.calculate(bundle, root=tmp_path)


def test_input_archive_is_compared_to_frozen_inputs(sample, tmp_path):
    bundle, _, _, _ = sample
    typed.validate_input_archive(
        bundle, root=tmp_path, archive_reader=lambda *_: deepcopy(bundle[typed.FIELD]["inputs"])
    )
    with pytest.raises(ValueError, match="ARCHIVE_MISMATCH"):
        typed.validate_input_archive(bundle, root=tmp_path, archive_reader=lambda *_: {"changed": True})


@pytest.mark.parametrize("point", [True, "NaN", "Infinity", "-1", "101"])
def test_invalid_points_are_not_silently_coerced(sample, point, tmp_path):
    bundle, result, _, _ = sample
    result["dimensions"]["risk"] = point
    with pytest.raises(ValueError):
        typed.calculate(bundle, root=tmp_path)


def test_missing_necessary_point_is_not_renormalized(sample, tmp_path):
    bundle, result, dimensions, weights = sample
    result["dimensions"]["risk"] = None
    dimensions["risk"] = {"lower": "0", "upper": "100", "candidate": None}
    assert typed.replay_snapshot(bundle, dimensions, weights, root=tmp_path) is None
    assert typed.calculate(bundle, root=tmp_path)["score"] is None
    dimensions["risk"]["candidate"] = "50"
    with pytest.raises(ValueError, match="UNKNOWN_BECAME_POINT"):
        typed.replay_snapshot(bundle, dimensions, weights, root=tmp_path)


def test_changed_weights_and_changed_frozen_points_are_rejected(sample, tmp_path):
    bundle, _, dimensions, weights = sample
    with pytest.raises(ValueError, match="OUTER_WEIGHTS"):
        typed.replay_snapshot(bundle, dimensions, {**weights, "fee": 100}, root=tmp_path)
    dimensions["risk"]["candidate"] = "100"
    with pytest.raises(ValueError, match="POINT_MISMATCH"):
        typed.replay_snapshot(bundle, dimensions, weights, root=tmp_path)


def test_year_window_cannot_skip_a_month_or_use_another_cutoff(sample, tmp_path):
    bundle, result, _, _ = sample
    result["facts"]["window"]["month_ends"][4] = "2026-02-01"
    with pytest.raises(ValueError, match="YEAR_WINDOW"):
        typed.calculate(bundle, root=tmp_path)


def test_material_risk_blocks_new_publication_and_total(sample, monkeypatch, tmp_path):
    bundle, result, _, _ = sample
    monkeypatch.setattr(fund_market_review, "rating_review_required", lambda _: False)
    result["blocked"] = True
    assert typed.calculate(bundle, root=tmp_path)["score"] is None
    with pytest.raises(ValueError, match="MATERIAL_RISK_UNRESOLVED"):
        typed.publication_check([bundle], root=tmp_path)


def test_current_event_blocks_new_publication_without_breaking_history(sample, monkeypatch, tmp_path):
    bundle, _, dimensions, weights = sample
    monkeypatch.setattr(fund_market_review, "rating_review_required", lambda _: True)
    assert typed.replay_snapshot(bundle, dimensions, weights, root=tmp_path)["score"] == "75.00"
    with pytest.raises(ValueError, match="EVENT_REVIEW_REQUIRED"):
        typed.publication_check([bundle], root=tmp_path)


def test_material_update_cannot_replace_gold_with_stock_template_or_another_type():
    old = {"members": [{"fund_code": "021740", "bundle": {typed.FIELD: {"profile": "GOLD"}}}]}
    correct = deepcopy(old)
    correct["members"][0]["bundle"][typed.FIELD]["new_period"] = True
    typed.protect_current_snapshot(old, correct)
    for bundle in ({"evidence_v1_calculation": {"version": "stock"}}, {typed.FIELD: {"profile": "BOND_INDEX"}}):
        with pytest.raises(ValueError, match="TYPED_DISCLOSURE_DOWNGRADE"):
            typed.protect_current_snapshot(old, {"members": [{"fund_code": "021740", "bundle": bundle}]})
    typed.protect_current_snapshot({"members": []}, correct)
