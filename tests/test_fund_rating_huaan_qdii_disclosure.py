"""专用补项不得误绑份额、改权重或让后来事项破坏历史重放。"""

from copy import deepcopy
from decimal import Decimal

import pytest
from app.services import fund_market_review
from app.services import fund_rating_huaan_qdii_disclosure as component
from app.services.fund_rating_reviewed_quality import PASSIVE_BASIS, policy
from app.services.fund_rating_reviewed_snapshot import BASIS, KEYS, publish_reviewed


def binding():
    inputs = {
        "fund_code": "014978",
        "share_class": "C",
        "as_of_date": "2026-09-30",
        "holding_date": "2026-06-30",
        "primary_ref": "primary",
        "legal_name": "legal",
        "manager_name": "manager",
    }
    bundle = {
        **{k: inputs[k] for k in ("fund_code", "share_class", "as_of_date", "holding_date")},
        "original_report_archive": {"sha256": "primary"},
        "identity": {"legal_name": "legal", "manager_name": "manager"},
        "family": "INDEX",
        "huaan_qdii_relative_calculation": {"existing": True},
        "huaan_qdii_benchmark_calculation": {"existing": True},
    }
    return bundle, inputs


@pytest.mark.parametrize(
    "field", ["fund_code", "share_class", "as_of_date", "holding_date", "primary_ref", "legal_name", "manager_name"]
)
def test_component_cannot_be_attached_to_another_frozen_identity(field):
    bundle, inputs = binding()
    component.check_binding(bundle, inputs)
    wrong = deepcopy(inputs)
    wrong[field] = "wrong"
    with pytest.raises(ValueError, match="BINDING"):
        component.check_binding(bundle, wrong)


def test_original_three_year_components_cannot_be_removed():
    bundle, inputs = binding()
    bundle.pop("huaan_qdii_relative_calculation")
    with pytest.raises(ValueError, match="EXISTING_REPLAY_REQUIRED"):
        component.check_binding(bundle, inputs)


def totals():
    weights = {key: policy(PASSIVE_BASIS)["weights"][value] for key, value in KEYS.items()}
    dimensions = {key: {"candidate": "75"} for key in KEYS}
    dimensions["return"]["candidate"] = "74.7519"
    return weights, dimensions


def test_historical_total_uses_two_decimals_and_ignores_later_live_events(monkeypatch, tmp_path):
    monkeypatch.setattr(component, "replay_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(fund_market_review, "rating_review_required", lambda code: True)
    weights, dimensions = totals()
    result = component.completed_total({"fund_code": "014978"}, dimensions, weights, root=tmp_path)
    assert Decimal(result["score"]) == Decimal("74.98")


def test_partial_points_and_changed_outer_weights_cannot_form_total(monkeypatch, tmp_path):
    monkeypatch.setattr(component, "replay_snapshot", lambda *args, **kwargs: None)
    weights, dimensions = totals()
    wrong_weights = {**weights, "fee": 100}
    with pytest.raises(ValueError, match="FROZEN_OUTER_WEIGHTS"):
        component.completed_total({}, dimensions, wrong_weights, root=tmp_path)
    dimensions["risk"]["candidate"] = None
    with pytest.raises(ValueError, match="EIGHT_POINTS_REQUIRED"):
        component.completed_total({}, dimensions, weights, root=tmp_path)


def test_pending_live_event_blocks_new_publication_before_database_access(monkeypatch, tmp_path):
    monkeypatch.setattr(fund_market_review, "rating_review_required", lambda code: True)
    dossier = {
        "basis": BASIS,
        "entries": [{"fund_code": "014978", "huaan_qdii_disclosure_calculation": {"inputs": {}}}],
    }
    with pytest.raises(ValueError, match="EVENT_REVIEW_REQUIRED"):
        publish_reviewed(None, dossier, "test pending event gate", root=tmp_path)
