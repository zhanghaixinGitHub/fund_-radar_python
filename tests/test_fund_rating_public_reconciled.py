"""来源冲突必须逐日解释，不能通过宽容差、漏分红或混份额强行出级。"""

from copy import deepcopy
from decimal import Decimal

import pytest
from app.services.fund_rating_public_index import index_policy
from app.services.fund_rating_public_reconciled import reconcile_series, reconciled_policy


def sample():
    days = ["2025-01-09", "2025-01-10", "2025-01-13"]
    return (
        {
            "dates": days,
            "rows": [
                {"nav_date": d, "unit_nav": str(v), "adjusted_nav": str(v), "content_hash": "original"}
                for d, v in zip(days, (1, 0.9, 0.99), strict=True)
            ],
            "adjusted_nav": ["1", ".9", ".99"],
        },
        {d: {"syNavRate": str(v)} for d, v in zip(days, (0, 0, 10), strict=True)},
        dict(zip(days, map(Decimal, ("1", ".9", ".99")), strict=True)),
        {"2025-01-10": Decimal(".1")},
    )


def test_true_cash_reinvestment_closes_conflict_preserves_original():
    h, issuer, units, events = sample()
    before = deepcopy(h)
    result = reconcile_series(h, issuer, units, events)
    assert h == before
    assert result["adjusted_nav"] == ["1", "1", "1.1"]
    assert result["rows"][1]["provider_adjusted_nav"] == "0.9"
    assert result["rows"][1]["provider_content_hash"] == "original"
    assert result["rows"][1]["content_hash"] != "original"
    assert result["return_source_reconciliation"]["maximum_cash_reinvestment_difference"] == "0"


@pytest.mark.parametrize("change", ["missing_dividend", "wrong_amount", "wrong_unit", "wrong_issuer"])
def test_unexplained_or_wrong_share_difference_rejects(change):
    h, issuer, units, events = sample()
    if change == "missing_dividend":
        events.clear()
    elif change == "wrong_amount":
        events["2025-01-10"] = Decimal(".01")
    elif change == "wrong_unit":
        units["2025-01-10"] = Decimal(".8")
    else:
        issuer["2025-01-13"]["syNavRate"] = "11"
    with pytest.raises(ValueError, match="CONFLICT"):
        reconcile_series(h, issuer, units, events)


def test_new_source_policy_does_not_change_existing_eight_dimension_weights():
    previous, current = index_policy(), reconciled_policy()
    for field in ("terms", "weights", "anchors"):
        assert previous[field] == current[field]
    assert previous["basis"] != current["basis"]
    assert previous["input_contract"]["history"] == "OWN_AUTHORIZED_ADJUSTED_NAV_COMPLETE_THREE_YEARS"
