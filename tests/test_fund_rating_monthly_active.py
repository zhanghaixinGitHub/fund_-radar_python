from decimal import Decimal

import pytest

from app.services.fund_rating_monthly_active import contract_composite


def test_contract_change_applies_on_effective_day_and_carries_only_hk():
    days = ["2026-09-18", "2026-09-21", "2026-09-22"]
    prices = {"equity": dict(zip(days, [100, 110, 121])),
              "bond": dict(zip(days, [100, 100, 100])),
              "hk": {days[0]: 100, days[2]: 110}}
    segments = [{"from": "2020-01-01", "weights": {"equity": ".6", "bond": ".4"}, "source_refs": ["old"]},
                {"from": days[1], "weights": {"equity": ".8", "bond": ".1", "hk": ".1"}, "source_refs": ["new"]}]
    result, carried = contract_composite(days, prices, segments, carry_indices=["hk"])
    assert Decimal(result[-1]) == Decimal("1.1772")
    assert any(row["valuation_date"] == days[1] and row["price_date"] == days[0] for row in carried)


def test_domestic_gap_is_not_filled_by_previous_close():
    with pytest.raises(ValueError, match="PRICE_GAP"):
        contract_composite(["2026-10-29", "2026-10-30"], {"domestic": {"2026-10-29": 100}},
                           [{"from": "2020-01-01", "weights": {"domestic": 1}, "source_refs": ["contract"]}])


@pytest.mark.parametrize("segment", [
    {"from": "2020-01-01", "weights": {"x": ".9"}, "source_refs": ["contract"]},
    {"from": "2020-01-01", "weights": {"x": 1}},
])
def test_unproved_or_incomplete_contract_is_rejected(segment):
    with pytest.raises(ValueError, match="CONTRACT"):
        contract_composite(["2026-10-30"], {"x": {"2026-10-30": 100}}, [segment])


def test_same_asof_resume_is_only_allowed_for_an_existing_partial():
    from app.services.fund_rating_monthly_active import rolling_active_bundle

    original = {"data_dates": {}, "evidence_v1_calculation": {"partial_release_only": True,
                "inputs": {"as_of_date": "2026-10-30", "target_refs": {"half": "half"}, "sources": {}}}}
    history = {"dates": ["2023-10-31", "2026-10-30"], "month_ends": list(range(37))}
    management = {"as_of_date": "2026-10-30", "sources": {"q3": {"sha256": "q3"}},
                  "claims": [{"source_hash": "q3", "quote": "actual role"}],
                  "continuity": {"period_start": "2023-10-31", "period_end": "2026-10-30"},
                  "current_period_review": {"source_hash": "q3", "role_table_quote": "actual role",
                     "expected_active_names": ["own manager"], "report_period_end": "2026-09-30", "report_published_on": "2026-10-27"}}
    args = dict(provider_ref={"sha256": "new-nav"}, benchmark_ref={"sha256": "real-new-index"},
                benchmark_sources=[], management_review=management, verified_at="2026-11-17T13:00:00+08:00")
    result = rolling_active_bundle(original, history, **args)
    assert result["as_of_date"] == "2026-10-30"
    original["evidence_v1_calculation"]["partial_release_only"] = False
    with pytest.raises(ValueError, match="NEW_PERIOD_HISTORY_REQUIRED"):
        rolling_active_bundle(original, history, **args)
