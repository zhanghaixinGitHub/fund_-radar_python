"""核验公开时点截断、披露先后歧义、修订追述的防倒灌边界。"""

import copy

import pytest
from app.services.fund_earnings_asof_v1 import asof_changes, review_correction


def claim(key="early", published="2021-10-08", available="2021-10-09T08:00:00+08:00", value="100"):
    return {
        "issuer": "002812",
        "period_start": "2021-01-01",
        "period_end": "2021-09-30",
        "metric": "PARENT_NET_PROFIT",
        "basis": "CONSOLIDATED_ATTRIBUTABLE_TO_PARENT",
        "currency": "CNY",
        "document_id": key,
        "published_date": published,
        "available_at": available,
        "kind": "FORECAST",
        "money": [{"cny": value}],
        "source_identity_verified": True,
        "revision_issues": [],
    }


def test_future_disclosure_does_not_change_past_result():
    first = claim()
    future = claim("later", "2021-12-31", "2022-01-01T08:00:00+08:00", "102")
    cutoff = "2021-10-30T08:00:00+08:00"
    assert asof_changes([first], cutoff) == asof_changes([future, first], cutoff)
    assert asof_changes([future, first], cutoff)[0]["change"] is None


def test_public_availability_boundary_and_real_change():
    first = claim()
    later = claim("later", "2021-12-31", "2022-01-01T08:00:00+08:00", "102")
    assert asof_changes([first, later], "2022-01-01T07:59:59+08:00")[0]["change"] is None
    assert asof_changes([first, later], later["available_at"])[0]["change"]["midpoint_change_cny"] == "2"


def test_same_instant_cannot_be_ordered_by_document_id():
    a, b = claim("a"), claim("b", value="101")
    result = asof_changes([b, a], a["available_at"])[0]
    assert result["status"] == "AMBIGUOUS_DISCLOSURE_ORDER" and result["change"] is None


def test_same_document_conflicting_values_rejected():
    with pytest.raises(ValueError, match="CONFLICTING_DUPLICATE"):
        asof_changes([claim(), claim(value="101")], claim()["available_at"])


def test_equal_instants_with_different_offsets_are_still_ambiguous():
    a = claim("a")
    b = claim("b", available="2021-10-09T00:00:00+00:00", value="101")
    assert asof_changes([a, b], a["available_at"])[0]["status"] == "AMBIGUOUS_DISCLOSURE_ORDER"


def test_duplicate_same_fact_is_deduplicated():
    a = claim()
    assert asof_changes([a, copy.deepcopy(a)], a["available_at"]) == asof_changes([a], a["available_at"])


def test_periods_are_separate_and_missing_is_not_zero():
    a = claim()
    b = {**claim("q3"), "period_start": "2021-07-01"}
    result = asof_changes([a, b], a["available_at"])
    assert len(result) == 2 and all(r["change"] is None for r in result)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("evidence_role", "QUOTED_PREVIOUS_VALUE", "QUOTED_HISTORY"),
        ("source_identity_verified", False, "SOURCE_REVIEW"),
        ("available_at", "2021-10-08T08:00:00+08:00", "PUBLIC_DATE_BACKDATED"),
    ],
)
def test_unsafe_fact_cannot_enter_timeline(field, value, reason):
    a = {**claim(), field: value}
    with pytest.raises(ValueError, match=reason):
        asof_changes([a], "2022-01-01T08:00:00+08:00")


def correction_specs():
    base = {
        **claim("correction", "2021-12-31"),
        "kind": "REPORTED_RESULT",
        "unit": "元",
        "page": 1,
    }
    return {**base, "quote": "归母净利润（元）100", "values": ["100"]}, {
        **base,
        "quote": "归母净利润（元）102",
        "values": ["102"],
    }


def test_correction_quotes_available_only_at_correction_date():
    old, new = correction_specs()
    result = review_correction(old, new, ["更正前：归母净利润（元）100 更正后：归母净利润（元）102"])
    assert result["change_cny"] == "2"
    assert result["before"]["available_at"] == "2022-01-01T08:00:00+08:00"
    assert not result["may_backfill_original_report"]


def test_reversed_correction_columns_rejected():
    old, new = correction_specs()
    with pytest.raises(ValueError, match="WRONG_SECTION"):
        review_correction(new, old, ["更正前：归母净利润（元）100 更正后：归母净利润（元）102"])
