"""业绩准备的关键防错：泄漏、跨期间、负值区间、目录误分类和选择范围。"""

from copy import deepcopy

import pytest
from app.services.fund_earnings_evidence_v1 import (
    compare_claims,
    earnings_kind,
    plan_groups,
    review_claim,
)


def claim(**updates):
    value = {
        "issuer": "002422",
        "period_start": "2022-01-01",
        "period_end": "2022-06-30",
        "metric": "PARENT_NET_PROFIT",
        "basis": "CONSOLIDATED_ATTRIBUTABLE_TO_PARENT",
        "currency": "CNY",
        "unit": "万元",
        "values": ["100", "120"],
        "kind": "FORECAST",
        "published_date": "2022-04-27",
        "document_id": "a",
        "page": 1,
        "quote": "归母净利润100万元至120万元",
        "source_identity_verified": True,
        "revision_issues": [],
    }
    value.update(updates)
    return review_claim(value, [value["quote"]])


@pytest.mark.parametrize("title", ["年度报告审议工作规程", "关于年报问询函的回复", "年度报告披露延期公告"])
def test_supporting_titles_are_not_financial_results(title):
    assert earnings_kind(title) is None


def test_correction_preserves_forecast_type():
    assert earnings_kind("2022年半年度业绩预告修正公告") == "FORECAST"
    assert earnings_kind("2022年半年度报告更正公告") == "CORRECTION_NOTICE"


def test_planner_uses_whole_group_and_fund_diversity():
    rows, previous = [], []
    for fund, report, days, stocks in [("A", "r1", 5, ["a"]), ("A", "r2", 4, ["b"]), ("B", "r3", 3, ["c", "d"])]:
        for day in range(1, days + 1):
            target = f"2022-01-{day:02d}" if report != "r2" else f"2022-02-{day:02d}"
            rows.append({"fund_code": fund, "report_sha256": report, "target": target})
            previous.append(
                {"fund_code": fund, "target": target, "catalog_complete": False, "remaining_companies": stocks}
            )
    result = plan_groups(rows, previous, max_groups=2, max_windows=3)
    assert [g["report_sha256"] for g in result["selected"]] == ["r1", "r3"]
    assert len(result["windows"]) == 3
    # 引入方向列也不能改变选择；无需加载、利用任何真实答案。
    changed = [{**r, "direction": "synthetic"} for r in rows]
    assert plan_groups(changed, previous, 2, 3) == result


def test_negative_forecast_and_cross_zero_remain_signed():
    a = claim(values=["-120", "-100"], quote="归母净利润-120万元至-100万元")
    b = claim(values=["-10", "20"], quote="归母净利润-10万元至20万元", document_id="b", published_date="2022-07-07")
    result = compare_claims(a, b, "2022-07-08T08:00:00+08:00")
    assert result["relation"] == "ABOVE_PREVIOUS_RANGE"
    assert result["midpoint_change_cny"] == "1150000"
    assert result["percent_change"] is None


def test_actual_point_compares_to_entire_forecast_range():
    b = claim(
        values=["119"], quote="归母净利润119万元", kind="REPORTED_RESULT", document_id="b", published_date="2022-08-30"
    )
    result = compare_claims(claim(), b, "2022-08-31T08:00:00+08:00")
    assert result["relation"] == "OVERLAPS_PREVIOUS_RANGE"
    assert result["comparison_kind"] == "FORECAST_TO_REPORTED_RESULT"


@pytest.mark.parametrize(
    "change", [{"period_end": "2022-09-30"}, {"issuer": "000001"}, {"basis": "PARENT_ONLY"}, {"metric": "REVENUE"}]
)
def test_different_accounting_claims_never_compare(change):
    newer = claim(document_id="b", published_date="2022-07-07", **change)
    with pytest.raises(ValueError, match="INCOMPARABLE"):
        compare_claims(claim(), newer, "2022-07-08T08:00:00+08:00")


@pytest.mark.parametrize("cutoff", ["2022-07-07T23:59:00+08:00", "2022-07-08T07:59:59+08:00"])
def test_date_only_disclosure_not_used_early(cutoff):
    newer = claim(document_id="b", published_date="2022-07-07")
    with pytest.raises(ValueError, match="NOT_AVAILABLE"):
        compare_claims(claim(), newer, cutoff)


def test_retrospective_old_range_is_not_original_disclosure():
    newer = claim(published_date="2022-07-07")
    with pytest.raises(ValueError, match="INDEPENDENT_DISCLOSURES"):
        compare_claims(claim(), newer, "2022-07-08T08:00:00+08:00")


@pytest.mark.parametrize("change", [{"revision_issues": ["LATER_MODIFIED"]}, {"source_identity_verified": False}])
def test_unresolved_source_cannot_enter_pair(change):
    newer = claim(document_id="b", published_date="2022-07-07", **change)
    with pytest.raises(ValueError, match="SOURCE_REVIEW"):
        compare_claims(claim(), newer, "2022-07-08T08:00:00+08:00")


def test_amount_requires_exact_anchor_not_substring():
    spec = deepcopy(claim())
    spec["values"] = ["10", "12"]
    with pytest.raises(ValueError, match="VALUE_NOT_IN_ANCHOR"):
        review_claim(spec, [spec["quote"]])


def test_reversed_negative_interval_rejected():
    with pytest.raises(ValueError, match="REVERSED"):
        claim(values=["-100", "-120"], quote="归母净利润-100万元至-120万元")


def test_missing_unit_and_missing_date_do_not_default():
    with pytest.raises(ValueError, match="UNIT_NOT_IN_ANCHOR"):
        claim(quote="归母净利润100至120")
    with pytest.raises(ValueError, match="INCOMPLETE"):
        claim(published_date=None)


def test_table_numbers_keep_cell_boundaries_and_double_dash_range():
    value = claim(values=["66,530", "76,387"], quote="归母净利润（万元）66,530 -- 76,387 49,281.71")
    assert [m["cny"] for m in value["money"]] == ["665300000", "763870000"]
