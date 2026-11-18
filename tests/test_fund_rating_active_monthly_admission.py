from decimal import Decimal as D

import pytest

from app.services.fund_rating_active_monthly_admission import exact_target_ratio, classify_exposure, replay_monthly_admission


def report():
    return {"success": 1, "data": {"security": [{"stock": "002112"}], "notice_date": "2026-10-27",
        "notice_content": "本报告期自2026年7月1日起至2026年9月30日。\n4.期末基金资产净值 100.00 100.00\n"
                          "5.期末基金份额净值 1.00 1.00\n期末基金资产组合情况\n1 权益投资 190.00 42.22\n"
                          "其中:股票 190.00 42.22\n9 合计 450.00 100.00"}}


def test_target_uses_all_share_net_assets_not_gross_or_single_share():
    value = exact_target_ratio(report(), code="002112", aliases=["001412", "002112"],
                               period_end="2026-09-30", as_of="2026-10-30")
    assert value == D("95")


def test_other_quarter_cannot_be_relabelled_as_current():
    with pytest.raises(ValueError, match="ORIGINAL_IDENTITY_PERIOD"):
        exact_target_ratio(report(), code="002112", aliases=["001412", "002112"],
                           period_end="2026-06-30", as_of="2026-10-30")


def points():
    dates = ["2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31", "2025-03-31",
             "2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30", "2026-09-30"]
    item = {"code": "001197", "quarters": [{"date": day, "equity_nav_pct": "90", "rounding_radius_pp": ".005"} for day in dates],
            "gaps": [], "region_status_by_cutoff": {"90": "REGION_PASS", "100": "REGION_PASS"}}
    return item, {day: D(90) for day in dates}


def test_new_quarter_can_remove_old_member_without_using_returns():
    item, target = points()
    item["quarters"][-1]["equity_nav_pct"] = "79"
    result = classify_exposure(item, target)
    assert result["budget_definitely_failed"] is True
    assert result["risk_admitted"] is False and result["structure_admitted"] is False


def test_risk_90_and_structure_100_do_not_share_one_membership_rule():
    item, target = points()
    item["region_status_by_cutoff"]["100"] = "REGION_SEPARATE"
    result = classify_exposure(item, target)
    assert result["risk_admitted"] is True and result["structure_admitted"] is False


def test_actual_amount_precision_closes_cross_boundary_without_midpoint():
    item, target = points()
    target["2024-09-30"] = D("94.09837010780821978961306890")
    item["quarters"][3].update(equity_nav_pct="84.10", rounding_radius_pp=".015")
    assert classify_exposure(item, target)["budget_precision_review_required"] is True
    result = classify_exposure(item, target, precise={"2024-09-30": D("84.09140153015130188408122400")})
    assert result["budget_definitely_failed"] is True
    assert result["budget_precision_review_required"] is False


def test_cannot_replace_original_parent_universe(tmp_path):
    with pytest.raises(ValueError, match="ORIGINAL_PARENT_REQUIRED"):
        replay_monthly_admission({"version": "R5_ACTIVE_ROLLING_COHORT_ORIGINALS_V1", "fund_code": "002112",
                                  "parent_cohort_ref": "a-smaller-convenient-peer-list"}, root=tmp_path)
