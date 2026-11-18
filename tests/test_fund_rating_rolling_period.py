from copy import deepcopy

import pytest
from app.services.fund_rating_period import validate_period


class Source:
    def body(self, key):
        assert key == "actual-report"
        return "本基金十月份更新所依据的真实报告原文已经核对，包括持仓日期和适用的份额。"


def test_new_month_needs_actual_review_and_replays_original_quote():
    values = {"fund_code": "019449", "as_of_date": "2026-10-30", "holding_date": "2026-09-30"}
    with pytest.raises(ValueError, match="CURRENT_PERIOD_REVIEW_REQUIRED"):
        validate_period(values, Source())
    values["monthly_period_review"] = {
        **values,
        "reason": (
            "本次实际阅读本基金新期公开报告、净值与基准，对适用日期和份额逐项核对；"
            "原半年报继续保留其真实期间，不改报告日期。"
        )
        * 2,
        "evidence": [{"source_ref": "actual-report", "quote": Source().body("actual-report")}],
    }
    validate_period(values, Source())
    changed = deepcopy(values)
    changed["monthly_period_review"]["evidence"][0]["quote"] += "伪造"
    with pytest.raises(ValueError, match="ORIGINAL_QUOTE_MISMATCH"):
        validate_period(changed, Source())


def test_changing_only_old_period_date_cannot_pass_new_month():
    values = {"fund_code": "019449", "as_of_date": "2026-09-30", "holding_date": "2026-06-30"}
    validate_period(values, Source())
    values["as_of_date"] = "2026-10-30"
    with pytest.raises(ValueError, match="CURRENT_PERIOD_REVIEW_REQUIRED"):
        validate_period(values, Source())


def test_reviewed_latest_half_year_is_not_rejected_by_an_extra_age_cutoff():
    values = {"fund_code": "019449", "as_of_date": "2026-02-27", "holding_date": "2025-06-30"}
    with pytest.raises(ValueError, match="CURRENT_PERIOD_REVIEW_REQUIRED"):
        validate_period(values, Source())
    values["monthly_period_review"] = {
        **values,
        "reason": (
            "本期逐项核对真实披露目录、最新完整报告和适用份额；年度报告尚未取得，"
            "继续明确使用已披露中报的真实持仓日期，不把旧持仓写成当前全仓。"
        ) * 2,
        "evidence": [{"source_ref": "actual-report", "quote": Source().body("actual-report")}],
    }
    validate_period(values, Source())
