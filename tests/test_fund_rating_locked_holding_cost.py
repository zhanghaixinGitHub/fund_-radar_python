"""验证锁定期原件、金额口径以及正式重放拒绝被改动的冻结费用。"""

from copy import deepcopy
from decimal import Decimal

import pytest
from app.services.fund_rating_locked_holding_cost import TERMS, calculate, replay

NAME = "大成匠心卓越三年持有期混合型证券投资基金"
SUMMARY = (
    NAME
    + "013853"
    + TERMS[-1]
    + "三、投资本基金涉及的费用申购费(前收费)M<100万元1.50%"
    + "大成匠心卓越三年持有混合A基金运作综合费率(年化)1.41%"
)
CONTRACT = "。".join(TERMS)


class Source:
    def pages(self, key):
        return [{"summary": SUMMARY, "prospectus": CONTRACT}[key]]


def inputs():
    result = calculate(SUMMARY, CONTRACT, code="013853", legal_name=NAME, share_class="A")
    return {
        "fund_code": "013853",
        "legal_name": NAME,
        "share_class": "A",
        "target_refs": {"summary": "summary", "prospectus": "prospectus"},
        "reviewed_cost": {
            "source_hash": "summary",
            "scenario": result,
            "score": result["score"],
            "annual_cost_per_10000": result["annual_cost_per_10000"],
        },
    }


def test_three_year_cost_uses_three_year_principal_and_has_no_first_year_exit():
    result = replay(inputs(), Source())
    # 金额独立核对至分：不把三年总费用当作一年费用，也不当成365天可赎回。
    assert Decimal(result["total_cost_per_10000"]).quantize(Decimal(".01")) == Decimal("558.68")
    assert Decimal(result["annual_cost_per_10000"]).quantize(Decimal(".01")) == Decimal("186.23")
    assert result["redeemable_at_365_days"] is False
    assert result["minimum_calendar_years"] == 3


@pytest.mark.parametrize("key,value", [("fund_code", "014978"), ("share_class", "C")])
def test_other_fund_or_share_cannot_use_reviewed_cost(key, value):
    value_inputs = inputs()
    value_inputs[key] = value
    with pytest.raises(ValueError, match="IDENTITY"):
        replay(value_inputs, Source())


@pytest.mark.parametrize("term", TERMS)
def test_missing_lock_or_exit_evidence_rejects_fee(term):
    with pytest.raises(ValueError, match="TERMS_MISSING"):
        calculate(SUMMARY, CONTRACT.replace(term, ""), code="013853", legal_name=NAME, share_class="A")


@pytest.mark.parametrize("field", ["score", "annual_cost_per_10000", "annual_operating_rate"])
def test_frozen_fee_tampering_is_rejected(field):
    value_inputs = deepcopy(inputs())
    value_inputs["reviewed_cost"]["scenario"][field] = "0"
    with pytest.raises(ValueError, match="FROZEN_CONFLICT"):
        replay(value_inputs, Source())


def test_submitted_score_and_claimed_365_day_redemption_are_rejected():
    value_inputs = inputs()
    value_inputs["reviewed_cost"]["score"] = "100"
    with pytest.raises(ValueError, match="REVIEWED_CONFLICT"):
        replay(value_inputs, Source())
    value_inputs = inputs()
    value_inputs["reviewed_cost"]["scenario"]["redeemable_at_365_days"] = True
    with pytest.raises(ValueError, match="REDEMPTION_CONFLICT"):
        replay(value_inputs, Source())


def test_duplicate_rate_table_is_rejected_instead_of_choosing_first():
    with pytest.raises(ValueError, match="RATES_NOT_UNIQUE"):
        calculate(SUMMARY + "M<100万元0.50%", CONTRACT, code="013853", legal_name=NAME, share_class="A")


def test_original_fee_label_wrapped_around_other_columns_still_binds_front_fee():
    wrapped = SUMMARY.replace("申购费(前收费)", "申购费100万元≤M<300万元1.2%(前收费)")
    result = calculate(wrapped, CONTRACT, code="013853", legal_name=NAME, share_class="A")
    assert result["subscription_rate"] == "0.015"
    with pytest.raises(ValueError, match="RATES_NOT_UNIQUE"):
        calculate(wrapped.replace("(前收费)", "(后收费)"), CONTRACT, code="013853", legal_name=NAME, share_class="A")
