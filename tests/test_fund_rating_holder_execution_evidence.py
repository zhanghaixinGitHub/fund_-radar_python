"""托管报告同等级事实的版式适配；不借另一基金原文、不把一般承诺当实际履职。"""

from copy import deepcopy

import pytest
from app.services.fund_rating_direct_index import holder_execution_evidence
from app.services.fund_rating_market_review import normalized_text

TRUSTEE = ("5.1报告期内本基金托管人遵规守信情况声明。5.2托管人对报告期内本基金投资运作情况："
           "基金份额申购赎回价格计算，托管人未发现损害基金份额持有人利益的行为。"
           "5.3托管人对本中期报告进行了核查，以上内容真实、准确和完整。")
OPERATION = ("报告期内，本基金严格遵守合同要求，认真对待投资者日常申购、赎回及成份股调整等工作，"
             "运作过程中未发生任何风险事件。")


class Source:
    def claim(self, claim):
        if normalized_text(claim["quote"]) not in normalized_text(TRUSTEE + OPERATION):
            raise ValueError("ORIGINAL_QUOTE_MISMATCH")


def review():
    return {"execution_profile": "TRUSTEE_NAV_REVIEW_AND_ACTUAL_REDEMPTION_OPERATION",
            "execution": {"source_hash": "own", "quote": TRUSTEE},
            "actual_operation": {"source_hash": "own", "quote": OPERATION}}


def test_same_report_actual_trustee_and_redemption_delivery_are_required_together():
    holder_execution_evidence(Source(), review(), "own")


@pytest.mark.parametrize("damage", [
    "other_fund", "missing_operation", "promise_only", "missing_nav_review", "unknown_profile",
])
def test_weaker_or_cross_product_evidence_cannot_mark_holder_execution_closed(damage):
    value = deepcopy(review())
    if damage == "other_fund":
        value["execution"]["source_hash"] = "another-product"
    elif damage == "missing_operation":
        value.pop("actual_operation")
    elif damage == "promise_only":
        value["actual_operation"]["quote"] = "本基金严格遵守合同要求"
    elif damage == "missing_nav_review":
        value["execution"]["quote"] = "5.1报告期内本基金托管人遵规守信情况声明"
    else:
        value["execution_profile"] = "ASSUME_NORMAL"
    with pytest.raises(ValueError):
        holder_execution_evidence(Source(), value, "own")


def test_legacy_template_still_requires_its_supervision_and_review_original():
    value = review()
    value.pop("execution_profile")
    with pytest.raises(ValueError, match="ACTUAL_HOLDER_EXECUTION_REQUIRED"):
        holder_execution_evidence(Source(), value, "own")
