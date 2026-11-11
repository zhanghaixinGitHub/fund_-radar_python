"""托管报告同等级事实的版式适配；不借另一基金原文、不把一般承诺当实际履职。"""

from copy import deepcopy

import pytest
from app.services.fund_rating_direct_index import holder_execution_evidence
from app.services.fund_rating_market_review import normalized_text

TRUSTEE = (
    "5.1报告期内本基金托管人遵规守信情况声明。5.2托管人对报告期内本基金投资运作情况："
    "基金份额申购赎回价格计算，托管人未发现损害基金份额持有人利益的行为。"
    "5.3托管人对本中期报告进行了核查，以上内容真实、准确和完整。"
)
OPERATION = (
    "报告期内，本基金严格遵守合同要求，认真对待投资者日常申购、赎回及成份股调整等工作，运作过程中未发生任何风险事件。"
)
PRICING_OPERATION = (
    "4.2管理人对报告期内本基金运作遵规守信情况的说明本基金管理人严格遵守基金合同，没有损害基金持有人利益的行为。"
)
ALTERNATE_PRICING_OPERATION = (
    "4.2管理人对报告期内本基金运作遵规守信情况的说明本报告期内，本基金管理人严格遵守"
    "《证券投资基金法》等有关法律法规的规定及基金合同，基金运作合法合规，无损害基金份额持有人利益的行为。"
)
RECONCILIATION = "报告期内对托管产品的投资行为进行监督，进行会计核算和资产估值并与管理人建立对账机制。"
QUARTER_REVIEW = "复核了本报告中的财务指标、净值表现和投资组合报告等内容，保证复核内容不存在虚假记载。"


class Source:
    def claim(self, claim):
        if normalized_text(claim["quote"]) not in normalized_text(
            TRUSTEE
            + OPERATION
            + PRICING_OPERATION
            + ALTERNATE_PRICING_OPERATION
            + ALTERNATE_PRICING_OPERATION.replace("无损害基金份额持有人", "无损害基金持有人")
            + RECONCILIATION
            + QUARTER_REVIEW
        ):
            raise ValueError("ORIGINAL_QUOTE_MISMATCH")


def review():
    return {
        "execution_profile": "TRUSTEE_NAV_REVIEW_AND_ACTUAL_REDEMPTION_OPERATION",
        "execution": {"source_hash": "own", "quote": TRUSTEE},
        "actual_operation": {"source_hash": "own", "quote": OPERATION},
    }


def test_same_report_actual_trustee_and_redemption_delivery_are_required_together():
    holder_execution_evidence(Source(), review(), "own")


@pytest.mark.parametrize(
    "damage",
    [
        "other_fund",
        "missing_operation",
        "promise_only",
        "missing_nav_review",
        "unknown_profile",
    ],
)
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


def test_trustee_pricing_and_full_report_review_template():
    value = review()
    value["execution_profile"] = "TRUSTEE_REDEMPTION_PRICING_AND_REPORT_REVIEW"
    value["actual_operation"]["quote"] = (
        "4.2管理人对报告期内本基金运作遵规守信情况的说明本基金管理人严格遵守基金合同，没有损害基金持有人利益的行为。"
    )

    holder_execution_evidence(Source(), value, "own")


def test_trustee_pricing_template_accepts_equivalent_contract_and_no_damage_wording():
    value = review()
    value["execution_profile"] = "TRUSTEE_REDEMPTION_PRICING_AND_REPORT_REVIEW"
    value["actual_operation"]["quote"] = ALTERNATE_PRICING_OPERATION

    holder_execution_evidence(Source(), value, "own")
    value["actual_operation"]["quote"] = ALTERNATE_PRICING_OPERATION.replace("无损害基金份额持有人", "无损害基金持有人")
    holder_execution_evidence(Source(), value, "own")


def test_trustee_supervision_and_reconciliation_template():
    value = review()
    value["execution_profile"] = "TRUSTEE_SUPERVISION_AND_RECONCILIATION"
    value["execution"]["quote"] = "报告期内对托管产品的投资行为进行监督，进行会计核算和资产估值并与管理人建立对账机制。"
    value.pop("actual_operation")

    holder_execution_evidence(Source(), value, "own")


def test_trustee_quarter_report_review_template():
    value = review()
    value["execution_profile"] = "TRUSTEE_QUARTER_REPORT_REVIEW"
    value["execution"]["quote"] = "复核了本报告中的财务指标、净值表现和投资组合报告等内容，保证复核内容不存在虚假记载。"
    value.pop("actual_operation")

    holder_execution_evidence(Source(), value, "own")
