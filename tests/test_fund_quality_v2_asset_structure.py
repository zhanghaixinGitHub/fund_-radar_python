"""新结构口径的原件关联及拒收边界；所有临时文本都不是校准产品。"""

import json
from decimal import Decimal, localcontext

import pytest
from app.services.fund_quality_v2_asset_structure import capco_publication, gross_asset_book, half_l1_structure
from app.services.fund_quality_v2_contract import metric_contract


def article(**updates):
    """仅构造协会发布正文的格式算例，不使用真实产品标识。"""
    data = {
        "title": "2026年上半年上市公司行业分类结果",
        "source": "中国上市公司协会",
        "date": "2026-09-30",
        "text": '<a href="https://example.invalid/sample.pdf\r\n">2026年上半年上市公司行业分类结果（按股票代码排序）.pdf</a>',
    }
    data.update(updates)
    return ("export const allData = " + json.dumps(data, ensure_ascii=False)).encode()


def test_article_verifies_actual_attachment_and_keeps_publication_separate_from_observation():
    result = capco_publication(article(), attachment_url="https://example.invalid/sample.pdf")
    assert result["period_end"] == "2026-06-30"
    assert result["published_on"] == "2026-09-30"
    with pytest.raises(ValueError, match="NOT_LINKED"):
        capco_publication(article(), attachment_url="https://example.invalid/replaced.pdf")


@pytest.mark.parametrize("problem", ["title", "source", "date", "script"])
def test_metadata_cannot_replace_original_period_or_execute_foreign_script(problem):
    raw = article(**{problem: "错误字段"}) if problem != "script" else article() + b"; run_foreign_script()"
    with pytest.raises(ValueError):
        capco_publication(raw, attachment_url="https://example.invalid/sample.pdf")


def test_contract_policy_is_frozen_against_callers_mutating_returned_objects():
    definition = metric_contract()
    definition["industry_structure_policy"]["denominator"] = "改成净资产的测试篡改"
    assert metric_contract()["industry_structure_policy"]["denominator"].startswith("资产负债表资产总计")


def test_missing_full_asset_statement_is_rejected_instead_of_zero_filled():
    with pytest.raises(ValueError, match="BALANCE_PROOF_CURRENT_CNY_TWO_COLUMN_BODY_UNRESOLVED"):
        gross_asset_book(["本报告期自2026年1月1日起至6月30日止。测试资料没有完整资产负债表。"])


def test_repeated_structure_sum_is_invariant_to_category_order_and_caller_decimal_precision():
    actual = {"EQUITY:J": Decimal("0.90000000000000000000000000001"), "ASSET:货币资金": Decimal("0.10")}
    target = {"EQUITY:J": Decimal("0.95"), "ASSET:货币资金": Decimal("0.05")}
    expected = half_l1_structure(actual, target)
    with localcontext() as context:
        context.prec = 6
        reordered = half_l1_structure(dict(reversed(list(actual.items()))), dict(reversed(list(target.items()))))
    assert reordered == expected
    assert expected[0] == "0.049999999999999999999999999995"
