"""类别适配的人工边界；这些文字不是原件、校准产品或独立验证样本。"""

import pytest
from app.services.fund_quality_v2_classification import classify_original as classify


@pytest.mark.parametrize("title,expected", [
    ("测试股票型基金(QDII)", "GLOBAL_ACTIVE_EQUITY"), ("测试股票型基金(qdii)", "GLOBAL_ACTIVE_EQUITY"),
    ("测试指数研究增强型基金", "INDEX_ENHANCED"), ("测试指数精选增强型基金", "INDEX_ENHANCED"),
    ("测试指数增强型基金", "INDEX_ENHANCED"), ("测试增强指数基金", "INDEX_ENHANCED"),
    ("测试量化指数型基金", "PASSIVE_INDEX"), ("测试沪港深股票型基金", "ACTIVE_EQUITY"),
    ("测试纳斯达克股票指数基金(QDII)", None), ("测试黄金ETF", None), ("测试股票基金中基金", None),
    ("测试全球股票型基金", None),
])
def test_title_scope(title, expected):
    assert classify(title)["kind"] == expected


def test_foreign_evidence_must_be_in_formal_product_section_not_manager_biography():
    section = ("2.2基金产品说明投资目标本基金主要投资于美国大盘成长型股票。投资策略采用主动量化投资策略。"
               "风险收益特征本基金主要投资于美国大盘成长型股票，属于主动投资的股票型基金。2.3基金管理人")
    result = classify("测试美国成长股票型基金", section)
    assert result["kind"] == "GLOBAL_ACTIVE_EQUITY" and result["rule"] == "FORMAL_PRODUCT_OBJECTIVE_AND_FOREIGN_SCOPE"
    assert classify("测试美国成长股票型基金", section.replace("2.2基金产品说明", "4.1基金经理履历"))["kind"] is None
    biography = "经理曾任QDII基金经理，投资美国股票。4.1基金经理履历"
    assert classify("测试国内股票型基金", biography)["kind"] == "ACTIVE_EQUITY"
    assert classify("测试美国股票型基金", section.replace("风险收益特征", "其他说明"))["kind"] is None


def test_global_objective_with_formal_foreign_custodian_and_active_scope():
    text = ("2.2基金产品说明投资目标主要通过在全球范围内进行积极的股票投资。投资策略精选个股。"
            "风险收益特征本基金为全球证券投资基金。2.3基金管理人2.4境外资产托管人")
    assert classify("测试全球精选股票型基金", text)["kind"] == "GLOBAL_ACTIVE_EQUITY"
    assert classify("测试全球指数股票型基金", text)["kind"] is None
