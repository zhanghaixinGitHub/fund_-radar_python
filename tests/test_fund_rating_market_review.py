"""市场逐只复核的费用份额、计费本金和缺项边界。"""

from decimal import Decimal

import pytest
from app.services.fund_rating_market_review import annual_fee_score, fee_profile, weighted_interval


def test_c_share_currency_qualified_free_subscription_needs_own_currency_statement():
    text = ('隔离基金(C类人民币份额)基金代码123451送出日期2026年8月11日'
            '三、投资本基金涉及的费用赎回费N≥30天0.00%'
            '注:本基金C类人民币份额在投资者申购时不收取申购费用。'
            '(二)运作费用基金运作综合费率(年化)1.55%')
    result = fee_profile(text, code='123451', share_class='C', legal_name='隔离基金')
    assert result['fees']['subscription'] == '0'
    alternate = text.replace('(C类人民币份额)', 'C(人民币份额)交易币种人民币')
    assert fee_profile(alternate, code='123451', share_class='C', legal_name='隔离基金')['fees']['subscription'] == '0'
    with pytest.raises(ValueError, match='免申购费'):
        fee_profile(text.replace('注:本基金C类人民币', '注:本基金C类美元'),
                    code='123451', share_class='C', legal_name='隔离基金')


def test_c_share_reverse_fee_definition_needs_complete_same_class_clause():
    text = ("隔离基金C123451交易币种人民币送出日期2026年9月18日"
            "三、投资本基金涉及的费用赎回费N≥7天0.00%"
            "(二)基金运作综合费率(年化)1.00%")
    support = ("在投资者认购或申购时不收取认购费或申购费，"
               "而从本类别基金资产中计提销售服务费的，称为C类基金份额。")
    parsed = fee_profile(text, code="123451", share_class="C", legal_name="隔离基金", supporting_text=support)
    assert Decimal(annual_fee_score(parsed["fees"])["score"]) == Decimal("75")
    for wrong in ["", support.replace("称为C类", "称为A类"), support.replace("不收取", "收取")]:
        with pytest.raises(ValueError, match="免申购费"):
            fee_profile(text, code="123451", share_class="C", legal_name="隔离基金", supporting_text=wrong)


def test_ordinary_share_retains_identity_and_real_nonzero_terminal_redemption_rate():
    text = ('隔离全球基金基金代码000099下属基金代码000099运作方式普通开放式'
            '送出日期2026年7月27日三、投资本基金涉及的费用申购费(前收费)M<100万元1.6%'
            '赎回费N<7天1.50%-N>=7天0.50%-(二)基金运作综合费率(年化)1.42%')
    result = fee_profile(text, code='000099', share_class='普通', legal_name='隔离全球基金')
    assert result['fees']['subscription'] == '0.016' and result['fees']['redemption'] == '0.005'
    for wrong in [text.replace('下属基金代码000099', '下属基金代码000098'),
                  text.replace('运作方式普通开放式', '运作方式封闭式'),
                  text.replace('N>=7天0.50%', 'N>=730天0.50%')]:
        with pytest.raises(ValueError):
            fee_profile(wrong, code='000099', share_class='普通', legal_name='隔离全球基金')


def test_nonzero_one_year_redemption_uses_same_product_off_exchange_table_and_explicit_day_definition():
    text = ('隔离基金(A类份额)基金代码123450送出日期2025年10月22日'
            '三、投资本基金涉及的费用申购费(前收费)M<100万元1.2%'
            '赎回费7天≤N<365天0.5%场外1年≤N<2年0.25%场外2年≤N0%场外'
            '7天≤N0.5%场内(二)运作费用基金运作综合费率(年化)1.10%')
    support = ('其中1年指365天A类份额的场外赎回费7日≤T<1年0.50%1年≤T<2年0.25%'
               '本基金A类份额的场内赎回费T≥7日0.5%')
    result = fee_profile(text, code='123450', share_class='A', legal_name='隔离基金', supporting_text=support)
    assert result['fees']['redemption'] == '0.0025'
    for altered in [support.replace('1年指365天', '按年计算'),
                    support.replace('1年≤T<2年0.25%', '1年≤T<2年0.50%'),
                    support.replace('A类份额的场外', 'C类份额的场外')]:
        with pytest.raises(ValueError, match='365天'):
            fee_profile(text, code='123450', share_class='A', legal_name='隔离基金', supporting_text=altered)


def test_fee_cost_uses_actual_invested_principal_and_redemption_base():
    result = annual_fee_score(
        {"subscription": ".01", "redemption": ".005", "operating_cost": ".012", "subscription_mode": "EXTERNAL"}
    )
    invested = 10000 / 1.01
    expected = 10000 - invested + invested * 0.012 + invested * (1 - 0.012) * 0.005
    assert abs(float(result["cost_cny"]) - expected) < 1e-9
    assert abs(float(result["score"]) - (100 - expected / 4)) < 1e-9


def test_fee_missing_or_non_finite_is_never_assigned_zero():
    for value in ["NaN", "Infinity", "-0.1", "1.1"]:
        with pytest.raises(ValueError):
            annual_fee_score(
                {"subscription": "0", "redemption": "0", "operating_cost": value, "subscription_mode": "EXTERNAL"}
            )
    with pytest.raises(KeyError):
        annual_fee_score({"subscription": "0", "redemption": "0", "subscription_mode": "EXTERNAL"})


def test_fee_c_share_uses_own_table_instead_of_first_a_rate():
    text = """隔离基金 基金代码A123450 基金代码C123451 送出日期2026年6月26日
    三、投资本基金涉及的费用
    隔离基金A费用类型 N≥365天0.00% 隔离基金C费用类型 N≥30天0.00%
    本基金C类份额无申购费。
    （二）基金运作相关费用
    隔离基金A基金运作综合费率（年化）1.20%
    隔离基金C基金运作综合费率（年化）1.60%"""
    result = fee_profile(text, code="123451", share_class="C", legal_name="隔离基金")
    assert Decimal(result["fees"]["operating_cost"]) == Decimal(".016")
    assert annual_fee_score(result["fees"])["score"] == "60.000"
    with pytest.raises(ValueError, match="身份"):
        fee_profile(text, code="123452", share_class="C", legal_name="隔离基金")


def test_comprehensive_cost_repeated_row_label_keeps_unique_class_and_never_adds_components_twice():
    text = """隔离基金（C类人民币份额）123451 交易币种人民币 送出日期2026年6月26日
    三、投资本基金涉及的费用 本基金C类基金份额不收取申购费 N≥7天0.00%
    （二）运作费用 管理费0.80% 托管费0.15% 销售服务费0.30%
    基金运作综合费率（年化）基金运作综合费率1.33%"""
    parsed = fee_profile(text, code="123451", share_class="C", legal_name="隔离基金")
    assert Decimal(parsed["fees"]["operating_cost"]) == Decimal(".0133")
    assert Decimal(annual_fee_score(parsed["fees"])["score"]) == Decimal("66.75")
    ambiguous = text + "基金运作综合费率（年化）基金运作综合费率1.45%"
    with pytest.raises(ValueError, match="尚未唯一定位"):
        fee_profile(ambiguous, code="123451", share_class="C", legal_name="隔离基金")


def test_missing_redemption_or_parallel_rate_is_not_guessed():
    text = """隔离基金（C类份额）123451 送出日期2026年6月26日
    三、投资本基金涉及的费用 不收取申购费 N<7天1.50%
    基金运作综合费率（年化）1.60%"""
    with pytest.raises(ValueError, match="赎回"):
        fee_profile(text, code="123451", share_class="C", legal_name="隔离基金")


def test_actual_bare_zero_fee_cell_uses_verified_publication_not_compilation_date():
    text = """隔离基金（C份额）123451 编制日期2026年6月9日
    三、投资本基金涉及的费用 不收取申购费 N<7天1.50% N≥7天 0
    （二）基金运作相关费用 基金运作综合费率（年化）1.34%"""
    parsed = fee_profile(text, code="123451", share_class="C", legal_name="隔离基金", published_on="2026-06-12")
    assert parsed["published_on"] == "2026-06-12" and parsed["fees"]["redemption"] == "0"
    with pytest.raises(ValueError, match="赎回"):
        fee_profile(text.replace("N≥7天 0", "N≥7天 0.5"), code="123451", share_class="C",
                    legal_name="隔离基金", published_on="2026-06-12")


def test_a_share_composite_cost_does_not_require_duplicate_zero_sales_fee_gate():
    text = """隔离基金（A类份额）123450 送出日期2026年6月26日
    三、投资本基金涉及的费用 申购费（前收费）M<100万元1.50% 赎回费180天≤N0%
    （二）基金运作相关费用 基金运作综合费率（年化）-1.43%"""
    result = fee_profile(text, code="123450", share_class="A", legal_name="隔离基金")
    assert result["fees"]["subscription"] == "0.015"
    assert result["fees"]["operating_cost"] == "0.0143"


def test_known_subitem_keeps_original_weight_and_unknown_domain():
    result = weighted_interval(
        {"capacity": None, "restricted": ("100", "100"), "payment": None, "holders": None},
        {"capacity": 40, "restricted": 20, "payment": 25, "holders": 15},
    )
    assert result == {"lower": "20", "upper": "100", "candidate": None}
