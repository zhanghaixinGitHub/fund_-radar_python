"""现金约束可测量性研究的关键反例；这些合成数据不进入真实产品校准。"""
from decimal import localcontext

import pytest
from app.services.fund_quality_v2_cash_constraints import (
    _basket_rule,
    deposit_noninterest_book_bound,
    explicit_margin_absence,
    shortfall_bounds,
    validate_deposit_additions,
)


def bounds(**overrides):
    return shortfall_bounds(**({'floor': '0.05', 'cash_lower': '90', 'cash_upper': '100',
                               'government_upper': '40', 'net_assets': '1000',
                               'margin_absence_excerpt': '本基金本报告期末未投资股指期货。'} | overrides))


def test_known_cash_can_identify_shortfall_without_filling_government_bonds():
    result = bounds()
    assert result['identified_shortfall_ratio'] == '0'
    assert result['government_amount_point_identified_by_bound'] is None
    assert result['government_amount_upper_bound'] == '40'
    assert result['state'] == 'POINT_IDENTIFIED_FOR_REPORT_DATE_NOT_DAILY'


def test_unknown_margin_does_not_become_zero_even_when_cash_is_abundant():
    result = bounds(margin_absence_excerpt=None)
    assert result['identified_shortfall_ratio'] is None
    assert result['shortfall_ratio_lower'] == '0'
    assert result['shortfall_ratio_upper'] is None
    assert result['available_amount_lower_bound'] is None
    assert result['unknown_margin_requirement']


def test_unknown_margin_with_empty_cash_basket_still_has_no_finite_shortfall_upper():
    result = bounds(cash_lower='0', cash_upper='0', government_upper='0', margin_absence_excerpt=None)
    assert result['shortfall_ratio_lower'] == '0.05'
    assert result['shortfall_ratio_upper'] is None
    assert result['identified_shortfall_ratio'] is None


def test_partial_cash_and_unknown_government_maturity_leave_interval():
    result = bounds(cash_lower='20', cash_upper='30', government_upper='100')
    assert result['shortfall_ratio_lower'] == '0'
    assert result['shortfall_ratio_upper'] == '0.03'
    assert result['identified_shortfall_ratio'] is None


def test_no_bonds_and_unique_cash_identify_positive_point_shortfall():
    result = bounds(cash_lower='20', cash_upper='20', government_upper='0')
    assert result['government_amount_point_identified_by_bound'] == '0'
    assert result['identified_shortfall_ratio'] == '0.03'


@pytest.mark.parametrize('text', [True, '本报告期末未持有衍生金融资产/负债。', '本基金可投资股指期货。'])
def test_flags_balance_absence_and_future_policy_cannot_prove_no_margin(text):
    with pytest.raises((ValueError, TypeError)):
        bounds(margin_absence_excerpt=text)


@pytest.mark.parametrize('overrides', [{'floor': '0'}, {'cash_lower': '110'},
                                     {'net_assets': '0'}, {'government_upper': '-1'}])
def test_invalid_or_inverted_bounds_rejected(overrides):
    with pytest.raises(ValueError, match='BOUND_DOMAIN'):
        bounds(**overrides)


def test_decimal_context_does_not_change_bound_output():
    with localcontext() as context:
        context.prec = 8
        first = bounds(net_assets='997.123456789')
    with localcontext() as context:
        context.prec = 50
        second = bounds(net_assets='997.123456789')
    assert first == second


@pytest.mark.parametrize('total,expected', [('80.00', '79.00'), ('76.00', '75.00')])
def test_unassigned_credit_loss_stays_unknown_and_book_residual_is_not_imputed_loss(total, expected):
    result = deposit_noninterest_book_bound(total=total, principal='79.00', interest='1.00', provision=None)
    assert result['provision_observed'] is None
    assert result['provision_state'] == 'UNASSIGNED_NOT_ZERO'
    assert result['noninterest_book_lower_bound'] == expected


def test_only_printed_zero_provision_is_an_explicit_zero():
    result = deposit_noninterest_book_bound(total='80.00', principal='79.00', interest='1.00', provision='0.00')
    assert result['provision_observed'] == '0.00' and result['provision_state'] == 'EXPLICIT_AMOUNT'
    with pytest.raises(ValueError, match='EXPLICIT_LOSS_NOT_RECONCILED'):
        deposit_noninterest_book_bound(total='80.00', principal='79.00', interest='1.00', provision='3.00')


@pytest.mark.parametrize('cell', ['15.00', '-', ''])
def test_unknown_positive_row_cannot_hide_principal_impairment_in_net_book_bound(cell):
    text = '活期存款 100.00\n等于：本金 100.00\n加：应计利息 5.00\n减：坏账准备 -\n加：其他调整 ' + cell
    with pytest.raises(ValueError, match='UNREVIEWED_AMOUNT_ROW'):
        validate_deposit_additions(text)
    with pytest.raises(ValueError, match='UNEXPLAINED_POSITIVE_ADDITION'):
        deposit_noninterest_book_bound(total='110.00', principal='100.00', interest='5.00', provision=None)


@pytest.mark.parametrize('statement', ['本基金本报告期内未投资股指期货。', '本基金在本报告期未投资股指期货。',
                                       '本基金本报告期末未持有股指期货合约。'])
def test_futures_absence_requires_original_observed_period_statement(statement):
    assert explicit_margin_absence(statement) == statement


@pytest.mark.parametrize('statement', ['本基金本报告期未进行股指期货交易。', '本基金不投资股指期货。',
                                       '本基金未来不持有股指期货。', '衍生金融资产 - -'])
def test_no_trades_policy_and_balance_dash_do_not_prove_no_margin(statement):
    assert explicit_margin_absence(statement) is None


def test_futures_presence_conflict_is_not_hidden_by_one_absence_sentence():
    with pytest.raises(ValueError, match='SOURCE_CONFLICT'):
        explicit_margin_absence('本基金本报告期末未持有股指期货。本基金本报告期末持有股指期货合约。')


@pytest.mark.parametrize('clause', [
    '每个交易日日终在扣除股指期货需缴纳的交易保证金后，应当保持现金或者到期日在一年以内的政府债券不低于基金资产净值的5%，其中，现金不包括结算备付金、存出保证金、应收申购款等',
    '每个交易日日终在扣除股指期货合约需缴纳的交易保证金后，应当保持不低于基金资产净值5%的现金或到期日在一年以内的政府债券，其中，现金不包括结算备付金、存出保证金、应收申购款等',
])
def test_same_fund_basket_clause_word_order_does_not_create_a_pure_cash_target(clause):
    floor, excerpt = _basket_rule('本基金的投资组合比例为：' + clause)
    assert str(floor) == '0.05' and excerpt == clause
    with pytest.raises(ValueError, match='CONSTRAINT_MISSING_OR_CONFLICTING'):
        _basket_rule('本基金所投资的目标基金：' + clause)


AND_CLAUSE = ('每个交易日日终在扣除股指期货合约需缴纳的交易保证金后，'
              '应当保持不低于基金资产净值5%的现金或到期日在一年以内的政府债券，'
              '其中现金不包括结算备付金、存出保证金和应收申购款等')


def test_explicit_exclusion_list_and_connector_preserves_all_three_items():
    floor, excerpt = _basket_rule('本基金的投资组合比例：' + AND_CLAUSE)
    assert str(floor) == '0.05' and excerpt == AND_CLAUSE


def test_link_fund_own_cash_clause_is_not_excluded_by_its_target_holding_sentence():
    text = '本基金投资于目标ETF的比例不低于基金资产净值的90%；本基金' + AND_CLAUSE
    floor, excerpt = _basket_rule(text)
    assert str(floor) == '0.05' and excerpt == AND_CLAUSE


def test_target_etf_quoted_self_reference_does_not_prove_link_fund_cash_rule():
    with pytest.raises(ValueError, match='CONSTRAINT_MISSING_OR_CONFLICTING'):
        _basket_rule('本基金所投资的目标ETF的投资范围：本基金' + AND_CLAUSE)


def test_all_margin_obligation_is_not_reduced_to_index_futures_margin():
    text = '本基金的投资组合比例：' + AND_CLAUSE.replace('股指期货合约需缴纳', '全部需缴纳')
    with pytest.raises(ValueError, match='CONSTRAINT_MISSING_OR_CONFLICTING'):
        _basket_rule(text)
