"""招商日基准实际代码、原精度、份额净值与完整日历不能被替换或填补。"""

from copy import deepcopy

import pytest
from app.services.fund_rating_cmf_benchmark import compile_history


@pytest.fixture
def original():
    days = ['2026-09-28', '2026-09-29', '2026-09-30']
    expression = '中证隔离指数收益率×95%+中国人民银行人民币活期存款利率（税后）×5%'
    raw = {'protocol': {'fund_code': '999999', 'as_of': days[-1], 'index_code': 'fixture.CSI',
                       'frozen_at': '2026-10-01', 'issuer_percent_decimal_places': 2,
                       'issuer_level_half_quantum': '.00005', 'index_close_half_quantum': '.00005'},
           'index': {'code': 0, 'data': {'fields': ['ts_code', 'trade_date', 'close'],
                                        'items': [['fixture.CSI', d, '100'] for d in days]}},
           'benchmark': {'code': 0, 'data': [{'date': d, 'indexName': expression if i == 0 else '',
                                            'navYield': '0.00', 'hs300Yield': '0.00'} for i, d in enumerate(days)]}}
    refs = {'benchmark': {'url': 'https://www.cmfchina.com/ws-business-server/fund/getIncomeList',
                         'params': {'productCode': '999999', 'startDate': days[0], 'endDate': days[-1]}}}

    class Source:
        sources = refs

        def json(self, key):
            return raw[key]

        def pages(self, _key):
            return ['业绩比较基准' + expression + '。']

        def plain(self, _key):
            return '活期存款0.350.35'

    inputs = {'fund_code': '999999', 'as_of_date': days[-1], 'verified_at': '2026-10-02',
              'index': {'code': 'fixture.CSI'},
              'target_refs': {'benchmark': 'benchmark', 'summary': 's', 'prospectus': 'p'},
              'cmf_benchmark_reconstruction': {'protocol_hash': 'protocol', 'annual_cash_rate': '.0035',
                                               'rate_source': 'rate', 'index_daily': 'index'}}
    history = {'dates': days, 'rows': [{'nav_date': d, 'adjusted_nav': '1'} for d in days]}
    return Source(), inputs, history, raw, refs


def test_cmf_observations_remain_distinct_from_contract_constructed_reference(original):
    source, inputs, history, _raw, _refs = original
    result = compile_history(source, inputs, deepcopy(history))
    assert result['publisher_cross_check']['published_valuation_dates'] == 3
    assert 'NOT_STITCHED' in result['publisher_cross_check']['scope']


@pytest.mark.parametrize('damage,error', [('product', 'REQUEST_OR_RESPONSE'), ('nav', 'OWN_SHARE_NAV'),
    ('benchmark', 'PRECISION_CONFLICT'), ('precision', 'PUBLISHED_PRECISION'), ('missing', 'DAILY_COVERAGE')])
def test_cmf_conflicting_or_missing_original_point_is_rejected(original, damage, error):
    source, inputs, history, raw, refs = original
    if damage == 'product':
        refs['benchmark']['params']['productCode'] = '999998'
    elif damage == 'nav':
        raw['benchmark']['data'][1]['navYield'] = '1.00'
    elif damage == 'benchmark':
        raw['benchmark']['data'][1]['hs300Yield'] = '1.00'
    elif damage == 'precision':
        raw['benchmark']['data'][1]['hs300Yield'] = '0.001'
    else:
        raw['benchmark']['data'].pop(1)
    with pytest.raises(ValueError, match=error):
        compile_history(source, inputs, deepcopy(history))
