"""广发累计曲线的原份额、原精度和全日历校验；假设样例不写真实评分。"""

from copy import deepcopy

import pytest
from app.services.fund_rating_gf_benchmark import compile_history


@pytest.fixture
def fixture():
    days = ['2026-09-28', '2026-09-29', '2026-09-30']
    expression = '隔离指数收益率×95%+活期存款利率（税后）×5%'
    raw = {'protocol': {'fund_code': '999999', 'as_of': days[-1], 'index_code': 'fixture.SZ',
                       'candidate_cash_rate': '.0035', 'frozen_at': '2026-10-01',
                       'publisher_level_half_quantum': '.00005', 'index_close_half_quantum': '.00005'},
           'index': {'code': 0, 'data': {'fields': ['ts_code', 'trade_date', 'close'],
                                        'items': [['fixture.SZ', d, '100'] for d in days]}},
           'benchmark': {'CREATE': [{'FUNDCODE': '999999', 'NAVDATE': d.replace('-', ''),
                'IDXTYPE': 'yield_his', 'YIELD': '0.00%', 'YIELD_BENCHMARK': '-50.00%'} for d in days]}}

    class Source:
        sources = {'benchmark': {'url': 'https://www.gffunds.com.cn/api/v1/funds/yield-history.shtml?fundcode=999999'}}

        def json(self, key):
            return raw[key]

        def pages(self, _key):
            return ['业绩比较基准' + expression + '。']

        def plain(self, key):
            return '活期存款0.350.35' if key == 'rate' else expression.translate(str.maketrans({'（': '(', '）': ')'}))

    inputs = {'fund_code': '999999', 'as_of_date': days[-1], 'verified_at': '2026-10-02',
        'index': {'code': 'fixture.SZ'}, 'target_refs': {'benchmark': 'benchmark', 'benchmark-page': 'page',
                                                     'summary': 'summary', 'prospectus': 'prospectus'},
        'gf_benchmark_reconstruction': {'protocol_hash': 'protocol', 'rate_source': 'rate',
                                       'annual_cash_rate': '.0035', 'index_daily': 'index'}}
    return Source(), inputs, {'dates': days, 'adjusted_nav': ['1'] * len(days)}, raw


def test_gf_cumulative_originals_require_one_common_scale_for_all_dates(fixture):
    source, inputs, history, raw = fixture
    result = compile_history(source, inputs, deepcopy(history))
    assert result['publisher_cross_check']['official_overlap'] == 3
    assert result['publisher_cross_check']['missing_publisher_dates'] == []
    raw['benchmark']['CREATE'][1]['YIELD_BENCHMARK'] = '-49.00%'
    with pytest.raises(ValueError, match='PRECISION_CONFLICT'):
        compile_history(source, inputs, deepcopy(history))


@pytest.mark.parametrize('damage,error', [('code', 'CODE_OR_PERIOD'), ('nav', 'NAV_CONFLICT'),
                                       ('precision', 'PERCENT_PRECISION'), ('hole', 'DAILY_COVERAGE')])
def test_gf_wrong_share_precision_or_missing_day_cannot_be_imputed(fixture, damage, error):
    source, inputs, history, raw = fixture
    if damage == 'code':
        raw['benchmark']['CREATE'][0]['FUNDCODE'] = '999998'
    elif damage == 'nav':
        raw['benchmark']['CREATE'][0]['YIELD'] = '1.00%'
    elif damage == 'precision':
        raw['benchmark']['CREATE'][0]['YIELD_BENCHMARK'] = '-50.001%'
    else:
        raw['benchmark']['CREATE'].pop(1)
    with pytest.raises(ValueError, match=error):
        compile_history(source, inputs, deepcopy(history))
