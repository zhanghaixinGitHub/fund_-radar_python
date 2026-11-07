"""隔离三日原件验证：不能把供应商复权断点、错误份额或缺日变成分数。"""

import json
from copy import deepcopy
from decimal import Decimal

import pytest
from app.services.fund_rating_southern_benchmark import compile_history


@pytest.fixture
def case():
    days = ['2026-09-28', '2026-09-29', '2026-09-30']
    compact_days = [d.replace('-', '') for d in days]
    refs = {key: key for key in ('benchmark', 'benchmark-page', 'benchmark-script', 'benchmark-main-script',
                                'provider', 'issuer-nav', 'issuer-events', 'issuer-identity', 'summary', 'prospectus')}
    params = {'fundCode': '999999', 'startDate': compact_days[0], 'endDate': compact_days[-1]}
    def curve(values):
        return [{'fdate': day, 'upratio': value} for day, value in zip(compact_days, values, strict=True)]
    raw = {
        'benchmark': {'jzzscode': '000905.SH', 'transdate': '',
                      'thisfundlist': curve(['0.00', '1.00', '2.00']), 'brList': curve(['0.00', '0.90', '1.80'])},
        'issuer-nav': {'startDate': 20260928, 'endDate': 20260930, 'dtlist': compact_days,
                       'navList': ['1', '1.01', '1.02'], 'navLJList': ['1.1', '1.11', '1.12']},
        'issuer-events': {k: {'total': 0, 'list': [], 'pages': 0, 'pageNum': 1, 'pageSize': 10}
                          for k in ('jjfhlist', 'splitlist')},
        'issuer-identity': {'fund_info': {'fundCode': '999999', 'fundNameEx': '隔离基金(A类份额)',
                                        'contractValidDate': '20090925',
                                        'yjbjjz': '隔离指数收益率×95%+银行活期存款利率（税后）×5%'}},
    }
    raw['benchmark']['thisfundlist'].append(deepcopy(raw['benchmark']['thisfundlist'][-1]))
    encoded = {key: json.dumps({'code': 'ETS-5BP00000', 'data': value}).encode() for key, value in raw.items()}
    encoded['provider'] = b'{"dividend_records":[]}'
    encoded['protocol'] = json.dumps({
        'fund_code': '999999', 'as_of': days[-1], 'first': days[0], 'last': days[-1], 'frozen_at': '2026-10-01',
        'issuer_percent_decimal_places': 2, 'method': 'COMPLETE_ISSUER_PUBLISHED_CURVE_NO_RECONSTRUCTION',
        'published_nonvaluation_dates_retained': [], 'identical_own_terminal_duplicates_retained': [days[-1]],
        'index_code': '000905.SH',
    }).encode()
    encoded['benchmark-page'] = ('<script src="/js/detail-abcd.js"></script>'
                                 '<script src="/js/main-abcd.js"></script>'
                                 '基金收益走势数据根据基金复权净值计算').encode()
    encoded['benchmark-script'] = ('url:q.earningsMovements,type:"post",dataType:"json",data:n;'
                                   'n=Object.assign({fundCode:a},n);i=o.thisfundlist;f=o.brList;'
                                   'x.push(t.upratio);w.push(t.upratio);seriesName1:"业绩基准：";series:x,series1:w;'
                                   'url:q.netValue,type:"post";o=n.navList,i=n.navLJList,r=n.dtlist;'
                                   'url:q.dividends,type:"post";e=t.jjfhlist,a=t.splitlist;'
                                   'url:q.overreview,type:"post",dataType:"json",data:{fundCode:a}').encode()
    encoded['benchmark-main-script'] = (b'earningsMovements:"/nfwebApi/fund/earningsMovements";'
                                        b'netValue:"/nfwebApi/fund/netValue";dividends:"/nfwebApi/fund/dividends"')

    class Source:
        def __init__(self):
            self.raw = encoded
            self.sources = {
                key: {'url': 'https://www.southernfund.com/nfwebApi/fund/' + endpoint,
                      'method': 'POST_FORM', 'params': deepcopy(p)}
                for key, endpoint, p in [('benchmark', 'earningsMovements', params), ('issuer-nav', 'netValue', params),
                                         ('issuer-identity', 'overreview', {'fundCode': '999999'}),
                                         ('issuer-events', 'dividends', {'fundCode': '999999', 'startDate': '20090101',
                                                                       'endDate': '20260930', 'curPage': 1,
                                                                       'pageSize': 10})]
            }
            self.sources.update({
                'benchmark-page': {'url': 'https://www.southernfund.com/new/personal-financing/detail.html?fundCode=999999'},
                'benchmark-script': {'url': 'https://www.southernfund.com/js/detail-abcd.js'},
                'benchmark-main-script': {'url': 'https://www.southernfund.com/js/main-abcd.js'},
            })

        def json(self, digest):
            return json.loads(self.raw[digest])

        def pages(self, _digest):
            return ['业绩比较基准隔离指数收益率×95%+银行活期存款利率（税后）×5%']

    inputs = {'fund_code': '999999', 'share_class': 'A', 'legal_name': '隔离基金', 'as_of_date': days[-1],
              'verified_at': '2026-10-02', 'target_refs': refs, 'southern_protocol': 'protocol'}
    history = {'dates': days, 'adjusted_nav': ['1', '.95', '.96'], 'rows': [
        {'nav_date': d, 'ann_date': d, 'unit_nav': u, 'accumulated_nav': a}
        for d, u, a in zip(days, ['1', '1.01', '1.02'], ['1.1', '1.11', '1.12'], strict=True)
    ]}
    return Source(), inputs, history


def test_original_precision_and_own_unit_nav_close_return_without_rewriting_source(case):
    source, inputs, history = case
    original = deepcopy(source.raw)
    result = compile_history(source, inputs, history)
    assert list(map(Decimal, result['adjusted_nav'])) == [Decimal(1), Decimal('1.01'), Decimal('1.02')]
    assert list(map(Decimal, result['benchmark_index'])) == [Decimal(1), Decimal('1.009'), Decimal('1.018')]
    assert result['publisher_cross_check']['provider_adjusted_nav_conflict_dates'] == history['dates'][1:]
    assert source.raw == original
    assert result['publisher_cross_check']['missing_values_filled'] == 0


@pytest.mark.parametrize('change,error', [
    ('share', 'REQUEST_SCOPE'), ('column', 'COLUMN_SEMANTICS'), ('precision', 'PRECISION'),
    ('missing', 'DAILY_COVERAGE'), ('duplicate_conflict', 'DUPLICATE'), ('nav', 'CASH_CONFLICT'),
    ('return', 'RETURN_CONFLICT'), ('dividend', 'DIVIDEND_ADAPTER'), ('paginated', 'PAGINATION'),
    ('split', 'SPLIT_ADAPTER'), ('identity', 'CONTRACT_IDENTITY'), ('route', 'ROUTE_CHANGE'),
])
def test_incomplete_or_conflicting_evidence_cannot_be_accepted(case, change, error):
    source, inputs, history = case
    raw = source.json('benchmark')
    if change == 'share':
        source.sources['benchmark']['params']['fundCode'] = '999998'
    elif change == 'column':
        source.raw['benchmark-script'] = source.raw['benchmark-script'].replace(b'f=o.brList', b'f=o.jzlist')
    elif change == 'precision':
        raw['data']['brList'][-1]['upratio'] = '1.801'
    elif change == 'missing':
        raw['data']['brList'].pop()
    elif change == 'duplicate_conflict':
        raw['data']['thisfundlist'][-1]['upratio'] = '2.01'
    elif change == 'nav':
        history['rows'][-1]['unit_nav'] = '1.021'
    elif change == 'return':
        raw['data']['thisfundlist'][-1]['upratio'] = '2.01'
        raw['data']['thisfundlist'][-2]['upratio'] = '2.01'
    elif change in ('dividend', 'paginated', 'split'):
        events = source.json('issuer-events')
        if change == 'dividend':
            table = events['data']['jjfhlist']
            table.update(total=1, pages=1, list=[{'f1': '999999', 'securitiesBo': {'f16': '999999'},
                                               'f3': '20260929', 'f8': '20260929', 'f10': '20260930'}])
        elif change == 'paginated':
            events['data']['jjfhlist']['total'] = 11
        else:
            events['data']['splitlist'].update(total=1, pages=1, list=[{'splitDate': '20260929'}])
        source.raw['issuer-events'] = json.dumps(events).encode()
    elif change == 'identity':
        identity = source.json('issuer-identity')
        identity['data']['fund_info']['fundNameEx'] = '隔离基金(C类份额)'
        source.raw['issuer-identity'] = json.dumps(identity).encode()
    elif change == 'route':
        raw['data']['transdate'] = '20251010'
    source.raw['benchmark'] = json.dumps(raw).encode()
    with pytest.raises(ValueError, match=error):
        compile_history(source, inputs, history)
