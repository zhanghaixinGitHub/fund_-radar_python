"""联接执行项不能因当前持仓成立就越过历史路线和本基金投资目标。"""

from datetime import date

import pytest
from app.services import fund_rating_passive_execution as execution
from app.services.fund_rating_inputs import canonical
from app.services.fund_rating_reviewed_quality import PASSIVE_BASIS


@pytest.mark.parametrize('mandate,error', [
    ('本基金主动选股', 'OWN_PASSIVE_MANDATE'),
    ('本基金为完全被动式指数基金', 'HISTORICAL_TARGET'),
    ('本基金为ETF联接基金。本基金为指数型基金。'
     '本基金通过主要投资于目标ETF，紧密跟踪业绩比较基准，追求跟踪偏离度和跟踪误差最小化。',
     'HISTORICAL_TARGET'),
    ('本基金为ETF联接基金。本基金为指数型基金。', 'OWN_PASSIVE_MANDATE'),
])
def test_feeder_requires_own_mandate_and_same_historical_legal_target(monkeypatch, tmp_path, mandate, error):
    own = '2.2基金产品说明' + mandate + '2.2.1目标基金产品说明本基金为完全被动式指数基金'

    class Source:
        sources = {'half': {}, 'annual': {}}

        def __init__(self, *_args):
            pass

        def pages(self, digest):
            target = '888888' if digest == 'half' else '777777'
            return ['2.1.1目标基金基本情况' + target + '隔离目标ETF2.2基金产品说明']

    inputs = {'version': execution.VERSION, 'basis': PASSIVE_BASIS, 'fund_code': '999999',
              'legal_name': '隔离联接基金', 'route': 'FEEDER', 'primary_ref': 'half',
              'holding_date': '2026-06-30', 'as_of_date': '2026-09-30',
              'target_refs': {'half': 'half'}, 'target_annuals': {'2025': 'annual'},
              'target_etf': {'fund_code': '888888', 'legal_name': '隔离目标ETF'}}
    monkeypatch.setattr(execution, 'Originals', Source)
    monkeypatch.setattr(execution, '_identity', lambda *_args: [own])
    monkeypatch.setattr(execution, 'report_period', lambda _: (date(2026, 1, 1), date(2026, 6, 30)))
    monkeypatch.setattr(execution, '_history', lambda *_args: {'dates': ['2023-09-28', '2026-09-30']})
    monkeypatch.setattr(execution, 'contract_history_refs', lambda *_args: {'2025': 'annual'})
    with pytest.raises(ValueError, match=error):
        execution._calculate_cached(canonical(inputs), str(tmp_path))
