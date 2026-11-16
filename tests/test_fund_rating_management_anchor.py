from datetime import date

from app.services.fund_rating_management_anchor import appointment_verified


def test_split_table_appointment_requires_same_person_and_adjacent_date():
    pages = ['杨梦    沪深300指数量化增  2019-        15年  泰资产管理\n'
             '强、博道中证A500  09-26    -          研究员\n'
             '刘兆阳  证800指数增强、博  2025-    -    4年\n'
             '道上证科创板综合    07-23                量化投资部']
    assert appointment_verified(pages, '杨梦', date(2019, 9, 26))
    assert appointment_verified(pages, '刘兆阳', date(2025, 7, 23))
    assert not appointment_verified(pages, '杨梦', date(2025, 7, 23))
    assert not appointment_verified(pages, '杨梦', date(2019, 9, 25))
    assert not appointment_verified(['杨梦 2019- 15年\n其他说明\n09-26 -'], '杨梦', date(2019, 9, 26))
    assert not appointment_verified(['杨梦 2019- 15年', '09-26 -'], '杨梦', date(2019, 9, 26))


def test_contiguous_appointment_still_works():
    assert appointment_verified(['杨梦 任职2019-09-26'], '杨梦', date(2019, 9, 26))


def test_full_issuer_json_can_anchor_quarter_but_market_json_cannot(monkeypatch):
    from app.services.fund_rating_management_anchor import verify_management_anchor

    legal = '测试股票型证券投资基金'
    quarter = legal + '2026年第3季度报告2026年9月30日007832杨梦基金经理2019-09-26'
    full = legal + '2026年中期报告2026年6月30日007832杨梦基金经理2019-09-26' + '完整报告原文' * 4000

    class Sources:
        def __init__(self, inputs, root, lazy):
            self.sources = inputs['sources']

        def pages(self, sha):
            assert sha != 'market'
            return [quarter if sha == 'quarter' else full]

        def json(self, sha):
            if sha == 'market':
                return {'data': []}
            return {'data': {'notice_content': quarter if sha == 'quarter' else full}}

    monkeypatch.setattr('app.services.fund_rating_management_anchor.Originals', Sources)
    inputs = {'fund_code': '007832', 'legal_name': legal, 'as_of_date': '2026-10-30',
              'holding_date': '2026-09-30', 'management_primary_report_hash': 'full',
              'sources': {k: {'kind': 'json'} for k in ['quarter', 'full', 'market']},
              'management_inputs': {'experience': {'source_refs': ['quarter']},
                  'continuity': {'period_end': '2026-10-30',
                                 'intervals': [{'name': '杨梦', 'start': '2019-09-26', 'end': None}]},
                  'claims': [{'source_hash': 'quarter', 'quote': '杨梦基金经理2019-09-26'}]}}
    result = verify_management_anchor(inputs, None, 'quarter')
    assert result['known_latest_full_period'] == '2026-06-30'
    assert result['latest_period'] == '2026-09-30'
