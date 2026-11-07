"""首年季度只能在已冻结规则和真实年度来源失败下覆盖整个窗口，不能择样。"""

from copy import deepcopy

import pytest
from app.services.fund_rating_direct_index import contract_history_refs


@pytest.fixture
def fixture():
    pages = {'q3': '本报告期自2023年7月1日起至2023年9月30日止。',
             'q4': '本报告期自2023年10月1日起至2023年12月31日止。',
             'y24': '本报告期自2024年1月1日起至2024年12月31日止。',
             'y25': '本报告期自2025年1月1日起至2025年12月31日止。'}
    raw = {'failed': {'data': {'notice_content': ''}},
           'catalog': {'ErrCode': 0, 'Data': [{'ID': 'AN2023', 'TITLE': '隔离基金2023年年度报告'}]}}
    refs = {'failed': {'url': 'https://np-cnotice-fund.eastmoney.com/api/content/ann',
                       'params': {'art_code': 'AN2023'}},
            'catalog': {'url': 'https://api.fund.eastmoney.com/f10/JJGG', 'params': {'fundcode': '999999'}},
            'q3': {}, 'q4': {}}

    class Source:
        sources = refs

        def pages(self, key):
            return [pages[key]]

        def json(self, key):
            return raw[key]

    alt = {'2023': {'failure_source': 'failed', 'catalog_source': 'catalog', 'quarter_refs': ['q3', 'q4']}}
    return Source(), {'2024': 'y24', '2025': 'y25'}, alt, raw, refs


def invoke(source, annuals, alternatives, frozen=None):
    declaration = frozen if frozen is not None else {'first_window_annual_alternative': '事前冻结首年完整季度替代'}
    return contract_history_refs(source, annuals, alternatives, '2023-09-28', '2026-09-30', '999999',
                                 declaration)


def test_first_year_quarters_cover_every_applicable_day_without_shortening_nav_window(fixture):
    source, annuals, alt, _raw, _refs = fixture
    result = invoke(source, annuals, alt)
    assert set(result.values()) == {'q3', 'q4', 'y24', 'y25'}
    wrong = deepcopy(alt)
    wrong['2023']['quarter_refs'] = ['q4']
    with pytest.raises(ValueError, match='DO_NOT_COVER_WINDOW'):
        invoke(source, annuals, wrong)
    with pytest.raises(ValueError, match='FROZEN_RULE_REQUIRED'):
        invoke(source, annuals, alt, {})


@pytest.mark.parametrize('damage', ['wrong_catalog', 'annual_available', 'derived_quarter'])
def test_failure_and_quarter_evidence_are_checked_instead_of_accepting_a_declaration(fixture, damage):
    source, annuals, alt, raw, refs = fixture
    if damage == 'wrong_catalog':
        refs['catalog']['params']['fundcode'] = '999998'
    elif damage == 'annual_available':
        raw['failed']['data']['notice_content'] = '实际取得完整年度正文'
    else:
        refs['q3']['archive_kind'] = 'DERIVED'
    with pytest.raises(ValueError):
        invoke(source, annuals, alt)
