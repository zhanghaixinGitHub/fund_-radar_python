"""日历与最近收盘引用的反例；文本和价格为工程假样本，不作真实校准。"""

import hashlib
import json
from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest
from app.services.fund_materials_store import versioned_save
from app.services.fund_quality_v2_market_calendar import (
    CALENDAR,
    CLOSED,
    HALF,
    calendar_endpoint,
    closure_evidence,
    holiday_circular,
    price_references,
    weather_closure,
)

URL = ('https://www.hkex.com.hk/-/media/HKEX-Market/Services/Circulars-and-Notices/'
       'Participant-and-Members-Circulars/SEHK/2023/ce_SEHK_CT_079_2023.pdf')


def pages():
    return ['THE STOCK EXCHANGE OF HONG KONG LIMITED\nDate: 09/06/2023\n'
        'Subject: Hong Kong Securities Market Holiday Schedule for Year 2024\n' + CLOSED + '\n'
        'Date Day Holiday\n25 December 2024 Wednesday Christmas Day\n',
        '26 December 2024 Thursday The first weekday after Christmas Day\n' + HALF + '\n'
        '24 December 2024 (Tuesday) - Eve of Christmas Day\n31 December 2024 (Tuesday) - Eve of New Year']


def test_holiday_continues_on_second_page_but_half_trading_dates_stay_open():
    result = holiday_circular(pages(), year=2024, as_of=date(2026, 9, 30))
    assert [r['date'] for r in result['holiday_rows']] == ['2024-12-25', '2024-12-26']
    assert result['half_days'] == ['2024-12-24', '2024-12-31']
    assert result['document_dated_on'] == '2023-06-09'


@pytest.mark.parametrize('old,new', [('Thursday', 'Friday'), ('Year 2024', 'Year 2025'),
                                   ('09/06/2023', '09/06/2027'), (CLOSED, 'other table'),
                                   ('(Tuesday)', '(Wednesday)')])
def test_mismatched_days_and_after_cutoff_documents_are_rejected(old, new):
    with pytest.raises(ValueError):
        holiday_circular([p.replace(old, new) for p in pages()], year=2024, as_of=date(2026, 9, 30))


def evidence():
    parsed = holiday_circular(pages(), year=2024, as_of=date(2026, 9, 30))
    hashes = ['actual-calendar', 'actual-receipt']
    return {'calendars': {'2024': parsed | {'source_hashes': hashes}},
            'explicit_closed_dates': {r['date']: r | {'state': 'HOLIDAY', 'source_hashes': hashes}
                                      for r in parsed['holiday_rows']}, 'rejected_originals': []}


def component(prices):
    return {'prices': prices, 'source_hashes': ['actual-index', 'actual-index-receipt'],
            'window_predecessor_observation': None}


def test_closed_dates_reference_actual_price_without_manufacturing_a_new_close():
    result = price_references(component({'2024-12-24': '100.12', '2024-12-27': '103.45'}),
        ['2024-12-24', '2024-12-25', '2024-12-26', '2024-12-27'], evidence())
    assert result['status'] == 'ALL_REQUESTED_DATES_EXPLAINED'
    assert result['actual_close_rows'] == result['closed_reference_rows'] == 2
    reference = result['rows'][2]
    assert reference['actual_close_date'] == '2024-12-24' and reference['reference_close'] == '100.12'
    assert reference['observed_close_on_valuation_day'] is None and 'actual_close' not in reference
    assert len(reference['closure_evidence']) == 2 and result['new_price_observations_created'] == 0
    assert not result['complete_cny_benchmark']


def test_half_day_missing_price_and_intervening_open_day_are_never_filled():
    result = price_references(component({'2024-12-23': '100.12'}),
        ['2024-12-24', '2024-12-25'], evidence())
    assert result['unresolved_dates'] == ['2024-12-24', '2024-12-25']
    assert result['closed_reference_rows'] == 0


def test_actual_price_on_closed_day_is_a_conflict_and_empty_fund_calendar_is_not_success():
    result = price_references(component({'2024-12-25': '100.12'}), ['2024-12-25'], evidence())
    assert result['unresolved_dates'] == ['2024-12-25']
    assert result['rows'][0]['state'] == 'PRICE_PRESENT_ON_DOCUMENTED_CLOSED_DAY_REVIEW_REQUIRED'
    result = price_references(component({}), [], evidence())
    assert result['status'] == 'NO_VERIFIED_FUND_VALUATION_CALENDAR'


def test_weekend_reference_requires_the_actual_year_circular_and_predecessor_is_preserved():
    value = component({'2024-12-27': '100.12'})
    result = price_references(value, ['2024-12-28', '2024-12-29'], evidence())
    assert result['closed_reference_rows'] == 2
    result = price_references(value, ['2024-12-28'], {'calendars': {}, 'explicit_closed_dates': {}})
    assert result['unresolved_dates'] == ['2024-12-28']


def weather():
    return (b'Securities and Derivatives Market Trading Arrangements under Typhoon Signal No. 8 '
            b'06 Sep 2024 If Typhoon Signal No. 8 or above, Black Rainstorm Warning or any announcement of Extreme '
            b'Conditions, remains issued at 12:00 noon, all trading sessions today will be cancelled.',
            b'Super Typhoon Yagi (2411) The No. 8 Northeast Gale or Storm Signal was replaced by the '
            b'No. 3 Strong Wind Signal at 12:40 p.m. on 6 September.')


def test_weather_closure_remains_an_explicit_two_premise_inference_for_one_date_only():
    event = weather_closure(*weather())
    assert event['date'] == '2024-09-06' and event['not_a_direct_final_cancellation_announcement']
    rule, actual = weather()
    with pytest.raises(ValueError, match='PREMISES_NOT_BOUND'):
        weather_closure(rule, actual.replace(b'12:40', b'11:40'))
    with pytest.raises(ValueError, match='PREMISES_NOT_BOUND'):
        weather_closure(rule.replace(b'06 Sep 2024', b'06 Sep 2025'), actual)


def test_endpoint_year_and_host_are_bound_before_reading_any_body():
    calendar_endpoint({'label': CALENDAR, 'url': URL, 'calendar_year': 2024})
    for url, year in [(URL.replace('www.hkex.com.hk', 'www.hkex.com.hk.example.org'), 2024), (URL, 2025)]:
        with pytest.raises(ValueError):
            calendar_endpoint({'label': CALENDAR, 'url': url, 'calendar_year': year})


def test_duplicate_year_rejects_both_versions_in_either_order(tmp_path, monkeypatch):
    import app.services.fund_quality_v2_market_calendar as module

    monkeypatch.setattr(module, 'PdfReader', lambda _raw: SimpleNamespace(
        pages=[SimpleNamespace(extract_text=lambda p=p: p) for p in pages()]))
    source = tmp_path/'sources'
    source.mkdir()
    refs = []
    for number in range(2):
        raw = f'TEST_PDF_{number}'.encode()
        sha = hashlib.sha256(raw).hexdigest()
        (source/(sha+'.pdf')).write_bytes(raw)
        received = '2026-10-17T17:00:00+08:00'
        receipt = source/f'{number}-receipt.json'
        versioned_save(receipt, {'url': URL, 'status': 200, 'bytes': len(raw), 'sha256': sha,
                                'received_at': received})
        proof = hashlib.sha256(receipt.read_bytes()).hexdigest()
        refs += [{'label': CALENDAR, 'url': URL, 'calendar_year': 2024, 'sha256': sha,
                  'file': (source/(sha+'.pdf')).relative_to(tmp_path).as_posix(), 'acquired_at': received,
                  'component_receipt_sha256': proof},
                 {'label': 'component-price-receipt', 'sha256': proof,
                  'file': receipt.relative_to(tmp_path).as_posix(), 'acquired_at': received}]
    single = closure_evidence(refs=refs[:2], root=tmp_path, as_of=date(2026, 9, 30),
                               frozen=datetime(2026, 10, 17, 18, tzinfo=UTC))
    assert '2024' in single['calendars'] and json.loads(json.dumps(single)) == single
    for sequence in [refs, list(reversed(refs))]:
        parsed = closure_evidence(refs=sequence, root=tmp_path, as_of=date(2026, 9, 30),
                                  frozen=datetime(2026, 10, 17, 18, tzinfo=UTC))
        assert parsed['calendars'] == {} and parsed['explicit_closed_dates'] == {}
        assert parsed['conflicting_calendar_years'] == [2024]


def test_http200_and_matching_receipt_do_not_make_html_a_valid_calendar_pdf(tmp_path):
    source = tmp_path/'sources'
    source.mkdir()
    raw = b'<html>not a calendar PDF</html>'
    sha = hashlib.sha256(raw).hexdigest()
    file = source/(sha+'.pdf')
    file.write_bytes(raw)
    receipt = source/'receipt.json'
    received = '2026-10-17T17:00:00+08:00'
    versioned_save(receipt, {'url': URL, 'status': 200, 'bytes': len(raw), 'sha256': sha, 'received_at': received})
    proof = hashlib.sha256(receipt.read_bytes()).hexdigest()
    refs = [{'label': CALENDAR, 'url': URL, 'calendar_year': 2024, 'sha256': sha,
             'file': file.relative_to(tmp_path).as_posix(), 'acquired_at': received,
             'component_receipt_sha256': proof},
            {'label': 'component-price-receipt', 'sha256': proof,
             'file': receipt.relative_to(tmp_path).as_posix(), 'acquired_at': received}]
    parsed = closure_evidence(refs=refs, root=tmp_path, as_of=date(2026, 9, 30),
                              frozen=datetime(2026, 10, 17, 18, tzinfo=UTC))
    assert not parsed['calendars'] and len(parsed['rejected_originals']) == 1
