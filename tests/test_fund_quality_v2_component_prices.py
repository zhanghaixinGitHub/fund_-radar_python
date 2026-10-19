"""成分原件准入反例；所有响应均为工程假样本，不进入真实校准。"""

import hashlib
import json
from datetime import UTC, date, datetime

import pytest
from app.services.fund_materials_store import versioned_save
from app.services.fund_quality_v2_component_prices import (
    CATALOG,
    HISTORY,
    HSI_PATH,
    MANIFEST,
    component_histories,
    hsi_price_history,
    reuse_component_originals,
)
from app.services.fund_quality_v2_materials import inspect_sources, role_sources

FROZEN = datetime(2026, 10, 17, 18, tzinfo=UTC)


def examples(code='00011.00'):
    catalog = {'code': '0', 'data': [{'indexCode': code, 'indexShortName': 'HSCI',
        'indexName': 'Hang Seng Composite Index', 'indexType': 'PI', 'indexSeriesCode': 'hsci'}]}
    history = {'code': '0', 'data': [{'indexCode': code, 'indexName': 'Hang Seng Composite Index',
        'valueHistory': [{'date': '2023-09-27', 'close': '3000.01'},
                         {'date': '2023-09-28', 'close': '3001.23'},
                         {'date': '2026-09-30', 'close': '3500.17'},
                         {'date': '2026-10-16', 'close': '3532.19'}]}]}
    prefix = 'https://www.hsi.com.hk'+HSI_PATH
    refs = [{'label': CATALOG, 'component_code': 'HSCI', 'url': prefix+'?data=allIndexes&language=en-hk'},
            {'label': HISTORY, 'component_code': 'HSCI',
             'url': prefix+'?data=valueHistoryPub&language=en-hk&indexCodes='+code}]
    for value, ref in zip([catalog, history], refs, strict=True):
        ref['sha256'] = hashlib.sha256(json.dumps(value).encode()).hexdigest()
    return catalog, history, refs


def parse(catalog, history, refs):
    return hsi_price_history(json.dumps(catalog).encode(), json.dumps(history).encode(),
        catalog_ref=refs[0], history_ref=refs[1], required_from=date(2023, 9, 28), as_of=date(2026, 9, 30),
        required_fund_dates=['2023-09-28', '2023-09-29', '2026-09-30'])


def test_actual_directory_namespace_not_guessed_and_missing_dates_are_not_filled():
    result = parse(*examples('04567.00'))
    assert result['publisher_internal_code'] == '04567.00'
    assert result['prices'] == {'2023-09-28': '3001.23', '2026-09-30': '3500.17'}
    assert result['required_fund_dates_without_reported_close'] == ['2023-09-29']
    assert result['future_observations_excluded'] == 1
    assert result['window_predecessor_observation'] == {'date': '2023-09-27', 'close': '3000.01'}
    assert result['unit'] == 'INDEX_POINTS' and result['scoring_metric'] is None


@pytest.mark.parametrize('change', ['tri', 'duplicate_identity', 'wrong_code', 'wrong_history_name',
                                  'duplicate_day', 'unordered', 'infinite', 'zero', 'invalid_date',
                                  'wrong_host', 'duplicate_query', 'business_failure'])
def test_success_or_similar_names_cannot_replace_actual_price_identity_and_dates(change):
    catalog, history, refs = examples()
    if change == 'tri':
        catalog['data'][0]['indexType'] = 'TRI_Grs'
    elif change == 'duplicate_identity':
        catalog['data'].append(dict(catalog['data'][0]))
    elif change == 'wrong_code':
        refs[1]['url'] = refs[1]['url'].replace('00011.00', 'HSCI.00')
    elif change == 'wrong_history_name':
        history['data'][0]['indexName'] += ' Total Return Index'
    elif change == 'duplicate_day':
        history['data'][0]['valueHistory'].append(dict(history['data'][0]['valueHistory'][1]))
    elif change == 'unordered':
        history['data'][0]['valueHistory'].reverse()
    elif change in {'infinite', 'zero'}:
        history['data'][0]['valueHistory'][1]['close'] = 'Infinity' if change == 'infinite' else '0'
    elif change == 'invalid_date':
        history['data'][0]['valueHistory'][1]['date'] = '2023-09-31'
    elif change == 'wrong_host':
        refs[0]['url'] = refs[0]['url'].replace('www.hsi.com.hk', 'www.hsi.com.hk.example.org')
    elif change == 'duplicate_query':
        refs[1]['url'] += '&indexCodes=00011.00'
    else:
        history['code'] = '500'
    with pytest.raises(ValueError):
        parse(catalog, history, refs)


def archive(tmp_path, mutation=None):
    source, root = tmp_path/'input', tmp_path/'rating'
    (source/'raw').mkdir(parents=True)
    catalog, history, refs = examples()
    items = []
    for number, (value, ref) in enumerate(zip([catalog, history], refs, strict=True)):
        raw = json.dumps(value).encode()
        file = source/'raw'/(ref['sha256']+'.json')
        file.write_bytes(raw)
        received = '2026-10-17T17:10:00+08:00'
        receipt = {'url': ref['url'], 'status': 200, 'bytes': len(raw), 'sha256': ref['sha256'],
                   'received_at': received, 'file': str(file)}
        if mutation == 'boolean_status':
            receipt['status'] = True
        if mutation == 'wrong_size':
            receipt['bytes'] += 1
        if mutation == 'future_time':
            receipt['received_at'] = '2026-10-18T17:10:00+08:00'
        proof = source/'raw'/f'{number}-receipt.json'
        versioned_save(proof, True if mutation == 'boolean_receipt' else receipt)
        items.append({'fund_code': '012835', 'label': ref['label'], 'file': file.relative_to(source).as_posix(),
            'receipt': {'file': proof.relative_to(source).as_posix(),
                        'sha256': hashlib.sha256(proof.read_bytes()).hexdigest()}})
    versioned_save(source/MANIFEST, {'schema': 'QUALITY_COMPONENT_ORIGINALS_V1', 'items': items})
    return source, root


def test_receipt_originals_are_replayed_and_components_do_not_satisfy_complete_benchmark_role(tmp_path):
    source, root = archive(tmp_path)
    refs = reuse_component_originals('012835', source, root, FROZEN)
    checked = inspect_sources(refs, root, FROZEN)
    assert not role_sources(checked)['benchmark']
    result = component_histories(refs=refs, root=root, required_from=date(2023, 9, 28),
        as_of=date(2026, 9, 30), required_fund_dates=['2023-09-28', '2026-09-30'], frozen=FROZEN)
    assert result['components']['HSCI']['observations_in_window'] == 2
    assert not result['complete_benchmark'] and result['scoring_metric'] is None
    (root/next(r['file'] for r in refs if r['label'] == HISTORY)).write_bytes(b'changed')
    result = component_histories(refs=refs, root=root, required_from=date(2023, 9, 28),
        as_of=date(2026, 9, 30), required_fund_dates=[], frozen=FROZEN)
    assert result['error'] == 'COMPONENT_ORIGINAL_ACTUAL_HASH'


@pytest.mark.parametrize('mutation', ['boolean_status', 'wrong_size', 'future_time', 'boolean_receipt'])
def test_receipt_flags_or_changed_actual_content_do_not_admit_source(tmp_path, mutation):
    source, root = archive(tmp_path, mutation)
    with pytest.raises(ValueError):
        reuse_component_originals('012835', source, root, FROZEN)


def test_missing_explicit_freeze_does_not_depend_on_reference_order(tmp_path):
    source, root = archive(tmp_path)
    refs = reuse_component_originals('012835', source, root, FROZEN)
    for sequence in [refs, list(reversed(refs))]:
        result = component_histories(refs=sequence, root=root, required_from=date(2023, 9, 28),
            as_of=date(2026, 9, 30), required_fund_dates=[], frozen=None)
        assert result['error'] == 'COMPONENT_EXPLICIT_FROZEN_TIME_REQUIRED'
        assert result['scoring_metric'] is None
