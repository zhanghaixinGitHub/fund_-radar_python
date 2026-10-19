"""从发布方目录和原始行情重建合同成分价格，尚不连接完整人民币合同基准。"""

import hashlib
import json
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from app.services.fund_exposure_common import _publish, read

VERSION = 'QUALITY_COMPONENT_PRICES_20261017_1'
MANIFEST = 'quality-component-originals.json'
CATALOG = 'component-price-catalog'
HISTORY = 'component-price-history'
RECEIPT = 'component-price-receipt'
HSI_PATH = '/api/wsit-hsi-ddoc-ea-public-website-proxy/v1/product-data/pub/indexes/metadata/v2'


def _query(ref: dict, *, role: str) -> dict:
    """只接受实际公开目录/历史端点及唯一参数；基金代码不冒充指数身份。"""
    address = urlsplit(ref['url'])
    query = parse_qs(address.query, keep_blank_values=True)
    expected = {'data': ['allIndexes' if role == CATALOG else 'valueHistoryPub'], 'language': ['en-hk']}
    if role == HISTORY:
        expected['indexCodes'] = query.get('indexCodes', [])
    if (ref.get('label') != role or role not in {CATALOG, HISTORY} or ref.get('component_code') != 'HSCI'
            or address.scheme != 'https' or address.hostname != 'www.hsi.com.hk' or address.path != HSI_PATH
            or address.username or address.password or address.port not in {None, 443} or address.fragment
            or query != expected or (role == HISTORY and len(query.get('indexCodes', [])) != 1)):
        raise ValueError('COMPONENT_HSI_ENDPOINT_ROLE_OR_QUERY')
    return query


def hsi_price_history(catalog_raw: bytes, history_raw: bytes, *, catalog_ref: dict, history_ref: dict,
                      required_from: date, as_of: date, required_fund_dates: list[str]) -> dict:
    """按目录精确别名、名称及PI类型找内部代码，不猜HSCI.00或用全收益指数。

    返回指数点位原值及窗口前最近一条观察，单位是指数点。日期缺失不自动
    当休市，不填前值、补零、插值或折算汇率；未来收盘值保留在原件但不进入
    资料窗口。1231个价格观察不能当作1231个独立校准产品。
    """
    _query(catalog_ref, role=CATALOG)
    query = _query(history_ref, role=HISTORY)
    if len(catalog_raw) > 12_000_000 or len(history_raw) > 12_000_000 or required_from > as_of:
        raise ValueError('COMPONENT_HSI_SIZE_OR_WINDOW')
    catalog, history = (json.loads(raw) for raw in (catalog_raw, history_raw))
    if (not isinstance(catalog, dict) or catalog.get('code') != '0'
            or not isinstance(catalog.get('data'), list) or not 0 < len(catalog['data']) <= 10000
            or not all(isinstance(row, dict) for row in catalog['data'])):
        raise ValueError('COMPONENT_HSI_CATALOG_SCHEMA')
    matches = [row for row in catalog['data'] if row.get('indexShortName') == 'HSCI'
               and row.get('indexName') == 'Hang Seng Composite Index' and row.get('indexType') == 'PI'
               and row.get('indexSeriesCode') == 'hsci']
    if len(matches) != 1 or not re.fullmatch(r'\d{5}\.\d{2}', str(matches[0].get('indexCode'))):
        raise ValueError('COMPONENT_HSI_PRICE_INDEX_IDENTITY_NOT_UNIQUE')
    identity = matches[0]
    if query['indexCodes'] != [identity['indexCode']]:
        raise ValueError('COMPONENT_HSI_CANONICAL_CODE_REQUEST_MISMATCH')
    if (not isinstance(history, dict) or history.get('code') != '0'
            or not isinstance(history.get('data'), list) or len(history['data']) != 1
            or not isinstance(history['data'][0], dict)):
        raise ValueError('COMPONENT_HSI_HISTORY_SCHEMA')
    series = history['data'][0]
    if series.get('indexCode') != identity['indexCode'] or series.get('indexName') != identity['indexName']:
        raise ValueError('COMPONENT_HSI_HISTORY_IDENTITY_CONFLICT')
    rows = series.get('valueHistory')
    if not isinstance(rows, list) or not 0 < len(rows) <= 10000:
        raise ValueError('COMPONENT_HSI_HISTORY_COUNT')
    points, sequence = {}, []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('date'), str):
            raise ValueError('COMPONENT_HSI_PRICE_ROW')
        day = str(date.fromisoformat(row['date']))
        try:
            price = Decimal(row['close']) if isinstance(row.get('close'), str) else None
        except InvalidOperation as exc:
            raise ValueError('COMPONENT_HSI_PRICE_NUMBER') from exc
        if price is None or not price.is_finite() or price <= 0 or day in points:
            raise ValueError('COMPONENT_HSI_PRICE_POSITIVITY_OR_DUPLICATE')
        sequence.append(day)
        points[day] = str(price)
    if sequence != sorted(sequence):
        raise ValueError('COMPONENT_HSI_PRICE_DATE_ORDER')
    selected = {day: value for day, value in points.items() if str(required_from) <= day <= str(as_of)}
    if not selected:
        raise ValueError('COMPONENT_HSI_NO_OBSERVATIONS_IN_WINDOW')
    before = [day for day in points if day < str(required_from)]
    predecessor = {'date': before[-1], 'close': points[before[-1]]} if before else None
    normalized_days = [str(date.fromisoformat(day)) for day in required_fund_dates]
    if (normalized_days != required_fund_dates or required_fund_dates != sorted(set(required_fund_dates))
            or any(not str(required_from) <= day <= str(as_of) for day in required_fund_dates)):
        raise ValueError('COMPONENT_FUND_DATE_WINDOW_OR_ORDER')
    return {'version': VERSION, 'component_name': '恒生综合指数', 'publisher_short_name': 'HSCI',
            'publisher_internal_code': identity['indexCode'], 'publisher_index_type': 'PI',
            'identity_evidence': identity, 'unit': 'INDEX_POINTS', 'prices': selected,
            'observations_in_original': len(points), 'observations_in_window': len(selected),
            'window_predecessor_observation': predecessor,
            'future_observations_excluded': sum(day > str(as_of) for day in points),
            'required_fund_dates_without_reported_close': [day for day in required_fund_dates if day not in selected],
            'source_hashes': [catalog_ref['sha256'], history_ref['sha256']],
            'alignment_state': 'ACTUAL_PRICE_DATES_ONLY_MARKET_CLOSURE_AND_FUND_VALUATION_MAPPING_REQUIRED',
            'scope': 'PUBLISHER_COMPONENT_PRICES_NOT_CNY_CONTRACT_BENCHMARK_OR_FX_SOURCE', 'scoring_metric': None}


def _verified_original(ref: dict, refs: list[dict], root: Path, frozen: datetime) -> tuple[bytes, str]:
    """原行情与下载收据均重读原字节，核URL、实际大小、摘要及有时区取得日。"""
    file = (root/ref['file']).resolve()
    if (not file.is_relative_to((root/'sources').resolve()) or not file.is_file()
            or file.stat().st_size > 12_000_000):
        raise ValueError('COMPONENT_ORIGINAL_PATH_OR_SIZE')
    raw = file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('COMPONENT_ORIGINAL_ACTUAL_HASH')
    proofs = [r for r in refs if r.get('label') == RECEIPT and r.get('sha256') == ref.get('component_receipt_sha256')]
    if len(proofs) != 1:
        raise ValueError('COMPONENT_ACTUAL_DOWNLOAD_RECEIPT_REQUIRED')
    proof = proofs[0]
    receipt_file = (root/proof['file']).resolve()
    if (not receipt_file.is_relative_to((root/'sources').resolve()) or not receipt_file.is_file()
            or receipt_file.stat().st_size > 100_000
            or hashlib.sha256(receipt_file.read_bytes()).hexdigest() != proof['sha256']):
        raise ValueError('COMPONENT_RECEIPT_PATH_SIZE_OR_HASH')
    receipt = read(receipt_file)
    if (not isinstance(receipt, dict) or type(receipt.get('status')) is not int or receipt['status'] != 200
            or type(receipt.get('bytes')) is not int or receipt['bytes'] != len(raw)
            or receipt.get('sha256') != ref['sha256'] or receipt.get('url') != ref['url']
            or receipt.get('received_at') != ref['acquired_at'] or proof.get('acquired_at') != ref['acquired_at']):
        raise ValueError('COMPONENT_RECEIPT_BYTES_URL_TIME_NOT_BOUND')
    received = datetime.fromisoformat(receipt['received_at'])
    if received.tzinfo is None or frozen.tzinfo is None or received > frozen:
        raise ValueError('COMPONENT_ACTUAL_TIME_NOT_FROZEN')
    return raw, proof['sha256']


def component_histories(*, refs: list[dict], root: Path, required_from: date, as_of: date,
                        required_fund_dates: list[str], frozen: datetime | None) -> dict:
    """只解析已经登记的实际成分原件，缺失/错误逐件保留，绝不拼成虚构基准。"""
    catalogs = [r for r in refs if r.get('label') == CATALOG and r.get('component_code') == 'HSCI']
    histories = [r for r in refs if r.get('label') == HISTORY and r.get('component_code') == 'HSCI']
    if not catalogs and not histories:
        return {}
    if not isinstance(frozen, datetime) or frozen.tzinfo is None:
        return {'version': VERSION, 'error': 'COMPONENT_EXPLICIT_FROZEN_TIME_REQUIRED',
                'complete_benchmark': False, 'scoring_metric': None}
    try:
        if len(catalogs) != 1 or len(histories) != 1:
            raise ValueError('COMPONENT_CATALOG_HISTORY_PAIR_NOT_UNIQUE')
        catalog, catalog_proof = _verified_original(catalogs[0], refs, root, frozen)
        history, history_proof = _verified_original(histories[0], refs, root, frozen)
        result = hsi_price_history(catalog, history, catalog_ref=catalogs[0], history_ref=histories[0],
            required_from=required_from, as_of=as_of, required_fund_dates=required_fund_dates)
        result['source_hashes'] = sorted(set(result['source_hashes'] + [catalog_proof, history_proof]))
        from app.services.fund_quality_v2_market_calendar import closure_evidence, price_references

        closures = closure_evidence(refs=refs, root=root, as_of=as_of, frozen=frozen)
        result['valuation_date_price_references'] = price_references(result, required_fund_dates, closures)
        result['market_calendar_evidence'] = closures
        return {'version': VERSION, 'status': 'COMPONENT_PRICE_ORIGINALS_REBUILT', 'components': {'HSCI': result},
                'remaining': ['OTHER_WEIGHTED_COMPONENTS_BINDING', 'ACTUAL_FUND_VALUATION_FX',
                              'MARKET_CLOSURE_AND_VALUATION_CALENDAR', 'COMPLETE_EFFECTIVE_CONTRACT_HISTORY'],
                'complete_benchmark': False, 'scoring_metric': None}
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return {'version': VERSION, 'error': str(exc), 'source_hashes': [r.get('sha256') for r in catalogs+histories],
                'complete_benchmark': False, 'scoring_metric': None}


def reuse_component_originals(code: str, source_root: Path, rating_root: Path, frozen: datetime) -> list[dict]:
    """从普通source-root复用行情及真实收据；清单的路由代码不能授予指数身份。"""
    source_root = source_root.resolve()
    manifest = source_root/MANIFEST
    if not manifest.exists():
        return []
    if not re.fullmatch(r'\d{6}', code) or manifest.stat().st_size > 1_000_000:
        raise ValueError('COMPONENT_MANIFEST_CODE_OR_SIZE')
    data = read(manifest)
    if (not isinstance(data, dict) or data.get('schema') != 'QUALITY_COMPONENT_ORIGINALS_V1'
            or not isinstance(data.get('items'), list)
            or len(data['items']) > 100 or not all(isinstance(item, dict) for item in data['items'])):
        raise ValueError('COMPONENT_MANIFEST_SCHEMA')
    from app.services.fund_quality_v2_market_calendar import ROLES, calendar_endpoint

    result = []
    for item in data['items']:
        if item.get('fund_code') != code:
            continue
        role = item.get('label')
        if role not in {CATALOG, HISTORY} | ROLES or not isinstance(item.get('receipt'), dict):
            raise ValueError('COMPONENT_MANIFEST_ROLE_OR_RECEIPT')
        proof = item['receipt']
        file = (source_root/proof['file']).resolve()
        if (not file.is_relative_to(source_root/'raw') or not file.is_file() or file.stat().st_size > 100_000
                or hashlib.sha256(file.read_bytes()).hexdigest() != proof.get('sha256')):
            raise ValueError('COMPONENT_DOWNLOAD_RECEIPT_ORIGINAL_PATH_HASH')
        receipt = read(file)
        if not isinstance(receipt, dict):
            raise ValueError('COMPONENT_DOWNLOAD_RECEIPT_STRUCTURE')
        raw_file = (source_root/item['file']).resolve()
        if (not raw_file.is_relative_to(source_root/'raw') or not raw_file.is_file()
                or raw_file.stat().st_size > 12_000_000):
            raise ValueError('COMPONENT_SOURCE_RAW_PATH_OR_SIZE')
        raw = raw_file.read_bytes()
        if hashlib.sha256(raw).hexdigest() != receipt.get('sha256'):
            raise ValueError('COMPONENT_SOURCE_RAW_ACTUAL_HASH')
        ref = {'label': role, 'component_code': 'HSCI', 'sha256': receipt['sha256'],
               'url': receipt['url'], 'acquired_at': receipt['received_at'],
               'component_receipt_sha256': proof['sha256'],
               'source_kind': 'INDEX_PUBLISHER_COMPONENT_HTTP_ORIGINAL_NOT_FULL_BENCHMARK',
               'origin_receipt_file': str(manifest)}
        if role in {CATALOG, HISTORY}:
            _query(ref, role=role)
        else:
            if 'calendar_year' in item:
                ref['calendar_year'] = item['calendar_year']
            ref['source_kind'] = 'COMPONENT_MARKET_CLOSURE_HTTP_ORIGINAL_NOT_PRICE_OR_BENCHMARK'
            calendar_endpoint(ref)
        for original, sha, label in [(raw, receipt['sha256'], role), (file.read_bytes(), proof['sha256'], RECEIPT)]:
            suffix = '.pdf' if original.startswith(b'%PDF') else '.html' if label in ROLES else '.json'
            target = rating_root/'sources'/(sha+suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != original:
                raise ValueError('COMPONENT_DESTINATION_BYTES_CONFLICT')
            if not target.exists():
                _publish(target, original)
            result.append(({**ref} if label == role else {
                'label': RECEIPT, 'sha256': sha, 'url': receipt['url'], 'acquired_at': receipt['received_at']})
                | {'file': target.relative_to(rating_root).as_posix()})
    # 转换和再次verify_packet时仍重读收据，而不是依靠本次通过的布尔标记。
    for ref in result:
        if ref['label'] in {CATALOG, HISTORY} | ROLES:
            _verified_original(ref, result, rating_root, frozen)
    return result
