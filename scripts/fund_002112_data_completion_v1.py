"""002112 本轮补数工具；只追加证据，绝不调用训练、登记、预测或旧采集入口。

同一输出目录按请求键去重，请求发出前记账。旧账、停止原因和原件均只读。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import re
import time
import ssl
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

PY = Path(__file__).resolve().parents[1]
WEB = Path('C:/WebStormProject/workSpace05')
JAVA = Path('C:/ideaProject/workSpace12')
ROOT = PY / '.local-runs/fund-exposure-002112'
OLD = ROOT / 'closure/20260929-v1'
OUT = ROOT / 'data-completion/20260930-v1'
MANIFEST = WEB / 'docs_zhx/implementation/fund-002112-data-completion-manifest-2026-09-30.json'

def now():
    return datetime.now().astimezone().isoformat()

def read(path):
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if isinstance(value, dict) and set(value) == {'hash', 'payload'}:
        from app.services.direction_1d_protocol import digest
        if digest(value['payload']) != value['hash']:
            raise ValueError('WRAPPED_EVIDENCE_HASH_MISMATCH')
        return value['payload']
    return value

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(path, value):
    """排他写入，不覆盖任何上一轮证据。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)

def protect():
    """保护三仓未提交源文件与研究文件摘要；不读取封存答案内容。"""
    if (OUT / 'protection-before.json').exists():
        return
    OUT.mkdir(parents=True, exist_ok=True)
    workspaces = []
    files = {}
    for name, repo in [('web', WEB), ('python', PY), ('java', JAVA)]:
        def git(*args):
            return subprocess.check_output(['git', '-C', str(repo), *args])
        status = git('status', '--porcelain=v1', '-uall').decode('utf-8')
        paths = set(git('diff', '--name-only', '-z').split(b'\0'))
        paths |= set(git('ls-files', '--others', '--exclude-standard', '-z').split(b'\0'))
        paths |= set(git('diff', '--cached', '--name-only', '-z').split(b'\0'))
        for rel in paths:
            if not rel:
                continue
            p = repo / rel.decode('utf-8')
            if p.is_file() and p != Path(__file__):
                files[str(p)] = sha(p)
                dest = OUT / 'workspace-backup' / name / p.relative_to(repo)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
        workspaces.append({'path': str(repo), 'head': git('rev-parse', 'HEAD').decode().strip(), 'status': status})
    # 字节摘要只用于完整性核对，不解析任何封存标签和旧预测值。
    count = 0
    for p in ROOT.rglob('*'):
        if p.is_file() and OUT not in p.parents:
            files[str(p)] = sha(p)
            count += 1
    m = read(MANIFEST)
    source_checks = []
    for s in [m['source_decision'], m['source_audit'], m['current_news_source']]:
        source_checks.append({'path': s['path'], 'passed': sha(s['path']).lower() == s['sha256'].lower()})
    for item in m['historical_missing_originals'] + m['historical_version_conflicts']:
        source_checks.append({'path': item['source_record'], 'passed': sha(item['source_record']) == item['source_record_sha256']})
    assert all(s['passed'] for s in source_checks)
    ledger = ROOT / 'information-research/20260928-v1'
    budget = read(ledger / 'budget-authorized.json')
    fits = list((ledger / 'fit-ledger').glob('*.json'))
    save(OUT / 'protection-before.json', {'at': now(), 'files': files, 'workspaces': workspaces,
        'source_checks': source_checks, 'historical_file_count': count,
        'old_body_requests_consumed': len(list((OLD / 'company-bodies/requests').glob('*.json'))),
        'cumulative_fits': budget['previous_actual_fits'] + len(fits), 'new_fits': 0})
    print(json.dumps({'protection': len(files), 'historical': count, 'sources_passed': len(source_checks)}, ensure_ascii=False), flush=True)

def inventory():
    """只搜既有回执和确切编号，优先复用已保存字节，原停止含义逐条保留。"""
    if (OUT / 'inventory.json').exists():
        return read(OUT / 'inventory.json')
    m = read(MANIFEST)
    wanted = {x['document_id'] for x in m['historical_missing_originals'] + m['historical_version_conflicts']}
    stops = {x['document_id']: x for x in read(ROOT / 'information-research/20260929-earnings-v24/source-stop-registry.json')['entries']}
    index = defaultdict(list)
    for directory in [ROOT, PY / 'data/fund-materials', WEB / '.local-runs/implementation-20260929-145323']:
        for p in directory.rglob('*.json'):
            if OUT in p.parents or ('receipts' not in p.parts and p.stem not in wanted):
                continue
            try:
                obj = read(p)
                if not isinstance(obj, dict):
                    continue
                for s in [obj, obj.get('receipt', {}), obj.get('source', {})]:
                    if s.get('url') and s.get('path') and s.get('sha256') and Path(s['path']).exists():
                        if sha(s['path']).lower() == s['sha256'].lower():
                            if not any(x['sha256'] == s['sha256'] for x in index[s['url']]):
                                index[s['url']].append({**s, 'receipt_file': str(p)})
            except (ValueError, OSError, TypeError):
                continue
    historical = []
    for x in m['historical_missing_originals'] + m['historical_version_conflicts']:
        row = x['original_catalog_row']
        url = 'https://static.cninfo.com.cn/' + row['adjunctUrl']
        historical.append({'id': x['document_id'], 'row': row, 'url': url,
            'category': 'version' if x in m['historical_version_conflicts'] else 'missing',
            'cached': index.get(url, []), 'old_stop': stops.get(x['document_id']),
            'old_reason': x.get('collection_stop_reason_preserved')})
    public = {}
    for rel in ['public-catalogs/catalog-104.json', 'public-catalogs/catalog-46.json', 'public-news-current-snapshot/result.json']:
        for row in read(OLD / rel)['entries_in_range']:
            public.setdefault(row['url'], {'url': row['url'], 'aliases': [], 'catalogs': [], 'cached': index.get(row['url'], [])})
            public[row['url']]['aliases'].append(row)
            public[row['url']]['catalogs'].append(str(OLD / rel))
    result = {'at': now(), 'historical': historical, 'public': list(public.values()), 'cache_index': index,
        'scope': 'original frozen company list; registered public columns 2016-12-24..2023-12-31', 'new_fits': 0}
    save(OUT / 'inventory.json', result)
    save(OUT / 'collection-contract.json', {'at': now(), 'user_authorization': '2026-09-30 exact data-completion request',
        'old_requests_consumed': 3523, 'old_request_budget_remaining': 0,
        'incremental_limits': {'historical': 46, 'version_official': 40, 'alternatives': 120,
            'public': len(public), 'attachments': 500, 'industry': 160},
        'max_bytes_per_pdf': 128_000_000, 'max_pages': 1200, 'max_total_new_bytes': 8_000_000_000,
        'max_retries': 0, 'interval_seconds': 0.3,
        'changed_conditions': ['streaming preserves partial downloads', 'PDF byte limit raised from 16MB to 128MB for exact missing IDs',
            'new explicit request covers earlier unattempted per-batch count stops; old resource stops unchanged',
            'semantic stops first reuse cached originals; never imply semantic clearance from download'],
        'old_scope_and_stops_unchanged': True, 'fit_budget_for_this_stage': 0})
    print(json.dumps({'historical_cached': sum(bool(x['cached']) for x in historical), 'historical': len(historical),
        'public_unique': len(public), 'public_cached': sum(bool(x['cached']) for x in public.values())},ensure_ascii=False), flush=True)
    return result

def fetch(url, group, maximum_bytes=128_000_000):
    """有界串行公开读取。失败记账不重试；HTTP 拒绝不更换身份绕过。"""
    import httpx
    key = hashlib.sha256(url.encode()).hexdigest()
    receipt_path = OUT / 'receipts' / f'{key}.json'
    if receipt_path.exists():
        return read(receipt_path)
    request = OUT / 'requests' / f'{key}.json'
    if request.exists():
        return {'url': url, 'ok': False, 'reason': 'INTERRUPTED_REQUEST_PRESERVED_NO_RETRY', 'request': str(request)}
    contract = read(OUT / 'collection-contract.json')
    supplement = OUT / 'recovery-limits.json'
    if supplement.exists():
        contract['incremental_limits'].update(read(supplement)['incremental_limits'])
    association_scope = OUT / 'association-source-scope.json'
    if association_scope.exists():
        contract['incremental_limits']['association_original'] = len(read(association_scope)['stocks'])
    image_scope = OUT / 'public-image-exact-scope.json'
    if group == 'body_media':
        scope_path=OUT/'public-body-media-scope-v5.json'
        if not scope_path.exists():scope_path=OUT/'public-body-media-scope-v4.json'
        if not scope_path.exists():scope_path=OUT/'public-body-media-scope-v3.json'
        scope=read(scope_path if scope_path.exists() else OUT/'public-body-media-scope-v2.json')
        if url not in scope['urls']:raise ValueError('BODY_MEDIA_OUTSIDE_FROZEN_DOCUMENT_SCOPE')
        contract['incremental_limits']['body_media']=len(scope['urls'])
    if image_scope.exists() and group == 'article_images':
        image_urls = set(read(image_scope)['urls'])
        if url not in image_urls:
            raise ValueError('ARTICLE_IMAGE_OUTSIDE_EXACT_FROZEN_SCOPE')
        contract['incremental_limits']['article_images'] = len(image_urls)
    used = sum(read(p)['group'] == group for p in (OUT / 'requests').glob('*.json'))
    if used >= contract['incremental_limits'][group]:
        raise ValueError('INCREMENTAL_BUDGET_EXHAUSTED:' + group)
    total = sum(p.stat().st_size for p in (OUT / 'raw').glob('*'))
    if total >= contract['max_total_new_bytes']:
        raise ValueError('TOTAL_NEW_BYTES_EXHAUSTED')
    save(request, {'url': url, 'group': group, 'at': now(), 'ordinal': used + 1})
    raw = OUT / 'raw' / key
    raw.parent.mkdir(exist_ok=True)
    result = {'url': url, 'group': group, 'ok': False, 'at': now()}
    time.sleep(contract['interval_seconds'])
    try:
        with httpx.Client(timeout=httpx.Timeout(90, connect=10), follow_redirects=False, trust_env=False,
                          verify=ssl.create_default_context()) as client:
            with client.stream('GET', url) as response:
                result.update(status=response.status_code, headers={k: response.headers[k] for k in
                    ('content-type','content-length','last-modified','etag','location') if k in response.headers})
                response.raise_for_status()
                size = 0
                with raw.open('xb') as f:
                    for chunk in response.iter_bytes(131072):
                        size += len(chunk)
                        if size > min(maximum_bytes, contract['max_total_new_bytes']-total):
                            raise ValueError('NEW_RESPONSE_SIZE_LIMIT')
                        f.write(chunk)
        result.update(ok=True, path=str(raw), sha256=sha(raw), bytes=size, received_at=now())
    except (httpx.HTTPError, OSError, ValueError) as e:
        result.update(reason=type(e).__name__ + ':' + str(e)[:400], partial_path=str(raw) if raw.exists() else None)
    save(receipt_path, result)
    return result

def historical():
    """覆盖全部 50 个历史缺口；取得、身份、时间、语义分别记录。"""
    from app.services.fund_earnings_batch_v4 import extract_pdf, issuer_identity
    from app.services.fund_information_history_v1 import pdf_revision
    inv = inventory()
    for spec in inv['historical']:
        dest = OUT / 'historical' / (spec['id'] + '.json')
        if dest.exists():
            continue
        result = {**spec, 'at': now(), 'original_saved': False, 'training_eligible': False,
            'full_semantics_verified': False, 'historical_relation_verified': False}
        if spec['cached']:
            receipt = spec['cached'][0]
        elif spec['category'] == 'version':
            result['reason'] = 'VERSION_REQUIRES_DIFFERENT_OFFICIAL_ORIGINAL'
            save(dest, result)
            continue
        else:
            receipt = fetch(spec['url'], 'historical')
        result['receipt'] = receipt
        if receipt.get('ok', True):
            try:
                path = Path(receipt['path'])
                assert sha(path) == receipt['sha256']
                pages, metadata = extract_pdf(path.read_bytes(), 1200)
                result.update(original_saved=True, pages=pages, metadata=metadata,
                    identity=issuer_identity(spec['row'], pages),
                    revision_issues=pdf_revision(metadata, spec['row']['published_date']),
                    cached_reused=bool(spec['cached']))
            except Exception as e:
                result['reason'] = type(e).__name__ + ':' + str(e)[:400]
        else:
            result['reason'] = receipt.get('reason')
        save(dest, result)
        print(json.dumps({'id':spec['id'], 'saved':result['original_saved'],
            'pages':len(result.get('pages',[])), 'identity':result.get('identity',{}).get('passed'),
            'date_issues':result.get('revision_issues'), 'reason':result.get('reason')},ensure_ascii=False),flush=True)

if __name__ == '__main__':
    if sys.argv[1] == 'protect':
        protect()
    elif sys.argv[1] == 'inventory':
        inventory()
    elif sys.argv[1] == 'historical':
        historical()
