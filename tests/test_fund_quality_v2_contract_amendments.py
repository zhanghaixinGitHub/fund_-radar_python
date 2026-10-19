"""修订原件准入的反例；所有HTML和收据均为工程假样本，不是校准材料。"""

import hashlib
import json
from datetime import UTC, date, datetime

import pytest
from app.services.fund_materials_store import versioned_save
from app.services.fund_quality_v2_contract_amendments import FEE_TITLE, verify_cmf_fee_amendment
from app.services.fund_quality_v2_source_reuse import reuse_originals


def archive(tmp_path, mutation=None):
    source, destination = tmp_path/'input', tmp_path/'rating'
    raw_dir = source/'raw'
    raw_dir.mkdir(parents=True)
    name = "测试股票型证券投资基金"
    html = (f'<meta charset="utf-8"><div class="article_detail_title"><div class="wrapfix">'
            f'<h2>{FEE_TITLE}</h2><div class="data">2023-07-08</div></div></div>'
            '<div class="article_detail_box">本公司决定自2023年7月10日起，调低旗下部分基金的管理费率及托管费率'
            '并对基金合同有关条款进行修订，同时对相关基金的基金管理人、基金托管人信息进行必要更新。'
            '除上述要素外，其余要素不作变更。<table><tr><td>1</td><td>' + name + '</td>'
            '<td>1.50%</td><td>0.25%</td><td>1.20%</td><td>0.20%</td></tr></table></div>')
    if mutation == 'wrong_fund':
        html = html.replace(name, '其他证券投资基金')
    if mutation == 'duplicate_row':
        html = html.replace('</table>', html[html.index('<tr>'):html.index('</table>')]+'</table>')
    if mutation == 'wrong_publication':
        html = html.replace('2023-07-08', '2023-07-09')
    if mutation == 'scope_gap':
        html = html.replace('其余要素不作变更', '其他事项另行公告')
    raw = html.encode()
    sha = hashlib.sha256(raw).hexdigest()
    html_file = raw_dir/(sha+'.html')
    html_file.write_bytes(raw)
    url = 'https://www.cmfchina.com/web/fundnoticedetail/123456/index.html'
    if mutation == 'wrong_endpoint':
        url = url.replace('www.cmfchina.com', 'www.cmfchina.com.example.org')
    received = '2026-10-17T15:00:00+08:00'
    receipt = {'url': url, 'status': 200, 'bytes': len(raw), 'sha256': sha, 'received_at': received,
               'file': str(html_file)}
    if mutation == 'bad_status':
        receipt['status'] = 500
    if mutation == 'wrong_size':
        receipt['bytes'] += 1
    proof = raw_dir/'receipt.json'
    versioned_save(proof, receipt)
    proof_sha = hashlib.sha256(proof.read_bytes()).hexdigest()
    meta = {'fund_code': '123456', 'title': FEE_TITLE, 'published_date': '2023-07-08',
            'source_kind': 'ISSUER_CMF_CONTRACT_AMENDMENT_HTML',
            'receipt': {**receipt, 'file': html_file.relative_to(source).as_posix()},
            'contract_notice_receipt': {'file': 'raw/receipt.json', 'sha256': proof_sha}}
    if mutation == 'boolean_proof':
        meta['contract_notice_receipt'] = True
    metadata = source/'fund-documents/123456/notice.json'
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
    refs = reuse_originals('123456', name, source, destination, date(2026, 9, 30),
                          datetime(2026, 10, 17, 23, tzinfo=UTC))
    return destination, refs, name


def test_actual_html_and_receipt_route_through_archive_and_original_parser(tmp_path):
    root, refs, name = archive(tmp_path)
    ref = next(r for r in refs if r['label'] == 'contract-amendment-notice')
    result = verify_cmf_fee_amendment((root/ref['file']).read_bytes(), ref, refs=refs,
                                     root=root, legal_name=name, as_of=date(2026, 9, 30))
    assert result['effective_on'] == '2023-07-10'
    assert result['legal_name'] == name and len(result['source_hashes']) == 2
    assert 'NOT_COMPLETE' in result['scope']


@pytest.mark.parametrize('mutation', ['wrong_fund', 'duplicate_row', 'wrong_publication', 'scope_gap',
                                     'wrong_endpoint', 'bad_status', 'wrong_size', 'boolean_proof'])
def test_metadata_or_boolean_claim_cannot_admit_an_unbound_original(tmp_path, mutation):
    with pytest.raises(ValueError):
        root, refs, name = archive(tmp_path, mutation)
        ref = next(r for r in refs if r['label'] == 'contract-amendment-notice')
        verify_cmf_fee_amendment((root/ref['file']).read_bytes(), ref, refs=refs,
                                root=root, legal_name=name, as_of=date(2026, 9, 30))


def test_archive_byte_mutation_and_after_cutoff_publication_are_rejected(tmp_path):
    root, refs, name = archive(tmp_path)
    ref = next(r for r in refs if r['label'] == 'contract-amendment-notice')
    raw = (root/ref['file']).read_bytes()
    with pytest.raises(ValueError, match='ACTUAL_HASH'):
        verify_cmf_fee_amendment(raw+b'changed', ref, refs=refs, root=root,
                                legal_name=name, as_of=date(2026, 9, 30))
    with pytest.raises(ValueError, match='PUBLICATION'):
        verify_cmf_fee_amendment(raw, ref, refs=refs, root=root, legal_name=name, as_of=date(2023, 7, 7))
