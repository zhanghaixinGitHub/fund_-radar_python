"""对 6167 份冻结原件逐项补身份、事实定位和版本关系；保留旧人工审核效力。"""
import json
import re
from collections import defaultdict, Counter
from datetime import date, timedelta

from scripts.fund_002112_data_completion_v1 import ROOT, OLD, OUT, read, save, sha, now
from app.services.fund_earnings_identity_v1 import profile_proof, supplement_identity

def doc_for(row):
    new=OUT/'historical'/(row['document_id']+'.json')
    if new.exists():
        d=read(new)
        if d.get('original_saved'):
            return d,new
    d=read(row['source_record'])
    if d.get('body_saved'):
        return d,__import__('pathlib').Path(row['source_record'])
    for ev in row.get('prior_evidence',[]):
        for field in ('path','source_file','document_path'):
            if ev.get(field):
                p=__import__('pathlib').Path(ev[field])
                if p.exists():
                    x=read(p)
                    if isinstance(x,dict) and x.get('pages') and x.get('receipt'):
                        return x,p
    # 无正文的旧停点按公告编号定位确切旧文档，只接受字节摘要可回读的原件。
    for p in (ROOT/'information-research').glob('*/documents/'+row['document_id']+'.json'):
        x=read(p)
        if x.get('body_saved') and x.get('receipt'):
            return x,p
    return d,__import__('pathlib').Path(row['source_record'])

def excerpts(doc, category):
    """保存必要事实的原页片段与位置，不将多个数字或不同期间自动绑定。"""
    terms = (r'归属于(?:上市公司|母公司)(?:所有者|股东)(?:的)?净利润|营业(?:总)?收入|报告期|追溯调整|会计政策变更'
        if category=='PERFORMANCE' else r'累计.*?回购|回购.*?股份|回购.*?金额|实施期限|尚未.*?回购|首次.*?回购|完成.*?回购'
        if category=='BUYBACK' else r'合同金额|合同总金额|生效|履行|履约|签订|签署|营业收入|中标|终止')
    result=[]
    for number,page in enumerate(doc.get('pages',[]),1):
        text=re.sub(r'\s+','',page)
        for m in re.finditer(terms,text):
            lo=max(0,m.start()-100);hi=min(len(text),m.end()+230)
            result.append({'page':number,'offset':m.start(),'keyword':m[0], 'quote':text[lo:hi]})
        # 每页最多提取8个必要片段，全文仍由文档引用保留，没有删困难原件。
        if len(result)>60:
            break
    return result[:64]

def run():
    audit=read(OLD/'admission-audit-v3/20260929-172820.json')
    rows=audit['rows'];profiles=defaultdict(list)
    cache=OUT/'historical-profile-references.json'
    if cache.exists():
        profiles.update(read(cache))
    else:
        for i,r in enumerate(rows):
            d,p=doc_for(r)
            if not d.get('pages'):
                continue
            proof=profile_proof(d)
            if proof:
                profiles[r['company']].append({'document':{'row':d['row'],'receipt':d['receipt'],'pages':d['pages'][:15]},'proof':proof,'source':str(p),'sha256':sha(p)})
        save(cache,dict(profiles))
    relations=defaultdict(list)
    for r in rows:
        d,p=doc_for(r)
        title=d['row'].get('title_plain','');period=re.search(r'(20\d{2})(?:年|年度)?(半年度|第一季度|一季度|第三季度|三季度|年度)',title)
        if period:
            relations[(r['company'],period[0])].append({'id':r['document_id'],'published':r['published_date'],'title':title,'stage':d['row'].get('stage')})
    for i,r in enumerate(rows,1):
        dest=OUT/'historical-audit'/(r['document_id']+'.json')
        if dest.exists():
            continue
        d,p=doc_for(r);pages=d.get('pages',[])
        body=bool(pages)
        rec={'id':r['document_id'],'company':r['company'],'published_date':r['published_date'],'category':r['category'],
            'at':now(),'source':str(p),'source_sha256':sha(p),'body_available':body,
            'prior_identity_verified':r['identity_after_exact_merge'],'prior_reviews':r['prior_evidence'],
            'old_stop_preserved':r['collection_stop_reason_preserved'],'old_revision_issues':r['revision_issues_preserved'],
            'full_semantics_verified':r.get('full_semantics_verified',False),'training_eligible':False}
        if body:
            identity=supplement_identity(d,profiles.get(r['company'],[]))
            rec.update(identity_supplement=identity, identity_verified=bool(r['identity_after_exact_merge'] or identity['passed']),
                revision_issues=d.get('revision_issues',[]),fact_evidence=excerpts(d,r['category']))
            title=d['row']['title_plain'];period=re.search(r'(20\d{2})(?:年|年度)?(半年度|第一季度|一季度|第三季度|三季度|年度)',title)
            rec['version_candidates_same_period']=relations.get((r['company'],period[0]),[]) if period else []
            rec['semantic_gap_fields']=(['PERIOD','METRIC_SCOPE','VALUE_UNIT_CURRENCY','ACCOUNTING_BASIS','PRIOR_PERIOD_AND_CORRECTION_CHAIN'] if r['category']=='PERFORMANCE'
                else ['BUYBACK_STAGE','ACTUAL_VS_PLANNED','SHARE_COUNT','AMOUNT_AND_UNIT','CUMULATIVE_ASOF_DATE'] if r['category']=='BUYBACK'
                else ['CONTRACT_STAGE','PARTIES','AMOUNT_UNIT_CURRENCY','EFFECTIVE_AND_PERFORMANCE_DATES','NOT_REVENUE'])
            rec['no_numeric_claim_from_keyword'] = True
        save(dest,rec)
        if i%200==0:print(json.dumps({'historical_audited':i,'total':len(rows)}),flush=True)
    saved=[read(p) for p in (OUT/'historical-audit').glob('*.json')]
    save(OUT/'historical-audit-result.json',{'at':now(),'scope':len(rows),'audited':len(saved),
        'body_available':sum(x['body_available'] for x in saved),'identity_verified':sum(x.get('identity_verified',False) for x in saved),
        'prior_reviews_preserved':True,'full_semantics_verified':sum(x['full_semantics_verified'] for x in saved),'new_fits':0})

if __name__=='__main__':run()
