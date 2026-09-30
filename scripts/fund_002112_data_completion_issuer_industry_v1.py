"""补核原冻结报告中发行人明确披露的历史行业，不采用当前行业标签。"""
import re
import sys
from collections import Counter
from datetime import date,timedelta
from scripts.fund_002112_data_completion_v1 import OUT,OLD,read,save,sha,now
from scripts.fund_002112_data_completion_semantics_v1 import doc_for

def run():
    relation=read(OUT/'historical-industry-relations-v4.json')
    expanded=len(sys.argv)>1 and sys.argv[1]=='all'
    wanted={h['stock_code'].split('.')[0] for r in relation['rows'] for h in r['holdings']} if expanded else {c.split('.')[0] for c in relation['unmatched_stocks']}
    output_folder='issuer-industry-v2' if expanded else 'issuer-industry'
    results=[]
    for r in read(OLD/'admission-audit-v3/20260929-172820.json')['rows']:
        d,p=doc_for(r);row=d['row'];code=row['secCode']
        if code not in wanted:continue
        dest=OUT/output_folder/f"{r['document_id']}.json"
        if dest.exists():results.append(read(dest));continue
        hits=[]
        for n,t in enumerate(d.get('pages',[]),1):
            t=re.sub(r'\s+','',t)
            for m in re.finditer(r'(?:本公司|发行人|公司)(?:主营业务|业务)?(?:所处|所属|所属于|属于|所在|的行业分类)[^。；]{0,180}',t):
                context=t[max(0,m.start()-150):m.end()+40]
                clause=re.split(r'根据(?:国家统计局|《国民)|按照(?:国家统计局|《国民)',m[0])[0]
                codes=re.findall(r'(?<![A-Z])([A-S]\d{2})(?!\d)',clause)
                if len(set(codes))!=1 or not re.search(r'证监会|上市公司行业分类',context):continue
                if re.search(r'子公司|客户|供应商|标的公司',t[max(0,m.start()-20):m.start()]):continue
                hits.append({'page':n,'offset':m.start(),'quote':context,'industry_code':codes[0],'literal':m[0]})
        codes={h['industry_code'] for h in hits}
        rec={'document_id':r['document_id'],'stock_code':code,'source':str(p),'source_sha256':sha(p),
            'raw_sha256':d.get('receipt',{}).get('sha256'),'published_date':row['published_date'],
            'available_at':str(date.fromisoformat(row['published_date'])+timedelta(days=1))+'T08:00:00+08:00',
            'hits':hits,'single_consistent_code':next(iter(codes)) if len(codes)==1 else None,
            'revision_issues':r['revision_issues_preserved'],'issuer_self_disclosure_not_regulator_roster':True,
            'original_identity_verified':read(OUT/'historical-identity-final'/f"{r['document_id']}.json")['identity_verified'] if (OUT/'historical-identity-final'/f"{r['document_id']}.json").exists() else read(OUT/'historical-audit'/f"{r['document_id']}.json")['identity_verified'],
            'training_eligible':False}
        save(dest,rec);results.append(rec)
    if expanded:
        save(OUT/'issuer-industry-result-v2.json',{'documents':len(results),'documents_with_consistent_code':sum(bool(d['single_consistent_code']) for d in results),
            'scope':'All stocks actually disclosed in the 1422 target rows; final historical association merged separately'})
        return
    bycode={}
    for d in results:
        if d['single_consistent_code'] and not d['revision_issues'] and d['original_identity_verified']:
            bycode.setdefault(d['stock_code'],[]).append(d)
    closed=[]
    for target in relation['rows']:
        for h in target['holdings']:
            if not h['gap']:continue
            candidates=[d for d in bycode.get(h['stock_code'].split('.')[0],[]) if d['available_at']<=target['as_of']]
            if not candidates:continue
            d=max(candidates,key=lambda x:x['available_at'])
            h['issuer_disclosed_classification']={k:d[k] for k in ['document_id','source','raw_sha256','published_date','available_at','single_consistent_code','hits']}
            h['gap']='REGULATOR_ROSTER_MISSING_ISSUER_DISCLOSURE_AVAILABLE'
            closed.append(h['stock_code'])
    relation.update({'at_issuer_enrichment':now(),'source_relation_v4_sha256':sha(OUT/'historical-industry-relations-v4.json'),
        'issuer_document_count':len(results),'issuer_disclosure_filled_holding_rows':len(closed),
        'issuer_disclosure_filled_stocks':dict(Counter(closed)),
        'remaining_without_historical_classification':dict(Counter(h['stock_code'] for r in relation['rows'] for h in r['holdings'] if h['gap'] and not h.get('issuer_disclosed_classification')))})
    save(OUT/'historical-industry-relations-v5.json',relation)
    save(OUT/'issuer-industry-result.json',{'documents':len(results),'documents_with_consistent_code':sum(bool(d['single_consistent_code']) for d in results),
        'filled_rows':len(closed),'filled_stock_codes':sorted(set(closed)),
        'unresolved_stock_count':len(relation['remaining_without_historical_classification'])})

if __name__=='__main__':run()
