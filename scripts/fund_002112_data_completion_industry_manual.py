"""五项剩余行业差异的原文复核：可证明的名称映射与粒度不足明确区分。"""
from collections import Counter
from scripts.fund_002112_data_completion_v1 import OUT,read,save,sha,now
import re

def run():
    relation=read(OUT/'historical-industry-relations-v8.json');proofs={}
    taxonomy=OUT/'industry-classification/taxonomy-2012.json';text=read(taxonomy)['text']
    definitions={
      '600968':('association-supplemental-sources',17,'B11','开采辅助活动','EXPLICIT_CODE'),
      '601698':('association-supplemental-sources',36,'I63','电信、广播电视和卫星传输服务','EXACT_OFFICIAL_2012_NAME_MAPPING'),
      '688235':('association-listing-sources',33,'C27','医药制造业','EXACT_OFFICIAL_2012_NAME_MAPPING'),
      '600956':('association-supplemental-sources',27,None,'D电力、热力、燃气及水生产和供应业','SECTION_ONLY_NOT_SUBINDUSTRY'),
      '301007':('association-corrected-sources',13,None,'汽车零部件及配件制造业','BUSINESS_AND_PRODUCTION_PROCESS_CLASSIFICATIONS_DIFFER'),
    }
    for code,(folder,page,value,literal,method) in definitions.items():
        p=OUT/folder/(code+'.json');d=read(p);t=re.sub(r'\s+','',d['pages'][page-1]);assert literal in t
        if method=='EXACT_OFFICIAL_2012_NAME_MAPPING':assert literal in text
        if method=='EXPLICIT_CODE':assert value in t
        q=t[max(0,t.index(literal)-130):t.index(literal)+len(literal)+170]
        proofs[code]={'stock_code':code,'industry_code':value,'source':str(p),'source_sha256':sha(p),'raw_sha256':d['receipt']['sha256'],
            'published_date':d['row']['published_date'],'available_at':d['available_at'],'page':page,'quote':q,'method':method,
            'taxonomy_source':str(taxonomy),'taxonomy_sha256':sha(taxonomy),'training_eligible':False}
        save(OUT/'industry-manual-review'/(code+'.json'),proofs[code])
    filled=0;remaining=[]
    for row in relation['rows']:
        for h in row['holdings']:
            code=h['stock_code'].split('.')[0];d=proofs.get(code)
            if d and not h['selected_evidence']:
                h['manual_review']=d
                if d['industry_code'] and d['available_at']<=row['as_of']:
                    h['selected_evidence']={**d,'basis':'ISSUER_DISCLOSURE_WITH_EXACT_2012_TAXONOMY'};h['gap']=None;filled+=1
                else:h['gap']=d['method'] if not d['industry_code'] else 'CLASSIFICATION_NOT_PUBLIC_BEFORE_TARGET'
            if h['gap']:remaining.append({'target':row['target'],'stock_code':code,'reason':h['gap'],'source':d['source'] if d else None})
        row['all_disclosed_holdings_bound']=all(h['selected_evidence'] for h in row['holdings'])
    relation.update(at=now(),unresolved=remaining,all_disclosed_holdings_bound_rows=sum(r['all_disclosed_holdings_bound'] for r in relation['rows']),
        manual_filled_holding_rows=filled,all_granularity_conflicts_preserved=True)
    save(OUT/'historical-industry-relations-final.json',relation)
    save(OUT/'historical-industry-relations-result-final.json',{'at':now(),'targets':len(relation['rows']),'complete_target_rows':relation['all_disclosed_holdings_bound_rows'],
        'newly_bound_holding_rows':filled,'remaining_holding_rows':len(remaining),'remaining_stocks':dict(Counter(r['stock_code'] for r in remaining)),
        'evidence_cutoffs_passed':all(h['selected_evidence']['available_at']<=r['as_of'] for r in relation['rows'] for h in r['holdings'] if h['selected_evidence']),
        'remaining_reasons':dict(Counter(r['reason'] for r in remaining))})

if __name__=='__main__':run()
