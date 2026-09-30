"""利用同期原表与 2012 年正式分类码表修复截断标签，并关联历史持仓。

原表记录不删，任何时间点均只使用此前已公布的分类结果。最新披露分类的
年龄原样保留，2021 年后的旧分类不冒充同期确认，也不映射成中证指数成分。
"""
import re
from collections import Counter
from datetime import date
from scripts.fund_002112_data_completion_v1 import OUT,ROOT,read,save,sha,now

def run():
    folder=OUT/'industry-classification';taxonomy=read(folder/'taxonomy-2012.json');mapping={}
    for m in re.finditer(r'([A-S])\t\t[^\t]+\t\s*本门类包括([^大]+)大类',taxonomy['text']):
        nums=[int(x) for x in re.findall(r'\d{2}',m[2])]
        for v in range(nums[0],nums[-1]+1):mapping[f'{v:02d}']=m[1]
    assert len(mapping)==90 and mapping['80']=='O' and mapping['27']=='C'
    tables=[]
    for p in sorted((folder/'parsed-v2').glob('*.json')):
        d=read(p);corrections=[]
        for r in d['records']:
            category=r['category_code'];proof=None
            if category=='4950' and ((d['title'].startswith('2017年1季度') and r['page']==52) or (d['title'].startswith('2017年2季度') and r['page']==54)):
                category='49' if r['stock_code']=='603929' else '50';proof='VISUALLY_VERIFIED_ADJACENT_LABELS_ALIGNED_TO_603929_AND_000018'
            elif category=='5354' and d['title'].startswith('2017年1季度') and r['page']==57:
                category='53' if r['stock_code']=='601333' else '54';proof='VISUALLY_VERIFIED_ADJACENT_LABELS_ALIGNED_TO_601333_AND_000088'
            if category in mapping:
                code=mapping[category]+category
                if code!=r['industry_code'] or not r['structure_verified']:
                    corrections.append({'stock_code':r['stock_code'],'page':r['page'],'old_code':r['industry_code'],
                        'new_code':code,'original_category':r['category_code'],'proof':proof or 'EXPLICIT_CATEGORY_CODE_AND_OFFICIAL_2012_TAXONOMY'})
                r['industry_code']=code;r['category_code']=category;r['classification_code_verified']=True
                r['label_truncated_in_original']=not r['structure_verified'] and proof is None
            else:r['classification_code_verified']=False
        # 第 1 页额外标题行保留在旧结果中；不影响其后带六位股票代码的逐行绑定。
        issues=[i for i in d['issues'] if i!='PAGE_1_NONSTOCK_ROW_0']
        dest=folder/'parsed-v3'/p.name
        result={**d,'source_parse':str(p),'source_parse_sha256':sha(p),'taxonomy_source':str(folder/'taxonomy-2012.json'),
            'taxonomy_sha256':sha(folder/'taxonomy-2012.json'),'corrections':corrections,'remaining_issues':issues,
            'classification_codes_verified':not issues and not d['duplicate_stock_codes'] and all(r['classification_code_verified'] for r in d['records']),
            'original_invalid_rows_preserved_in_v2':True,'at':now()}
        if not dest.exists():save(dest,result)
        tables.append((result,dest))
    reports=read(ROOT/'holdings-compatibility/20260928-v1/reports.json')
    if isinstance(reports,dict):reports=reports['reports']
    bysha={r['raw']['sha256']:r for r in reports if r['fund_code']=='002112'}
    rows=[]
    for target in read(OUT/'historical-fund-relations.json')['rows']:
        eligible=[(d,p) for d,p in tables if d['available_at'] and d['available_at']<=target['as_of']]
        d,p=max(eligible,key=lambda x:x[0]['available_at']) if eligible else (None,None)
        original=bysha[target['report_sha256']];records={r['stock_code']:r for r in d['records']} if d else {}
        holdings=[]
        for h in original['holdings']:
            code=h['stock_code'];source=records.get(code.split('.')[0])
            holdings.append({'stock_code':code,'holding_name':h.get('stock_name'),'disclosed_nav_weight_pct':h.get('nav_weight_pct'),
                'classification':source if source and source.get('classification_code_verified') else None,
                'gap':None if source and source.get('classification_code_verified') else 'NOT_IN_LATEST_PUBLIC_CLASSIFICATION_OR_INVALID_SOURCE_ROW'})
        rows.append({'target':target['target'],'as_of':target['as_of'],'holding_report_sha256':target['report_sha256'],
            'holding_report_end':target['report_end'],'holding_disclosure_full':target['full_stock_disclosure'],
            'classification_source':str(p) if p else None,'classification_published_date':d['published_date'] if d else None,
            'classification_available_at':d['available_at'] if d else None,
            'classification_age_days':(date.fromisoformat(target['target'])-date.fromisoformat(d['published_date'])).days if d else None,
            'holdings':holdings,'all_disclosed_holdings_bound':all(h['classification'] for h in holdings),
            'basis':'LATEST_PREVIOUSLY_PUBLISHED_CLASSIFICATION_NOT_CONTINUOUSLY_CONFIRMED_MEMBERSHIP',
            'csi_industry_price_weighting_permitted':False,'training_eligible':False})
    save(OUT/'historical-industry-relations-v4.json',{'at':now(),'rows':rows,'target_rows':len(rows),
        'all_disclosed_holdings_bound_rows':sum(r['all_disclosed_holdings_bound'] for r in rows),
        'unmatched_stocks':dict(Counter(h['stock_code'] for r in rows for h in r['holdings'] if h['gap'])),
        'classification_periods':len(tables),'all_classification_codes_verified':all(d['classification_codes_verified'] for d,p in tables),
        'no_future_classification_used':all(r['classification_available_at']<=r['as_of'] for r in rows if r['classification_available_at']),
        'independent_sector_series_are_not_fund_holdings_proxy':True})

if __name__=='__main__':run()
