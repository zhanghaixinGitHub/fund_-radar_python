"""核验新增历史发行文件并合并当时已公开的行业依据，保留监管原表冲突。"""
import re
import sys
from collections import defaultdict, Counter
from datetime import date
from scripts.fund_002112_data_completion_v1 import OUT, read, save, sha, now

def norm(s):
    return re.sub(r'\s+', '', s or '')

def run():
    version=sys.argv[1] if len(sys.argv)>1 else 'v1'
    new=version!='v1'
    latest=version not in ('v1','v2')
    sources=[]
    folders=['association-sources','association-listing-sources']+(['association-corrected-sources'] if new else [])
    corrected={p.name for p in (OUT/'association-corrected-sources').glob('*.json')} if new else set()
    for folder in folders:
        for p in sorted((OUT/folder).glob('*.json')):
            d=read(p)
            if not d.get('pages'): continue
            if p.name in corrected and folder!='association-corrected-sources':continue
            dest=OUT/('association-audit-'+version if new else 'association-audit')/p.name
            if dest.exists(): sources.append(read(dest)); continue
            row=d['row']; stock=d['stock']; pages=[norm(t) for t in d['pages']]
            bridge_path=OUT/'association-supplemental-sources'/p.name
            bridge=read(bridge_path) if latest and bridge_path.exists() else None
            catalog=read(d['catalog_receipt']['path'])
            binding=[r for r in catalog.get('announcements',[]) if r.get('announcementId')==row['announcementId'] and r.get('secCode')==stock['code'] and r.get('adjunctUrl')==row['adjunctUrl'] and r.get('announcementTime')==row['announcementTime']]
            assert sha(d['receipt']['path'])==d['receipt']['sha256']
            names={row.get('secName',''),stock['name_at_holding']}
            names={re.sub(r'^(?:N|C|\*?ST)', '', n) for n in names if n}
            identity=[{'page':i+1,'name':n,'quote':t[max(0,t.index(n)-50):t.index(n)+len(n)+80]} for i,t in enumerate(pages[:20]) for n in names if len(n)>=2 and n in t]
            identity_bridge=None
            if not identity and bridge:
                bp=[norm(t) for t in bridge.get('pages',[])];samecode=bridge['row']['secCode']==stock['code']
                shortproof=any(n in t for t in bp[:20] for n in names if len(n)>=2)
                legal=re.findall(r'[\u4e00-\u9fff]{4,35}股份有限公司',pages[0])
                matched=next((n for n in legal if any(n in t for t in bp[:20])),None)
                if samecode and shortproof and matched and not bridge['revision_issues']:
                    identity=[{'page':1,'name':matched,'quote':matched}]
                    identity_bridge={'source':str(bridge_path),'sha256':sha(bridge_path),'available_at':bridge['available_at'],'same_full_legal_name':matched,'same_code_and_shortname_in_listing':True}
            title=any(('招股说明书' in t or '上市公告书' in t) and re.search(r'首次(?:A股)?公开发行',t) for t in pages[:20])
            hits=[]
            for i,t in enumerate(pages):
                for m in re.finditer(r'(?:本公司|发行人|公司)(?:主营业务|业务)?(?:所处|所属|所属于|属于|所在|的行业分类)[^。；]{0,240}',t):
                    context=t[max(0,m.start()-150):m.end()+40]
                    clause=re.split(r'根据(?:国家统计局|《国民)|按照(?:国家统计局|《国民)',m[0])[0]
                    codes=re.findall(r'(?<![A-Z])([A-S]\d{2})(?!\d)',clause)
                    # 防止后一国民经济标准条款借用前一证监会标准的上下文。
                    nearest=t[max(0,m.start()-150):m.start()]
                    wrong_standard=max(nearest.rfind('国民经济'),nearest.rfind('国家统计局'))>max(nearest.rfind('证监会'),nearest.rfind('上市公司行业分类'))
                    if len(set(codes))==1 and not wrong_standard and re.search(r'证监会|上市公司行业分类',context) and not re.search(r'子公司|客户|供应商|标的公司|可比公司',t[max(0,m.start()-20):m.start()]):
                        hits.append({'page':i+1,'quote':context,'literal':clause,'industry_code':codes[0]})
                if new:
                    # 从明确的分类指引句起读至下一标准，不跨用客户或可比公司行业。
                    for m in re.finditer(r'上市公司行业分类[^。；]{0,240}',t):
                        clause=re.split(r'根据(?:国家|《国民)|按照(?:国家|《国民)',m[0])[0]
                        codes=re.findall(r'(?<![A-Z])([A-S]\d{2})(?!\d)',clause)
                        if len(set(codes))==1 and re.search(r'(?:公司|发行人|'+re.escape(stock['name_at_holding'])+r').{0,35}(?:属于|所属|所从事|行业)',clause) and not re.search('可比公司|子公司|客户|供应商|与发行人分属|与公司分属',clause):
                            hits.append({'page':i+1,'quote':clause,'literal':clause,'industry_code':codes[0]})
                    # 上市公告书公司基本资料中的明确行业栏保留原代码；不截断四位国民经济代码。
                    if '上市公告书' in row['title_plain']:
                        for m in re.finditer(r'所属行业[：:]?[^。；]{0,130}',t):
                            codes=re.findall(r'(?<![A-Z])([A-S]\d{2})(?!\d)',m[0])
                            if len(set(codes))==1:
                                hits.append({'page':i+1,'quote':m[0],'literal':m[0],'industry_code':codes[0],'basis':'EXPLICIT_LISTING_ISSUER_PROFILE_FIELD'})
            supplemental_hits=[]
            if latest and bridge and not bridge['revision_issues']:
                # 同一截止日前的上市公告提供更明确的股票简称和行业栏，原招股证据不覆盖。
                for i,txt in enumerate(bridge['pages']):
                    txt=norm(txt)
                    for m in re.finditer(r'(?:上市公司行业分类|所属行业)[^。；]{0,220}',txt):
                        q=re.split('可比上市公司|截至|邮政编码|董事会秘书',m[0])[0]
                        codes=re.findall(r'(?<![A-Z])([A-S]\d{2})(?!\d)',q)
                        if len(set(codes))==1 and not re.search('可比公司|子公司|客户|供应商|与发行人分属',q):
                            supplemental_hits.append({'page':i+1,'quote':q,'literal':q,'industry_code':codes[0],'source':str(bridge_path)})
                if supplemental_hits and len({x['industry_code'] for x in supplemental_hits})==1:
                    hits=supplemental_hits
            codes={h['industry_code'] for h in hits}
            rec={'at':now(),'stock_code':stock['code'],'document_id':row['announcementId'],'source':str(p),'source_sha256':sha(p),'raw_sha256':d['receipt']['sha256'],
                 'published_date':row['published_date'],'available_at':d['available_at'],'catalog_binding_verified':len(binding)==1,
                 'identity_verified':bool(len(binding)==1 and identity and title),'identity_anchors':identity,'document_title_verified':title,
                 'revision_issues':d['revision_issues'],'hits':hits,'single_consistent_code':next(iter(codes)) if len(codes)==1 else None,
                 'basis':'ISSUER_SELF_DISCLOSED_CSRC_CLASSIFICATION','training_eligible':False,'gaps':[]}
            rec['document_status']='PRE_DISCLOSURE_DRAFT' if re.search('不具有据以发行股票|预先披露之用',''.join(pages[:3])) else 'PUBLISHED_ISSUER_DOCUMENT'
            rec['identity_bridge']=identity_bridge
            rec['supplemental_listing_source']=str(bridge_path) if supplemental_hits else None
            if identity_bridge or supplemental_hits:rec['available_at']=max(rec['available_at'],bridge['available_at'])
            rec['status_does_not_imply_actual_listing']=True
            if not rec['identity_verified']: rec['gaps'].append('ISSUER_BODY_IDENTITY_UNVERIFIED')
            if rec['revision_issues']: rec['gaps'].append('LATE_PDF_REVISION')
            if not codes: rec['gaps'].append('NO_EXPLICIT_CSRC_ISSUER_CLASSIFICATION_FOUND')
            if len(codes)>1: rec['gaps'].append('MULTIPLE_CLASSIFICATION_CODES_REQUIRE_CONTEXT')
            save(dest,rec); sources.append(rec)
    save(OUT/('association-audit-result-'+version+'.json' if new else 'association-audit-result.json'),{'at':now(),'documents':len(sources),'identity_verified':sum(d['identity_verified'] for d in sources),'industry_verified':sum(not d['gaps'] for d in sources),'gaps':dict(Counter(g for d in sources for g in d['gaps']))})
    bycode=defaultdict(list)
    for p in (OUT/'issuer-industry-v2').glob('*.json'):
        d=read(p)
        extra=OUT/'historical-identity-additional'/p.name
        if extra.exists(): d['original_identity_verified']=d['original_identity_verified'] or read(extra)['identity_verified']
        if d['single_consistent_code'] and not d['revision_issues'] and d['original_identity_verified']:
            bycode[d['stock_code']].append({**d,'basis':'ISSUER_SELF_DISCLOSED_CSRC_CLASSIFICATION'})
    for d in sources:
        if not d['gaps']: bycode[d['stock_code']].append(d)
    relation=read(OUT/'historical-industry-relations-v4.json');tables={};unresolved=[];conflicts=[]
    for row in relation['rows']:
        path=row['classification_source']
        if path not in tables:
            records=defaultdict(list)
            for r in read(path)['records']: records[r['stock_code']].append(r)
            tables[path]=records
        for h in row['holdings']:
            code=h['stock_code'].split('.')[0]; choices=tables[path].get(code,[])
            if len(choices)>1:
                exact=[c for c in choices if norm(c['stock_name'])==norm(h['holding_name'])]
                h['classification']=exact[0] if len(exact)==1 else None
                h['original_duplicate_rows']=choices
                h['duplicate_resolution']='EXACT_HOLDING_NAME' if len(exact)==1 else 'UNRESOLVED'
                conflicts.append({'target':row['target'],'stock':code,'resolution':h['duplicate_resolution']})
            selected=None
            if h['classification'] and h['classification'].get('classification_code_verified'):
                selected={'industry_code':h['classification']['industry_code'],'available_at':row['classification_available_at'],
                          'published_date':row['classification_published_date'],'basis':'REGULATOR_PUBLISHED_ROSTER','source':path,'row':h['classification']}
            docs=[d for d in bycode.get(code,[]) if d['available_at']<=row['as_of']]
            if docs:
                latest=max(docs,key=lambda d:d['available_at']);h['latest_issuer_disclosure']=latest
                if not selected or latest['available_at']>selected['available_at']:
                    selected={'industry_code':latest['single_consistent_code'],'available_at':latest['available_at'],'published_date':latest['published_date'],
                              'basis':latest['basis'],'source':latest['source'],'document_id':latest['document_id'],'hits':latest['hits']}
            h['selected_evidence']=selected
            h['gap']=None if selected else 'NO_PREVIOUSLY_PUBLISHED_VERIFIED_CLASSIFICATION'
            if selected:
                assert selected['available_at']<=row['as_of']
                h['evidence_age_days']=(date.fromisoformat(row['target'])-date.fromisoformat(selected['published_date'])).days
            else: unresolved.append({'target':row['target'],'stock_code':code,'name':h['holding_name'],'as_of':row['as_of']})
        row['all_disclosed_holdings_bound']=all(h['selected_evidence'] for h in row['holdings'])
    relation.update(at=now(),all_disclosed_holdings_bound_rows=sum(r['all_disclosed_holdings_bound'] for r in relation['rows']),
                    unresolved=unresolved,original_duplicate_resolutions=conflicts,
                    selection_rule='LATEST_PREVIOUSLY_PUBLISHED_PROVEN_EVIDENCE_WITH_SOURCE_TYPE_AND_AGE_PRESERVED',
                    no_current_industry_profile=True,no_csi_membership_or_weights_inferred=True)
    rv='v8' if latest else 'v7' if new else 'v6'
    save(OUT/('historical-industry-relations-'+rv+'.json'),relation)
    save(OUT/('historical-industry-relations-result-'+rv+'.json'),{'at':now(),'target_rows':len(relation['rows']),'complete_rows':relation['all_disclosed_holdings_bound_rows'],
        'unresolved_holding_rows':len(unresolved),'unresolved_stocks':dict(Counter(r['stock_code'] for r in unresolved)),'original_duplicate_resolutions':len(conflicts),'no_future_evidence':True})

if __name__=='__main__':run()
