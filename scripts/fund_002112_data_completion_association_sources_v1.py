"""按已识别的历史关联缺口补首发文件，目录一次查询、原件每股至多一份。

股票及截止日来自已冻结持仓关联，新增消费单独追加保留，不续领或重置旧
3523 次采集额度及模型额度。元数据中的当前机构编号仅用于路由公开查询。
"""
import hashlib
import json
import re
import sys
import time
from datetime import date,timedelta,datetime,timezone
from pathlib import Path
import httpx
from scripts.fund_002112_data_completion_v1 import OUT,ROOT,read,save,sha,now,fetch
from app.services.fund_materials_store import source_path
from app.services.fund_earnings_batch_v4 import extract_pdf
from app.services.fund_information_history_v1 import pdf_revision

def run():
    listing=len(sys.argv)>1 and sys.argv[1]=='listing'
    correction=len(sys.argv)>1 and sys.argv[1]=='correction'
    supplemental=len(sys.argv)>1 and sys.argv[1]=='supplemental'
    if supplemental: listing=True
    if correction: listing=True
    scope=OUT/'association-source-scope.json'
    if not scope.exists():
        relation=read(OUT/'historical-industry-relations-v5.json');stockmap=read(source_path(ROOT,'supplement/cninfo-stock-map.json'))['rows'];wanted={}
        for row in relation['rows']:
            for h in row['holdings']:
                if not h['gap'] or h.get('issuer_disclosed_classification'):continue
                code=h['stock_code'].split('.')[0]
                if code not in wanted or row['target']<wanted[code]['first_required_target']:
                    wanted[code]={'code':code,'name_at_holding':h['holding_name'],'first_required_target':row['target'],
                        'last_allowed_publication':str(date.fromisoformat(row['target'])-timedelta(days=1)),
                        'public_routing_org_id':stockmap.get(code,{}).get('orgId')}
        save(scope,{'at':now(),'stocks':list(wanted.values()),'scope_evidence':str(OUT/'historical-industry-relations-v5.json'),
            'scope_sha256':sha(OUT/'historical-industry-relations-v5.json'),'one_catalog_query_per_stock':True,'one_original_max_per_stock':True,
            'old_request_consumption_preserved':3523,'model_budget_reallocated':False,'fits':0,
            'reason':'Historical IPO classification is missing from latest previously published regulator rosters and existing issuer reports'})
    results=[]
    for s in read(scope)['stocks']:
        if listing:
            previous=read(OUT/'association-sources'/(s['code']+'.json'))
            if correction:
                oldpath=OUT/'association-listing-sources'/(s['code']+'.json')
                prior=read(oldpath) if oldpath.exists() else previous
                if not re.search('提示性公告|更正公告|确认意见',prior.get('row',{}).get('title_plain','')):continue
            elif supplemental:
                audit=read(OUT/'association-audit-v2'/(s['code']+'.json'))
                if not audit['gaps']:continue
                if (OUT/'association-listing-sources'/(s['code']+'.json')).exists() or (OUT/'association-corrected-sources'/(s['code']+'.json')).exists():continue
            elif 'NO_FINAL_PROSPECTUS_BEFORE_REQUIRED_DATE' not in previous['gaps']:continue
        folder=OUT/('association-supplemental-sources' if supplemental else 'association-corrected-sources' if correction else 'association-listing-sources' if listing else 'association-sources');dest=folder/(s['code']+'.json')
        if dest.exists():results.append(read(dest));continue
        rec={'stock':s,'at':now(),'training_eligible':False,'gaps':[]}
        if not s['public_routing_org_id']:
            rec['gaps'].append('PUBLIC_ROUTING_ORG_ID_UNAVAILABLE');save(dest,rec);results.append(rec);continue
        params={'stock':s['code']+','+s['public_routing_org_id'],'tabName':'fulltext','pageSize':30,'pageNum':1,
            'column':'szse','category':'','plate':'','seDate':'2016-01-01~'+s['last_allowed_publication'],
            'searchkey':'上市公告书' if listing else '招股说明书','secid':'','sortName':'time','sortType':'desc','isHLtitle':'true','trade':''}
        request=folder/'requests'/(s['code']+'.json');receipt=folder/'receipts'/(s['code']+'.json')
        cached_query=OUT/'association-listing-sources'/'receipts'/(s['code']+'.json')
        if correction and cached_query.exists():cr=read(cached_query)
        elif receipt.exists():cr=read(receipt)
        elif request.exists():cr={'ok':False,'reason':'INTERRUPTED_QUERY_PRESERVED'}
        else:
            save(request,{'at':now(),'url':'https://www.cninfo.com.cn/new/hisAnnouncement/query','params':params})
            time.sleep(.4);cr={'ok':False,'request':str(request)}
            try:
                with httpx.Client(timeout=httpx.Timeout(45,connect=10),trust_env=False) as client:
                    response=client.post('https://www.cninfo.com.cn/new/hisAnnouncement/query',data=params)
                    cr['status']=response.status_code;response.raise_for_status()
                    raw=folder/'raw'/(s['code']+'.json');raw.parent.mkdir(parents=True,exist_ok=True)
                    with raw.open('xb') as f:f.write(response.content)
                    data=response.json();assert 'announcements' in data
                    cr.update(ok=True,path=str(raw),sha256=sha(raw),bytes=len(response.content))
            except Exception as e:cr['reason']=type(e).__name__+':'+str(e)[:180]
            save(receipt,cr)
        rec['catalog_receipt']=cr
        candidates=[]
        if cr.get('ok'):
            data=read(cr['path']);rec['catalog_has_more']=bool(data.get('hasMore'))
            for r in data.get('announcements',[]) or []:
                title=re.sub('<[^>]+>','',r.get('announcementTitle',''))
                pub=datetime.fromtimestamp(r['announcementTime']/1000,timezone(timedelta(hours=8))).date().isoformat()
                matching_title=('上市公告书' in title and '首次公开发行' in title) if listing else '招股说明书' in title
                excluded=['申报稿','注册稿','审核稿','摘要','审阅','审计','回复','问询','上市保荐','提示性公告','更正公告','确认意见']
                if r.get('secCode')==s['code'] and pub<=s['last_allowed_publication'] and matching_title and not any(t in title for t in excluded):
                    candidates.append({**r,'title_plain':title,'published_date':pub})
        if candidates:
            row=max(candidates,key=lambda r:r['announcementTime']);url='https://static.cninfo.com.cn/'+row['adjunctUrl']
            ar=fetch(url,'alternatives' if correction or supplemental else 'association_original');rec.update(row=row,receipt=ar)
            if ar.get('ok'):
                try:
                    pages,metadata=extract_pdf(Path(ar['path']).read_bytes(),1200);hits=[]
                    for n,text in enumerate(pages,1):
                        t=re.sub(r'\s+','',text)
                        for m in re.finditer(r'(?:公司|发行人)(?:所处|所属|所属于|属于|所在|的行业分类)[^。；]{0,180}',t):
                            q=t[max(0,m.start()-150):m.end()+50];codes=re.findall(r'(?<![A-Z])([A-S]\d{2})(?!\d)',m[0])
                            if len(set(codes))==1 and re.search('证监会|上市公司行业分类',q) and not re.search('子公司|客户|供应商',t[max(0,m.start()-20):m.start()]):hits.append({'page':n,'quote':q,'code':codes[0],'literal':m[0]})
                    rec.update(pages=pages,metadata=metadata,revision_issues=pdf_revision(metadata,row['published_date']),industry_anchors=hits,
                        consistent_industry_code=hits[0]['code'] if len({h['code'] for h in hits})==1 else None,
                        available_at=str(date.fromisoformat(row['published_date'])+timedelta(days=1))+'T08:00:00+08:00')
                    if not hits:rec['gaps'].append('EXPLICIT_ISSUER_CLASSIFICATION_NOT_FOUND')
                    if rec['revision_issues']:rec['gaps'].append('LATE_PDF_REVISION')
                except Exception as e:rec['gaps'].append('PDF_PARSE_FAILED:'+type(e).__name__)
            else:rec['gaps'].append('ORIGINAL_UNAVAILABLE')
        else:rec['gaps'].append('NO_FINAL_PROSPECTUS_BEFORE_REQUIRED_DATE' if cr.get('ok') else 'PUBLIC_CATALOG_UNAVAILABLE')
        save(dest,rec);results.append(rec)
        print(json.dumps({'association_stock':s['code'],'original':bool(rec.get('pages')),'code':rec.get('consistent_industry_code'),'gaps':rec['gaps']}),flush=True)
    save(OUT/('association-supplemental-source-result.json' if supplemental else 'association-corrected-source-result.json' if correction else 'association-listing-source-result.json' if listing else 'association-source-result.json'),{'at':now(),'stocks':len(results),'originals':sum(bool(r.get('pages')) for r in results),
        'documents_with_consistent_code':sum(bool(r.get('consistent_industry_code')) for r in results),
        'all_requests_accounted':True,'no_training':True})

if __name__=='__main__':run()
