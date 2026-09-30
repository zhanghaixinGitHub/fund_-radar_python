"""全冻结历史清单的字段验收与引用关系，不把候选片段提升为完整训练准入。"""
import re,json,sys
from collections import Counter,defaultdict
from datetime import date,timedelta
from scripts.fund_002112_data_completion_v1 import OUT,OLD,read,save,sha,now
from scripts.fund_002112_data_completion_semantics_v1 import doc_for
from scripts.fund_002112_data_completion_facts_v1 import purpose,parse_buyback

def identity(id):
    evidence=[]
    for folder in ['historical-audit','historical-identity-final','historical-identity-additional','historical-identity-completion','historical-missing-audit-v4','historical-identity-resolved-v3']:
        p=OUT/folder/f'{id}.json'
        if p.exists() and read(p).get('identity_verified'):evidence.append(str(p))
    return evidence

def run():
    version=sys.argv[1] if len(sys.argv)>1 else '';suffix='-'+version if version else ''
    rows=read(OLD/'admission-audit-v3/20260929-172820.json')['rows'];result=[];issues=defaultdict(list);references=[];families=defaultdict(list)
    for i,r in enumerate(rows,1):
        id=r['document_id'];dest=OUT/('historical-semantic-closure'+suffix)/f'{id}.json'
        if dest.exists():rec=read(dest)
        else:
            d,p=doc_for(r);old=read(OUT/'historical-facts-v2'/f'{id}.json');scan=OUT/'historical-ocr'/f'{id}.json'
            pages=read(scan)['pages'] if scan.exists() else d.get('pages',[]);text=[re.sub(r'\s+','',t) for t in pages]
            title=d['row']['title_plain'];kind=old['document_kind'];ids=identity(id);gaps=[]
            # 确认目录简写掩盖了回购注销时，以正文标题范围为准；不看后来的目录撤销标记。
            header=''.join(text[:2])[:700].split('本公司')[0]
            if r['category']=='BUYBACK' and ('限制性股票' in header or '回购注销' in header):kind='INCENTIVE_SHARE_CANCELLATION'
            if r['category']=='PERFORMANCE' and any(w in title for w in ['确认意见','书面确认','签字页']):kind='SUPPORTING_CONFIRMATION_NOT_EARNINGS'
            if '偿付能力' in title:kind='SOLVENCY_REPORT_NOT_PARENT_EARNINGS'
            if id in ['1212455504','1216218313']:kind='PRELIMINARY_FINANCIAL_STATEMENTS_NOT_FINAL_AUDITED_RESULT'
            if id=='1214354641':kind='SPONSOR_TRACKING_REPORT_NOT_ISSUER_FINANCIAL_REPORT'
            if id=='1212747167':kind='EARNINGS_PRESS_RELEASE_APPENDIX'
            rec={'id':id,'at':now(),'category':r['category'],'company':r['company'],'title':title,'document_kind':kind,
                'source':str(p),'source_sha256':sha(p),'raw_sha256':d.get('receipt',{}).get('sha256'),'published_date':r['published_date'],
                'declared_available_at':str(date.fromisoformat(r['published_date'])+timedelta(days=1))+'T08:00:00+08:00',
                'identity_verified':bool(ids),'identity_evidence':ids,'old_collection_stop':r['collection_stop_reason_preserved'],
                'old_revision_issues':r['revision_issues_preserved'],'old_field_reviews':old['prior_verified_fields'],
                'literal_facts_source':str(OUT/'historical-facts-v2'/f'{id}.json'),'full_training_admission':False,'gaps':gaps}
            if not ids:gaps.append('SOURCE_IDENTITY_REMAINS_UNPROVEN')
            if r['revision_issues_preserved']:gaps.append('ORIGINAL_LATE_REVISION_EVIDENCE_PRESERVED')
            if r['category']=='BUYBACK':
                facts=parse_buyback(pages);actual=[];notyet=[]
                for f in facts:
                    if f['kind']=='NO_REPURCHASE_YET_EXPLICIT':notyet.append(f);continue
                    q=f['anchor']['quote'];m=re.search(r'(20\d{2})年(\d{1,2})月(\d{1,2})日',q[:60]);asof=None
                    if m:
                        try:asof=str(date(*map(int,m.groups())))
                        except ValueError:pass
                    elif q.startswith('截至本公告披露日'):asof=r['published_date']
                    f['explicit_as_of']=asof
                    f['as_of_not_future']=bool(asof and asof<=r['published_date'])
                    f['mandatory_field_gaps']=[]
                    if not f['share_count']:f['mandatory_field_gaps'].append('EXECUTED_SHARE_QUANTITY_NOT_BOUND')
                    if not f['amount']:f['mandatory_field_gaps'].append('EXECUTED_AMOUNT_NOT_BOUND')
                    elif any(x.get('currency') is None for x in f['amount']):f['mandatory_field_gaps'].append('CURRENCY_NOT_EXPLICIT_IN_AMOUNT_CLAUSE')
                    if not f['as_of_not_future']:f['mandatory_field_gaps'].append('EXECUTION_ASOF_DATE_NOT_CLOSED')
                    actual.append(f)
                rec.update(actual_cumulative_facts=actual,not_implemented_facts=notyet,
                    plan_numbers_never_used_as_execution=True,body_purpose_reclassification=kind!=old['document_kind'])
                bpath=OUT/'buyback-execution-fields'/f'{id}.json'
                if bpath.exists():rec['updated_execution_field_evidence']=str(bpath);rec['updated_execution_field_evidence_sha256']=sha(bpath)
                if kind in ['CUSTOMER_FINANCING_REPURCHASE_LIABILITY','INCENTIVE_SHARE_CANCELLATION','CREDITOR_NOTICE']:
                    rec['fact_scope']='NOT_ORDINARY_MARKET_SHARE_REPURCHASE_PROGRESS'
                elif not actual and not notyet:gaps.append('NO_COMPLETE_EXECUTION_CLAUSE_IN_THIS_DOCUMENT')
                elif actual and not any(not f['mandatory_field_gaps'] for f in actual):gaps.append('EXECUTION_CLAUSE_FIELDS_PARTIALLY_UNRESOLVED')
            elif r['category']=='MAJOR_CONTRACT':
                cp=OUT/'contract-scope-review'/f'{id}.json';c=read(cp)
                rec.update(contract_stage=c['event_stage'],contract_scope_review=str(cp),contract_scope_sha256=sha(cp),stage_and_amount_scope_verified=True)
                rec['unknowns_are_not_zero']=['UNDISCLOSED_COUNTERPARTY_OR_AMOUNT','CONDITION_FULFILMENT_IF_NOT_DECLARED','ISSUER_ALLOCATION_IF_CONSORTIUM','ACTUAL_REVENUE_OR_PAYMENT']
            else:
                fp=OUT/'financial-table-facts-v4'/f'{id}.json'
                f=read(fp) if fp.exists() else {};lp=OUT/'financial-layout-v4'/f'{id}.json';gp=OUT/'financial-grid-v4'/f'{id}.json'
                # 表格位置已由独立 PDF 文本坐标及边框核验；失败记录与修复记录同时保留。
                layout=read(lp) if lp.exists() else {};grid=read(gp) if gp.exists() else {}
                rec.update(financial_facts_source=str(fp) if fp.exists() else None,financial_fact_count=len(f.get('facts',[])),
                    financial_layout_source=str(lp) if lp.exists() else None,financial_grid_source=str(gp) if gp.exists() else None,
                    financial_parser_gaps=f.get('gaps',[]))
                if not f.get('facts') and kind in ['FINANCIAL_REPORT','FINANCIAL_REPORT_TRANSLATION']:gaps.append('PARENT_PROFIT_TABLE_PERIOD_UNIT_OR_COLUMNS_NOT_CLOSED')
                elif kind in ['EARNINGS_FORECAST','PRELIMINARY_EARNINGS']:
                    rec['literal_yoy_clauses']=[x for x in old['field_facts'] if x['kind']=='PARENT_PROFIT_YOY_LITERAL']
                    forecast=OUT/'forecast-table-facts-v2'/f'{id}.json'
                    if forecast.exists():rec['forecast_original_cell_evidence']=str(forecast);rec['forecast_original_cell_count']=len(read(forecast)['facts'])
                    gaps.append('FORECAST_OR_PRELIMINARY_RANGE_AND_COMPARATIVE_BASE_NOT_FULLY_BOUND')
                elif kind not in ['FINANCIAL_REPORT','FINANCIAL_REPORT_TRANSLATION']:
                    rec['fact_scope']='SUPPORTING_OR_CORRECTION_DOCUMENT_NOT_NEW_EARNINGS_RESULT'
                for fact in f.get('facts',[]):
                    if fact.get('currency') is None:gaps.append('TABLE_CURRENCY_NOT_EXPLICIT')
                    for cell in fact['cells']:
                        period=cell['column'].get('period')
                        if period and period.startswith((re.search(r'20\d{2}',title) or [''])[0]):families[(r['company'],period)].append(id)
            own=[];refs=[]
            for n,t in enumerate(text):
                for m in re.finditer(r'公告(?:编号|编码)[：:]?(?:临)?[（(]?(20\d{2})[）)]?[-－—](\d{1,4})',t):
                    number=f'{m[1]}-{int(m[2]):03d}';anchor={'number':number,'page':n+1,'quote':t[max(0,m.start()-65):m.end()+80]}
                    if n==0 and m.start()<350 and not own:own.append(anchor)
                    else:refs.append(anchor)
            rec.update(own_announcement_numbers=own,explicit_announcement_number_references=refs)
            rec['gaps']=list(dict.fromkeys(gaps));save(dest,rec)
        result.append(rec)
        for own in rec['own_announcement_numbers']:issues[(rec['company'],own['number'])].append(rec)
        references.extend((rec,ref) for ref in rec['explicit_announcement_number_references'])
        if i%1000==0:print(json.dumps({'semantic_rows':i}),flush=True)
    edges=[];unmatched=[]
    for source,ref in references:
        candidates=[r for r in issues.get((source['company'],ref['number']),[]) if r['id']!=source['id'] and r['published_date']<=source['published_date']]
        edge={'from_id':source['id'],'from_available_at':source['declared_available_at'],'cited_number':ref['number'],'anchor':ref,'type':'EXPLICIT_CITATION_NOT_AUTOMATIC_REVISION',
              'to_ids':[r['id'] for r in candidates],'no_backfill':True}
        (edges if candidates else unmatched).append(edge)
    save(OUT/('historical-explicit-reference-graph'+suffix+'.json'),{'at':now(),'edges':edges,'unmatched_references':unmatched,'same_period_does_not_imply_revision':True})
    save(OUT/('historical-period-families'+suffix+'.json'),{'at':now(),'groups':[{'company':k[0],'reported_period':k[1],'ids':sorted(set(v))} for k,v in families.items()],
        'summaries_translations_and_full_reports_not_independent_events':True})
    save(OUT/('historical-semantic-closure-result'+suffix+'.json'),{'at':now(),'frozen_scope':len(rows),'processed':len(result),'identity_verified':sum(r['identity_verified'] for r in result),
        'document_kinds':dict(Counter(r['document_kind'] for r in result)),'gaps':dict(Counter(g for r in result for g in r['gaps'])),
        'buyback_actual_clause_documents':sum(bool(r.get('actual_cumulative_facts')) for r in result),
        'buyback_complete_literal_amount_documents':sum(any(not f['mandatory_field_gaps'] for f in r.get('actual_cumulative_facts',[])) for r in result),
        'buyback_explicit_not_implemented_documents':sum(bool(r.get('not_implemented_facts')) for r in result),
        'contract_stage_reviewed':sum(bool(r.get('stage_and_amount_scope_verified')) for r in result),'reference_edges':len(edges),'unmatched_explicit_references':len(unmatched),
        'all_frozen_documents_retained':{r['id'] for r in result}=={r['document_id'] for r in rows},'new_fits':0})

if __name__=='__main__':run()
