"""全冻结清单的分字段语义重放，严格保留阶段、范围、缺失和原页。

只有字面上闭合的事实才作为字段通过；整份文书准入与历史版本链另行核验。
公告中的预测、计划、审计、实际回购及客户回购责任不是同一种事实。
"""
import json
import re
import sys
from collections import Counter,defaultdict
from datetime import date,timedelta
from decimal import Decimal
from scripts.fund_002112_data_completion_v1 import OUT,OLD,read,save,sha,now
from scripts.fund_002112_data_completion_semantics_v1 import doc_for

NUM=r'[-−]?(?:\d{1,3}(?:[,，]\d{3})+|\d+)(?:\.\d+)?'
def norm(s):return re.sub(r'\s+','',s)
def evidence(page,number,start,end):
    return {'page':number,'offset':start,'quote':page[start:end]}

def purpose(title,category):
    if category=='BUYBACK':
        if '购车客户' in title or '回购责任' in title:return 'CUSTOMER_FINANCING_REPURCHASE_LIABILITY'
        if '限制性股票' in title or '激励' in title:return 'INCENTIVE_SHARE_CANCELLATION'
        if '债权人' in title:return 'CREDITOR_NOTICE'
        if '实施完毕' in title or '完成' in title:return 'REPURCHASE_COMPLETION_NOTICE'
        if '首次' in title:return 'FIRST_REPURCHASE'
        if '进展' in title or '比例' in title:return 'REPURCHASE_PROGRESS'
        if any(x in title for x in ['预案','方案','回购报告书','提议','提案']):return 'REPURCHASE_PLAN'
        return 'REPURCHASE_RELATED_DOCUMENT'
    if category=='MAJOR_CONTRACT':
        if '中标' in title:return 'CONTRACT_AWARD'
        if '更正' in title:return 'CONTRACT_CORRECTION'
        if '进展' in title:return 'CONTRACT_PROGRESS'
        return 'CONTRACT_DISCLOSURE'
    if any(x in title for x in ['专项说明','问询','监管工作函','核查','法律意见','监事会','独立董事','审议','工作报告','工作规程','制度','董事会','持续督导']):return 'SUPPORTING_OR_GOVERNANCE_DOCUMENT'
    if '更正公告' in title or '修订公告' in title or '补充公告' in title:return 'PERFORMANCE_CORRECTION_NOTICE'
    if '审计报告' in title:return 'AUDIT_REPORT'
    if any(x in title for x in ['预告','预增','预减','预盈','预亏']):return 'EARNINGS_FORECAST'
    if '快报' in title:return 'PRELIMINARY_EARNINGS'
    if '报告' in title:return 'FINANCIAL_REPORT_TRANSLATION' if '英文' in title else 'FINANCIAL_REPORT'
    return 'UNRESOLVED_DOCUMENT_PURPOSE'

def parse_buyback(pages):
    facts=[]
    for n,raw in enumerate(pages,1):
        p=norm(raw)
        for m in re.finditer(r'截至(?:目前|本公告披露日|20\d{2}年\d{1,2}月\d{1,2}日)[^。]{0,1000}。',p):
            q=m[0]
            if '回购' not in q:continue
            if any(x in q for x in ['拟回购','计划回购','回购责任','应回购','回购注销']):continue
            if re.search(r'(?:尚未|未)(?:实施|进行|首次实施|开展|回购)',q):
                facts.append({'kind':'NO_REPURCHASE_YET_EXPLICIT','stage':'NOT_IMPLEMENTED','as_of_original':q[:q.find('日')+1] if '日' in q else q[:6],
                    'share_count':None,'amount':None,'not_zero_imputed':True,'anchor':evidence(p,n,m.start(),m.end()),'field_verified':True});continue
            shares=re.findall(r'(?:累计|已|共)[^。]{0,70}?回购(?:公司)?(?:股份|股票|本公司股份)?(?:数量)?(?:为|共|合计|约|总计)?[：:]?('+NUM+r')(万股|股)',q)
            money=re.findall(r'(?:成交总金额|成交金额|交易总金额|支付的(?:总)?金额|已支付的总金额|使用资金总额|使用资金|支付总金额|支付金额)(?:为|共|合计|约|总计)?(?:人民币)?[：:]?('+NUM+r')(万元|亿元|元)',q)
            day=re.search(r'20\d{2}年\d{1,2}月\d{1,2}日',q[:45])
            if shares or money:
                facts.append({'kind':'ACTUAL_CUMULATIVE_REPURCHASE_DISCLOSURE','stage':'ACTUAL_EXECUTED',
                    'as_of_original':day[0] if day else None,'share_count':[{'value_original':x,'unit':u} for x,u in shares] or None,
                    'amount':[{'value_original':x,'unit':u,'currency':'CNY' if '人民币' in q else None} for x,u in money] or None,
                    'includes_transaction_fees':False if '不含' in q and any(x in q for x in ['费用','佣金']) else None,
                    'anchor':evidence(p,n,m.start(),m.end()),'field_verified':True,
                    'complete_amount_fact':len(shares)==1 and len(money)==1 and day is not None and '人民币' in q})
    return facts

def parse_contract(pages):
    facts=[]
    for n,raw in enumerate(pages,1):
        p=norm(raw)
        for m in re.finditer(r'(?:合同(?:总金额|金额|价款|总价)|总合同金额)[^。；]{0,180}',p):
            q=m[0]
            if any(x in q for x in ['未约定','豁免披露','商业秘密']):
                facts.append({'kind':'CONTRACT_AMOUNT_UNDISCLOSED','amount':None,'reason':'EXPLICIT_NO_AMOUNT_OR_DISCLOSURE_EXEMPTION','anchor':evidence(p,n,m.start(),m.end()),'field_verified':True});continue
            amounts=re.findall('('+NUM+r')(万美元|亿美元|美元|万元|亿元|元)',q)
            if amounts:facts.append({'kind':'CONTRACT_DISCLOSED_AMOUNTS','amounts':[{'value_original':v,'unit':u,'currency':'USD' if '美元' in u else 'CNY' if '人民币' in q else None} for v,u in amounts],
                'amount_scope':'ESTIMATE' if any(x in q for x in ['暂估','估算','预计']) else 'DISCLOSED_CONTRACT_AMOUNT',
                'as_realized_revenue':False,'anchor':evidence(p,n,m.start(),m.end()),'field_verified':True})
        for label in ['合同生效条件','合同生效时间','合同履行期限','合同签订日期','合同签署时间和地点','合同主体','甲方','乙方','发包人','承包人']:
            for m in re.finditer(re.escape(label)+r'[：:]([^。；]{2,230})',p):
                facts.append({'kind':'CONTRACT_EXPLICIT_CLAUSE','field':label,'value_original':m[1],'anchor':evidence(p,n,m.start(),m.end()),'field_verified':True,'does_not_prove_condition_fulfilled':True})
    return facts

def parse_earnings(pages,title):
    facts=[]
    for n,raw in enumerate(pages,1):
        p=norm(raw)
        # 期间必须在正文逐字给出；报告标题年份不自动生成起止日期。
        for m in re.finditer(r'(20\d{2})年(\d{1,2})月(\d{1,2})日[至—－-](?:(20\d{2})年)?(\d{1,2})月(\d{1,2})日',p):
            before=p[max(0,m.start()-18):m.start()]
            if not any(x in before for x in ['报告期','业绩预告期间','会计期间','本期']):continue
            try:lo=date(int(m[1]),int(m[2]),int(m[3]));hi=date(int(m[4] or m[1]),int(m[5]),int(m[6]))
            except ValueError:continue
            if lo<=hi:facts.append({'kind':'EXPLICIT_ACCOUNTING_PERIOD','start':str(lo),'end':str(hi),'anchor':evidence(p,n,max(0,m.start()-18),m.end()),'field_verified':True})
        for m in re.finditer(r'[^。]{0,40}归属于(?:上市公司|母公司)(?:股东|所有者)的?净利润[^。]{0,220}。',p):
            q=m[0]
            if not re.search(r'比上年同期|较上年同期|同比',q):continue
            if not re.search(r'增长|增加|下降|减少',q):continue
            percent=re.findall('('+NUM+r')%',q)
            if percent:
                facts.append({'kind':'PARENT_PROFIT_YOY_LITERAL','metric':'NET_PROFIT_ATTRIBUTABLE_TO_PARENT',
                    'reported_percent_literals':percent,'forecast':any(x in q for x in ['预计','预告']),
                    'approximate':any(x in q for x in ['约','左右']),'not_consensus_surprise':True,
                    'not_derived_current_amount':True,'anchor':evidence(p,n,m.start(),m.end()),'field_verified':True,
                    'comparative_basis_complete':False})
        if n<=20:
            for label in ['企业会计准则','国际财务报告准则','香港财务报告准则','未经审计','审计意见']:
                at=p.find(label)
                if at>=0:facts.append({'kind':'ACCOUNTING_OR_AUDIT_DISCLOSURE','literal':label,'anchor':evidence(p,n,max(0,at-65),min(len(p),at+120)),'field_verified':True,'entire_report_basis_not_inferred_from_mention':True})
    return facts

def run():
    suffix='-'+sys.argv[1] if len(sys.argv)>1 else ''
    rows=read(OLD/'admission-audit-v3/20260929-172820.json')['rows'];out=[]
    for i,r in enumerate(rows,1):
        dest=OUT/('historical-facts'+suffix)/f"{r['document_id']}.json"
        if dest.exists():out.append(read(dest));continue
        d,p=doc_for(r);scan=OUT/'historical-ocr'/f"{r['document_id']}.json"
        if scan.exists():d={**d,'pages':read(scan)['pages']};textsource=str(scan)
        else:textsource=str(p)
        pages=d.get('pages',[]);title=d['row']['title_plain'];kind=purpose(title,r['category'])
        facts=parse_buyback(pages) if r['category']=='BUYBACK' else parse_contract(pages) if r['category']=='MAJOR_CONTRACT' else parse_earnings(pages,title)
        existing=[]
        for ev in r['prior_evidence']:
            if ev.get('field_review_only') and ev.get('review_frozen_document_hash_verified') and ev.get('original_sha256')==d.get('receipt',{}).get('sha256'):
                existing.append(ev)
        a=read(OUT/'historical-audit'/f"{r['document_id']}.json")
        ids=OUT/'historical-identity-v2'/f"{r['document_id']}.json";identity=read(ids)['passed'] if ids.exists() else a.get('identity_verified',False)
        for folder in ['historical-identity-final','historical-identity-additional']:
            ip=OUT/folder/f"{r['document_id']}.json"
            if ip.exists(): identity=identity or read(ip)['identity_verified']
        rec={'id':r['document_id'],'at':now(),'company':r['company'],'category':r['category'],'published_date':r['published_date'],
            'source':str(p),'source_sha256':sha(p),'raw_sha256':d.get('receipt',{}).get('sha256'),'text_source':textsource,
            'title':title,'document_kind':kind,'identity_verified':identity,'body_text_present':sum(len(norm(x)) for x in pages)>100,
            'prior_verified_fields':existing,'field_facts':facts,'field_verified_count':sum(f['field_verified'] for f in facts),
            'old_stops':r['collection_stop_reason_preserved'],'revision_issues':r['revision_issues_preserved'],
            'full_semantics_verified':False,'version_chain_verified':False,'training_eligible':False,
            'unresolved':['EXACT_PERIOD_METRIC_CURRENCY_TABLE_COLUMNS_AND_CORRECTION_CHAIN_REQUIRE_CLOSED_PROOF'] if r['category']=='PERFORMANCE' else
                ['UNIQUE_EVENT_AND_PLAN_ACTUAL_SEQUENCE_NOT_CLOSED','CURRENCY_MISSING_IF_NOT_EXPLICIT'] if r['category']=='BUYBACK' else
                ['MULTIPLE_CONTRACTS_AND_PARTIES_REQUIRE_INDIVIDUAL_BINDING','CONDITIONAL_EFFECTIVENESS_NOT_INFERRED']}
        save(dest,rec);out.append(rec)
        if i%300==0:print(json.dumps({'facts_processed':i}),flush=True)
    save(OUT/('historical-facts-result'+suffix+'.json'),{'at':now(),'scope':len(rows),'processed':len(out),
        'documents_with_field_facts':sum(bool(x['field_facts']) for x in out),'new_field_facts':sum(x['field_verified_count'] for x in out),
        'prior_field_evidence_retained':sum(len(x['prior_verified_fields']) for x in out),
        'document_kinds':dict(Counter(x['document_kind'] for x in out)),'all_unresolved_preserved':True,'new_fits':0})

if __name__=='__main__':run()
