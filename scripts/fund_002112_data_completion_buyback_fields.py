"""回购实际执行的句级字段补核；跨句只连接同一段明确的成交说明。"""
import re
from collections import Counter
from datetime import date
from scripts.fund_002112_data_completion_v1 import OUT,read,save,sha,now
from scripts.fund_002112_data_completion_facts_v1 import NUM,norm

SHARES=re.compile(r'回购(?:了)?(?:公司|本公司)?(?:股份|股票|公司股份)?(?:的股份)?(?:的?数量)?(?:为|共计|共|合计|约|总计)?[：:]?('+NUM+r')(万股|股)')
MONEY=re.compile(r'(?:成交总金额|成交金额|交易总金额|支付的(?:总)?金额|已支付的总金额|使用资金总额|使用资金|支付总金额|支付金额|回购总金额|累计回购金额|支付的资金总额|支付资金总额|耗资)(?:约为|为|共计|共|合计|约|总计)?(?:人民币)?[：:]?('+NUM+r')(万元|亿元|元)')

def run():
 results=[]
 for p in (OUT/'historical-semantic-closure').glob('*.json'):
  old=read(p)
  if old['category']!='BUYBACK':continue
  d=read(old['source']);pages=d.get('pages',[]);facts=[];noexec=[]
  for page,raw in enumerate(pages,1):
   txt=norm(raw)
   for m in re.finditer(r'截至(?:目前|本公告披露日|20\d{2}年\d{1,2}月\d{1,2}日)[^。]{0,1200}。',txt):
    q=m[0]
    if '回购' not in q or any(x in q for x in ['拟回购','计划回购','回购责任','应回购','回购注销']):continue
    anchor={'page':page,'start':m.start(),'quote':q}
    if re.search(r'(?:尚未|未)(?:实施|进行|首次实施|开展|回购)',q):noexec.append(anchor);continue
    joined=False
    if not MONEY.search(q):
     nxt=txt[m.end():m.end()+650].split('。')[0]
     if re.match(r'(?:回购|最高成交|最低成交|成交|已支付|支付|公司累计支付)',nxt) and MONEY.search(nxt) and not re.search(r'截至|拟|计划|提议|方案|20\d{2}年',nxt):q+=nxt+'。';joined=True
    sharelist=[{'original':v,'unit':u} for v,u in SHARES.findall(q)];moneylist=[{'original':v,'unit':u,'currency':'CNY' if '人民币' in q else None} for v,u in MONEY.findall(q)]
    if not sharelist and not moneylist:continue
    dm=re.search(r'(20\d{2})年(\d{1,2})月(\d{1,2})日',q[:60]);asof=None
    if dm:
     try:asof=str(date(*map(int,dm.groups())))
     except ValueError:pass
    elif q.startswith('截至本公告披露日'):asof=old['published_date']
    gaps=[]
    if len(sharelist)!=1:gaps.append('SINGLE_EXECUTED_QUANTITY_NOT_BOUND')
    if len(moneylist)!=1:gaps.append('SINGLE_EXECUTED_AMOUNT_NOT_BOUND')
    elif not moneylist[0]['currency']:gaps.append('CURRENCY_NOT_EXPLICIT_IN_EXECUTION_CLAUSE')
    if not asof or asof>old['published_date']:gaps.append('EXECUTION_ASOF_NOT_CLOSED')
    facts.append({'stage':'ACTUAL_CUMULATIVE_EXECUTION','as_of':asof,'shares':sharelist,'amounts':moneylist,
      'anchor':{**anchor,'quote':q},'same_paragraph_continuation':joined,
      'excludes_transaction_fees':bool(re.search(r'不含[^）)]*(?:交易|佣金|费用)',q)),
      'amount_is_approximate':bool(re.search(r'金额约|总额约|耗资约',q)),'field_gaps':gaps,'literal_fields_complete':not gaps})
  rec={'id':old['id'],'at':now(),'source':old['source'],'source_sha256':sha(old['source']),'raw_sha256':old['raw_sha256'],
    'published_date':old['published_date'],'document_kind':old['document_kind'],'actual_facts':facts,'explicit_not_implemented':noexec,
    'plan_or_incentive_cancellation_not_used_as_market_execution':True,'version_and_unique_event_chain_verified':False,'training_eligible':False}
  save(OUT/'buyback-execution-fields'/p.name,rec);results.append(rec)
 save(OUT/'buyback-execution-result.json',{'at':now(),'scope':len(results),'with_actual_facts':sum(bool(x['actual_facts']) for x in results),
  'complete_literal_fields_documents':sum(any(f['literal_fields_complete'] for f in r['actual_facts']) for r in results),
  'explicit_not_implemented_documents':sum(bool(r['explicit_not_implemented']) for r in results),'field_gaps':dict(Counter(g for r in results for f in r['actual_facts'] for g in f['field_gaps'])),
  'no_full_event_chain_admission_claim':True})

if __name__=='__main__':run()
