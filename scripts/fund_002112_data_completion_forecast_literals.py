"""补预告文字式事实：仅绑定同句归母指标，旧预告与更正后分开，不反推金额。"""
import re,sys
from collections import Counter
from scripts.fund_002112_data_completion_v1 import OUT,read,save,sha,now

NUM=r'[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
METRIC=re.compile(r'归属于?(?:上市公司|母公司|本公司|本行)(?:股东|所有者)的?净(?:利润|亏损)')

def clauses(pages,title):
 rows=[]
 for pn,raw in enumerate(pages,1):
  page=re.sub(r'\s+','',raw).replace('，',',')
  for m in METRIC.finditer(page):
   start=max(page.rfind('。',0,m.start()),page.rfind('；',0,m.start()))+1
   end=next((i+1 for i in range(m.end(),min(len(page),m.end()+240)) if page[i] in '。；'),min(len(page),m.end()+240))
   q=page[m.start():end]
   # 多指标、表格串行以及第二个归母指标不属于无歧义的同句数值。
   for stop in ['扣除非经常性','扣非','基本每股','每股收益','归属于','净资产','营业收入','利润总额']:
    pos=q.find(stop,len(m[0]))
    if pos>=0:q=q[:pos]
   tail=q[len(m[0]):]
   near=page[max(0,m.start()-500):m.start()]
   sectionmatches=list(re.finditer(r'前次业绩预告情况|更正后的业绩预告情况|上年同期业绩情况|上年同期业绩|本次业绩预告情况|本期业绩预告情况|更正前|更正后|修正前|修正后',near))
   section=sectionmatches[-1][0] if sectionmatches else None
   role='UNRESOLVED_REFERENCE_ROLE'
   if section and any(x in section for x in ['前次','更正前','修正前']):role='PREVIOUS_FORECAST_QUOTED'
   elif section and any(x in section for x in ['更正后','修正后']):role='REVISED_FORECAST'
   elif section and '上年同期' in section:role='PRIOR_PERIOD_COMPARATIVE_IN_CURRENT_DOCUMENT'
   elif section and any(x in section for x in ['本次','本期']):role='CURRENT_DISCLOSURE'
   elif any(x in page[max(start,m.start()-60):m.start()] for x in ['预计','实现']):role='CURRENT_DISCLOSURE' if not any(x in title for x in ['修正','更正']) else role
   money=re.match(r'(?:为|预计为|约为|为约|[：:])?(?:人民币)?(?:盈利|亏损)?[：:]?(?:人民币)?(?:约)?('+NUM+r')(万元|亿元|元)(?:(?:至|到|~|～|—|－|–|-)(?:约)?('+NUM+r')(万元|亿元|元)?)?(左右|以上|以下)?',tail)
   yoy=re.search(r'(?:比上年同期|较上年同期|同比|与上年同期相比)[,，]?(增长|增加|减少|下降|降低|上升)[：:]?(?:约|为)?('+NUM+r')(?:%|％)(?:(?:至|到|~|～|—|－|–|-)(?:约)?('+NUM+r')(?:%|％))?(左右|以上|以下)?',tail)
   if not money and not yoy:continue
   # “亏损”符号仅来自同一金额片段；不能由同比下降推为当期亏损。
   literal=money[0] if money else None
   sign='LOSS_EXPLICIT' if '净亏损' in m[0] or literal and '亏损' in literal else 'PROFIT_EXPLICIT' if literal and '盈利' in literal else None
   rows.append({'page':pn,'offset':m.start(),'quote':q,'metric':'NET_LOSS_ATTRIBUTABLE_TO_ISSUER_SHAREHOLDERS' if '净亏损' in m[0] else 'NET_PROFIT_ATTRIBUTABLE_TO_PARENT','literal_metric_label':m[0],
    'reference_role':role,'section_anchor':section,'section_context':near[-220:],
    'amount':{'first':money[1],'second':money[3],'unit':money[2],'second_unit':money[4] or money[2],
      'currency':'CNY' if '人民币' in money[0] else None,'profit_loss_role':sign,'raw':money[0]} if money else None,
    'yoy':{'direction':yoy[1],'first':yoy[2],'second':yoy[3],'approximation':yoy[4],'raw':yoy[0]} if yoy else None,
    'approximate':bool(re.search(r'约|左右',literal or '')),'only_literal_not_full_accounting_admission':True,
    'not_consensus_surprise':True,'current_amount_never_reconstructed_from_yoy':True})
 return rows

def run():
 suffix='-'+sys.argv[1] if len(sys.argv)>1 else ''
 results=[]
 for p in (OUT/'forecast-table-facts-v2').glob('*.json'):
  old=read(p);f=read(OUT/'historical-facts-v2'/p.name);d=read(f['text_source'])
  pages=d.get('pages',[]);facts=clauses(pages,old['title'])
  rec={'id':old['id'],'at':now(),'title':old['title'],'source':f['source'],'raw_sha256':old['raw_sha256'],
   'text_source':f['text_source'],'text_source_sha256':sha(f['text_source']),
   'facts':facts,'explicit_periods':[x for x in f['field_facts'] if x['kind']=='EXPLICIT_ACCOUNTING_PERIOD'],
   'all_original_quotes_replayed':all(x['quote'] in re.sub(r'\s+','',pages[x['page']-1]).replace('，',',') for x in facts),
   'training_eligible':False,'not_required_amounts_imputed':False}
  assert rec['all_original_quotes_replayed'];save(OUT/('forecast-literal-facts'+suffix)/p.name,rec);results.append(rec)
 save(OUT/('forecast-literal-result'+suffix+'.json'),{'at':now(),'scope':len(results),'documents_with_literal_facts':sum(bool(r['facts']) for r in results),
  'facts':sum(len(r['facts']) for r in results),'reference_roles':dict(Counter(f['reference_role'] for r in results for f in r['facts'])),
  'all_quotes_replayed':all(r['all_original_quotes_replayed'] for r in results),'new_fits':0})

if __name__=='__main__':run()
