"""核对目录简写、英文名称及非上市公司本体文书；矛盾标题绝不自动纠正。"""
import re,sys
from collections import Counter
from scripts.fund_002112_data_completion_v1 import OUT,read,save,sha,now
from scripts.fund_002112_data_completion_identity_completion import normalized
from opencc import OpenCC

# 下列名称均须由同代码、发布不晚于目标报告的中文原件再次逐字证明。
ENGLISH={'002415':'Hangzhou Hikvision Digital Technology Co., Ltd.', '000333':'Midea Group Co., Ltd.',
 '002611':'Guangdong Dongfang Precision Science & Technology Co., Ltd.', '000568':'Luzhou Laojiao Co., Ltd.',
 '603713':'Milkyway Chemical Supply Chain Service Co., Ltd.'}
# 只规范不改变事实的标题表达；已知年度/文书性质冲突单独保留。
CONFLICTS={'1216517762':'目录2021年度，正文2022年度，不能替换年度',
 '1216439874':'目录书面确认意见，原件独立董事述职报告',
 '1217757139':'目录注销2020年回购股份，正文标题为限制性股票，需保留正文内部差异'}

def run():
 version=sys.argv[1] if len(sys.argv)>1 else '';suffix='-'+version if version else ''
 cc=OpenCC('t2s');n=lambda s:re.sub(r'[.,&（）()\-]','',normalized(s,cc));profiles=read(OUT/'historical-issuer-name-profiles.json')
 allrows=[read(p) for p in (OUT/'historical-semantic-closure').glob('*.json')];enproof={};results=[]
 for row in sorted(allrows,key=lambda r:r['published_date']):
  code=row['company']
  if code not in ENGLISH or code in enproof or not row['identity_verified']:continue
  d=read(row['source']);needle=n(ENGLISH[code])
  for page,t in enumerate(d.get('pages',[])[:25],1):
   if needle in n(t):
    at=n(t).index(needle);enproof[code]={'name':ENGLISH[code],'id':row['id'],'source':row['source'],'sha256':sha(row['source']),
      'raw_sha256':row['raw_sha256'],'published_date':row['published_date'],'page':page,'quote':n(t)[max(0,at-100):at+len(needle)+100]};break
 for old in allrows:
  if old['identity_verified']:continue
  d=read(old['source']);p=OUT/'historical-ocr'/f"{old['id']}.json";pages=read(p)['pages'] if p.exists() else d.get('pages',[])
  ts=[n(t) for t in pages];title=n(old['title']);code=old['company'];entity=[];titleproof=[];role='ISSUER_DOCUMENT';gaps=[]
  for page,t in enumerate(ts,1):
   m=re.search(r'(?:证券|股票|公司|stockexchange|stock|securit(?:y|ies))代码?[:：a-z\u4e00-\u9fff(),.]{0,60}'+code+r'(?!\d)',t)
   if not m:m=re.search(r'(?:stockexchange|stock|security|securities)?code[:：]{0,2}'+code,t)
   if not m:m=re.search(r'(?:证券简称及代码|a股代码)[:：]?[a-z\u4e00-\u9fff]{0,35}'+code,t)
   if not m:
    # 多个交易所股票代码在同一行时，不能在去空白后的文本上检查数字边界。
    m=re.search(r'(?:证券|股票|公司)\s*代码\s*[:：]?\s*'+code+r'(?!\d)',pages[page-1])
   if m:entity.append({'type':'EXPLICIT_SECURITY_CODE','page':page,'quote':m[0]});break
  for pr in profiles.get(code,[]):
   if pr['published_date']>old['published_date']:continue
   for name in pr['names']:
    hit=next((i for i,t in enumerate(ts[:25],1) if n(name['name']) in t),None)
    if hit:entity.append({'type':'EARLIER_LEGAL_NAME','page':hit,'name':name['name'],'profile':pr});break
   if entity:break
  ep=enproof.get(code)
  if ep and ep['published_date']<=old['published_date'] and any(n(ep['name']) in t for t in ts[:4]):entity.append({'type':'EARLIER_BILINGUAL_NAME','proof':ep})
  # 保险子公司偿付能力报告必须保留实际主体，不能作为上市母公司的净利润报告。
  if '偿付能力' in title:
   role='SOLVENCY_REPORT_OF_NAMED_ENTITY_NOT_PARENT_EARNINGS'
   names=re.findall(r'(?:中国太保:)?([\u4e00-\u9fff]{3,30}(?:股份有限公司))',old['title'])
   for name in names:
    if n(name) in ''.join(ts[:2]):entity.append({'type':'EXACT_NAMED_SUBJECT_IN_CATALOG_AND_BODY','name':name,'parent_ownership_not_inferred':True})
   if '偿付能力季度报告摘要' in ''.join(ts[:2]):titleproof.append({'page':1,'quote':pages[0][:700],'kind':role})
  # 年报/季报的标题及期间可以分行，必须在同一页同时出现；不得用发表年替代报告年。
  clean=title.replace('第1季度','第一季度').replace('第3季度','第三季度')
  year=re.search(r'20\d{2}',clean)
  terms=['半年度报告','interimreport'] if '半年度' in clean else ['第一季度报告','firstquarter','reportofq1','1stquarterreport','q1'+(year[0] if year else '')+'report'] if '一季度' in clean else ['第三季度报告','thirdquarter','reportofq3','3rdquarterreport','q3'+(year[0] if year else '')+'report'] if '三季度' in clean else ['年度报告','年报','annualreport']
  if '报告' in clean and not any(s in clean for s in ['审计','确认','制度','跟踪','偿付能力','回购']):
   for i,t in enumerate(ts[:25],1):
    if year and year[0] in t and any(s in t for s in terms) and ('摘要' not in clean or '摘要' in t or 'summary' in t):
     titleproof.append({'page':i,'quote':pages[i-1][:900],'year':year[0]});break
  # 简写只在目录与正文有相同文书关键字、主体已绑定时接受，正文原题完整保存。
  head=''.join(ts[:1])[:1400]
  header=head.split('本公司')[0].split('公司及董事会')[0]
  pairs=[('回购','进展'),('回购','首次'),('回购','完成'),('回购','实施完成'),('回购','方案'),('回购','报告书'),('回购','内部控制制度'),('回购','价格'),
    ('回购','提议'),('回购','实施结果'),('年度报告','重大差错责任追究制度'),('业绩','预告'),('业绩','快报'),
    ('年度报告','书面确认意见'),('半年度报告','书面确认意见'),('审计','报告'),('半年度报告','跟踪报告')]
  for a,b in pairs:
   if a in title and b in title and a in header and b in header and (not year or year[0] in head or b=='重大差错责任追究制度'):
    titleproof.append({'page':1,'quote':pages[0][:1400],'matched_document_terms':[a,b]});break
  if old['id']=='1216166027' and '2022年年度报告' in head and '603306' in head:
   titleproof.append({'page':1,'quote':pages[0],'reason':'CATALOG_YEAR_TRUNCATED_202_BODY_EXPLICIT_2022','catalog_error_preserved':True})
  if old['id']=='1216624182' and '年度报告' in head and '2022' in head:titleproof.append({'page':1,'quote':pages[0],'year':'2022'})
  if old['id']=='1214938782' and '2022年第三季度报告' in head:titleproof.append({'page':1,'quote':pages[0],'year':'2022','year_omitted_from_catalog':True})
  if old['id']=='1214354641' and '2022年半年度跟踪报告' in head:
   titleproof.append({'page':1,'quote':pages[0]});role='SPONSOR_TRACKING_REPORT_NOT_ISSUER_FINANCIAL_REPORT'
  if old['id']=='1212747167' and '安道麦发布2021年第四季度及全年业绩' in head:
   titleproof.append({'page':1,'quote':pages[0]});role='EARNINGS_PRESS_RELEASE_APPENDIX_NOT_STATUTORY_ANNUAL_REPORT'
  if old['id']=='1216234519' and '审计报告' in head and any('2022年12月31日' in t for t in ts[:8]):
   titleproof.append({'page':1,'quote':pages[0],'audited_period_evidence':[{'page':i+1,'quote':t[:1600]} for i,t in enumerate(ts[:8]) if '2022年12月31日' in t]});role='AUDIT_REPORT_OF_NAMED_ISSUER'
  if old['id']=='1218296567' and '表格10q' in head and '百济神州有限公司' in head:
   entity.append({'type':'EXACT_NAMED_SUBJECT','name':'百济神州有限公司','US_filing_not_CN_accounting':True});titleproof.append({'page':1,'quote':pages[0]});role='US_FORM_10Q_OF_ISSUER'
  if old['id']=='1209772733' and '中国平安保险集团股份有限公司' in head:
   ep=OUT/'historical-identity-resolved-v2/1209239168.json';prior=read(ep);pd=read(prior['source'])
   if prior['identity_verified'] and any('中国平安保险集团股份有限公司' in n(t) for t in pd['pages']):
    entity.append({'type':'EXACT_EARLIER_CODE_BOUND_LEGAL_NAME','name':'中国平安保险（集团）股份有限公司','source':str(ep),'sha256':sha(ep)})
  if old['id'] in CONFLICTS:gaps.append(CONFLICTS[old['id']])
  if not entity:gaps.append('NO_EXPLICIT_BODY_ENTITY_BINDING')
  if not titleproof:gaps.append('BODY_TITLE_PERIOD_OR_PURPOSE_STILL_UNPROVEN')
  rec={'id':old['id'],'at':now(),'identity_verified':bool(entity and titleproof and not gaps),'source':old['source'],
    'source_sha256':sha(old['source']),'raw_sha256':old['raw_sha256'],'catalog_title':old['title'],'body_first_page':pages[0] if pages else None,
    'entity_evidence':entity,'title_evidence':titleproof,'document_role':role,'gaps':gaps,'training_eligible':False}
  save(OUT/('historical-identity-resolved'+suffix)/f"{old['id']}.json",rec);results.append(rec)
 save(OUT/('historical-identity-resolved-result'+suffix+'.json'),{'at':now(),'scope':len(results),'newly_verified':sum(r['identity_verified'] for r in results),
  'unresolved':[{'id':r['id'],'title':r['catalog_title'],'gaps':r['gaps']} for r in results if not r['identity_verified']],
  'bilingual_source_evidence':enproof,'all_original_titles_and_conflicts_retained':True})

if __name__=='__main__':run()
