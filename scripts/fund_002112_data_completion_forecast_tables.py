"""按 PDF 原表格单元格核对业绩预告/快报；盈利区间、亏损、扣非和同比分列。"""
import re,hashlib,json,sys
from pathlib import Path
from collections import Counter
from datetime import datetime
import pdfplumber
from scripts.fund_002112_data_completion_financial_layout_v1 import OUT,ROOT,read,save,norm

NUM=r'[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
def reported_period(title):
 """标题中的第三季度不自动当作前三季度；明确累计字样才返回 YTD9。"""
 year=re.search(r'20\d{2}',title)
 kind='YTD9' if any(x in title for x in ['前三季度','1-9']) else 'Q1' if any(x in title for x in ['一季度','1-3']) else 'H1' if any(x in title for x in ['半年度','半年','上半年']) else 'FY' if '年度' in title and not any(x in title for x in ['三季度','第三季度']) else None
 return f'{year[0]}-{kind}' if year and kind else None
def value(cell):
 q=norm(cell);money=re.findall('('+NUM+r')(万元|亿元|元)',q)
 # 与金额紧邻的盈利/亏损指明符号，原符号保留；不借用同比正负反推。
 labeled=re.findall(r'(盈利|亏损)[：:]?('+NUM+r')(万元|亿元|元)(?:[-—至~～–](?:盈利|亏损)?[：:]?('+NUM+r')(万元|亿元|元)?)?',q)
 percents=re.findall('('+NUM+r')%',q)
 return {'raw':cell,'explicit_money_literals':[{'value':v,'unit':u} for v,u in money],
  'signed_ranges':[{'profit_or_loss':s,'first':a,'unit':u,'second':b or None,'second_unit':v or u if b else None} for s,a,u,b,v in labeled],
  'percentage_literals':percents,'numeric_tokens_are_unassigned_not_signed_semantics':True,
  'currency':'CNY' if '人民币' in q else None,'no_missing_value_imputed':True}

def run():
 version=sys.argv[1] if len(sys.argv)>1 else '';suffix='-'+version if version else '';totals=[]
 for p in (OUT/'historical-semantic-closure').glob('*.json'):
  r=read(p)
  if r['document_kind'] not in ['EARNINGS_FORECAST','PRELIMINARY_EARNINGS']:continue
  dest=OUT/('forecast-table-facts'+suffix)/p.name
  if dest.exists():totals.append(read(dest));continue
  d=read(r['source']);receipt=d['receipt'];raw=Path(receipt['path']) if receipt.get('path') else ROOT/receipt['file'];facts=[];gaps=[]
  originalperiods=[f for f in read(r['literal_facts_source'])['field_facts'] if f['kind']=='EXPLICIT_ACCOUNTING_PERIOD']
  period=reported_period(r['title'])
  try:
   assert hashlib.sha256(raw.read_bytes()).hexdigest()==receipt['sha256']
   with pdfplumber.open(raw) as pdf:
    for page_no,pg in enumerate(pdf.pages,1):
     txt=norm(pg.extract_text() or '')
     if '归属' not in txt or not any(x in txt for x in ['净利润','净亏损']):continue
     for table in pg.find_tables():
      cells=table.extract();compact=[[norm(c or '') for c in row] for row in cells]
      for index,row in enumerate(compact):
       labels=[i for i,c in enumerate(row) if c.startswith('归属')]
       if len(labels)!=1:continue
       li=labels[0];label=row[li]
       endindex=index
       while not any(x in label for x in ['净利润','净亏损']) and endindex+1<min(len(compact),index+4):
        endindex+=1;label+=compact[endindex][li]
       if not re.fullmatch(r'归属于?(?:上市公司|母公司|本公司|本行)(?:股东|所有者)的?净(?:利润|亏损)(?:[（(](?:人民币)?(?:万元|亿元|元)[）)])?',label):continue
       headers=compact[:index];curheads=[];priorheads=[]
       for hi,hrow in enumerate(headers):
        for ci,h in enumerate(hrow):
         rect=table.rows[hi].cells[ci]
         if not rect:continue
         # 只有明确会计期间与正文独立期间完全一致的日期表头，才补充为本期列。
         explicit_dates=re.findall(r'(20\d{2})年(\d{1,2})月(\d{1,2})日',h)
         dates=['%04d-%02d-%02d'%tuple(map(int,x)) for x in explicit_dates]
         titleyear=re.search(r'20\d{2}',r['title'])
         exact_period=bool(titleyear and titleyear[0] in h and '上年' not in h and (len(dates)==2 or re.search(r'20\d{2}年(?:前三季度|第三季度|\d{1,2}[至—－~～–-]\d{1,2}月)',h)))
         exact_year=bool(titleyear and re.fullmatch(titleyear[0]+r'年(?:[（(]未经审计[）)])?',h))
         if (any(x in h for x in ['本报告期','本期','本年度']) or exact_period or exact_year) and not any(x in h for x in ['比','增减']):curheads.append((hi,ci,rect))
         if ('上年同期' in h or titleyear and re.fullmatch(str(int(titleyear[0])-1)+r'年(?:[（(]经审计[）)])?',h)) and not any(x in h for x in ['比','增减']):priorheads.append((hi,ci,rect))
       if len(curheads)!=1 or len(priorheads)!=1:continue
       labelrects=[table.rows[j].cells[li] for j in range(index,endindex+1) if table.rows[j].cells[li]]
       ly0=min(x[1] for x in labelrects);ly1=max(x[3] for x in labelrects)
       def matching_column(head):
        center=(head[2][0]+head[2][2])/2
        matches={}
        for ri,trow in enumerate(table.rows):
         for ci,rect in enumerate(trow.cells):
          cv=compact[ri][ci]
          is_amount=bool(re.search(r'\d',cv)) and (not re.search(r'[%％]|比上年|同比',cv) or bool(re.search(r'\d(?:万元|亿元|元)',cv)))
          if rect and ci!=li and is_amount and rect[0]<=center<=rect[2] and min(ly1,rect[3])-max(ly0,rect[1])>0.5:
           matches[tuple(rect)]=(ri,ci,rect)
        return list(matches.values())
       cur=matching_column(curheads[0]);prior=matching_column(priorheads[0])
       if len(cur)!=1 or len(prior)!=1 or cur[0][2]==prior[0][2]:continue
       cri,cci,crect=cur[0];pri,pci,prect=prior[0]
       unitmatch=re.findall(r'单位[：:]?(?:人民币)?(百万元|千元|万元|亿元|元)',txt[:txt.find(row[0]) if row[0] in txt else len(txt)])
       facts.append({'page':page_no,'reported_period':period,'explicit_period_clauses':originalperiods,'metric':'NET_LOSS_ATTRIBUTABLE_TO_ISSUER_SHAREHOLDERS' if '净亏损' in label else 'NET_PROFIT_ATTRIBUTABLE_TO_PARENT',
        'stage':r['document_kind'],'header_cells':cells[curheads[0][0]],'current_column':cci,'prior_column':pci,
        'current_row':cri,'prior_row':pri,'label_row_range':[index,endindex],
        'current_cell_rectangle':crect,'prior_cell_rectangle':prect,'label_cell_rectangles':labelrects,
        'original_header_cell_rectangles':{'current':curheads[0][2],'prior':priorheads[0][2]},'metric_label':label,
        'current':value(cells[cri][cci] or ''),'prior':value(cells[pri][pci] or ''),
        'row_cells':cells[index],'cell_rectangles':table.rows[index].cells,'table_bbox':table.bbox,
        'explicit_table_units':sorted(set(unitmatch)),'currency_explicit_on_page':'人民币' in txt,
        'same_row_and_column_layout_verified':True,'period_verified':bool(period or originalperiods),
        'accounting_and_revision_chain_complete':False,'forecast_not_actual_realized_earnings':r['document_kind']=='EARNINGS_FORECAST',
        'yoy_not_used_to_reconstruct_amount':True})
  except Exception as e:gaps.append({'reason':'ORIGINAL_TABLE_READ_FAILED','details':type(e).__name__+':'+str(e)[:250]})
  if not facts:gaps.append({'reason':'NO_UNAMBIGUOUS_GRID_WITH_PARENT_PROFIT_AND_CURRENT_PRIOR_HEADERS',
    'textual_facts_retained_at':r['literal_facts_source'],'no_automatic_template_guess':True})
  rec={'id':r['id'],'at':datetime.now().astimezone().isoformat(),'source':r['source'],'raw_sha256':receipt['sha256'],
    'title':r['title'],'document_kind':r['document_kind'],'facts':facts,'gaps':gaps,'training_eligible':False}
  save(dest,rec);totals.append(rec)
  if len(totals)%50==0:print(json.dumps({'processed':len(totals),'with_facts':sum(bool(x['facts']) for x in totals)}),flush=True)
 save(OUT/('forecast-table-result'+suffix+'.json'),{'at':datetime.now().astimezone().isoformat(),'scope':len(totals),'with_grid_facts':sum(bool(x['facts']) for x in totals),
  'facts':sum(len(x['facts']) for x in totals),'gaps':dict(Counter(g['reason'] for r in totals for g in r['gaps'])),'new_fits':0})

if __name__=='__main__':run()
