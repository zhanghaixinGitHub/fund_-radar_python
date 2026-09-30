"""独立从 PDF 字符坐标回读预告表格；表内期间优先于整篇标题，避免三季度串期。"""
import re,hashlib,json,sys
import calendar
from pathlib import Path
from datetime import datetime,date
from collections import Counter
import pdfplumber
from scripts.fund_002112_data_completion_financial_layout_v1 import OUT,ROOT,read,save,norm

def hashfile(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def period_from_header(header):
 text=norm(''.join(c or '' for c in header))
 pat=r'(20\d{2})年(\d{1,2})月(\d{1,2})日[至—－~～–-](?:(20\d{2})年)?(\d{1,2})月(\d{1,2})日'
 periods=[]
 for m in re.finditer(pat,text):
  try:lo=date(int(m[1]),int(m[2]),int(m[3]));hi=date(int(m[4] or m[1]),int(m[5]),int(m[6]))
  except ValueError:continue
  if lo<=hi:periods.append({'start':str(lo),'end':str(hi),'quote':m[0],'basis':'EXACT_CURRENT_TABLE_HEADER'})
 if not periods:
  for m in re.finditer(r'(20\d{2})年(\d{1,2})[至—－~～–-](\d{1,2})月',text):
   year,first,last=map(int,m.groups())
   if 1<=first<=last<=12:periods.append({'start':str(date(year,first,1)),'end':str(date(year,last,calendar.monthrange(year,last)[1])),
     'quote':m[0],'basis':'EXPLICIT_CALENDAR_MONTH_RANGE_IN_TABLE_HEADER'})
 unique={(r['start'],r['end']):r for r in periods}
 return next(iter(unique.values())) if len(unique)==1 else None

def run():
 suffix='-'+sys.argv[1] if len(sys.argv)>1 else ''
 totals=[];renders=[]
 for p in sorted((OUT/'forecast-table-facts-v5').glob('*.json')):
  r=read(p)
  for folder in ['forecast-table-facts-v4','forecast-table-facts-v2']:
   prior=OUT/folder/p.name
   if not r['facts'] and read(prior)['facts']:p=prior;r=read(prior)
  dest=OUT/('forecast-final'+suffix)/p.name
  if dest.exists():totals.append(read(dest));continue
  d=read(r['source']);receipt=d['receipt'];raw=Path(receipt['path']) if receipt.get('path') else ROOT/receipt['file'];assert hashfile(raw)==receipt['sha256']
  facts=[];gaps=[]
  with pdfplumber.open(raw) as pdf:
   for f in r['facts']:
    pg=pdf.pages[f['page']-1]
    current=f.get('current_cell_rectangle') or f['cell_rectangles'][f['current_column']]
    previous=f.get('prior_cell_rectangle') or f['cell_rectangles'][f['prior_column']]
    current_text=norm(pg.crop(current).extract_text() or '');prior_text=norm(pg.crop(previous).extract_text() or '')
    checks={'current_characters_match':current_text==norm(f['current']['raw']),
      'prior_characters_match':prior_text==norm(f['prior']['raw']),
      'distinct_cells':tuple(current)!=tuple(previous),'current_header_inside_amount_x':False,'prior_header_inside_amount_x':False}
    for key,rect in [('current',current),('prior',previous)]:
     h=f['original_header_cell_rectangles'][key];center=(h[0]+h[2])/2
     checks[key+'_header_inside_amount_x']=rect[0]<=center<=rect[2]
    period=period_from_header(f['header_cells'])
    if period is None:
     column_header=pg.crop((current[0],f['table_bbox'][1],current[2],current[1])).extract_text() or ''
     period=period_from_header([column_header])
    if period is None:
     context=norm(pg.crop((0,0,pg.width,f['table_bbox'][1])).extract_text() or '')[-260:]
     ranges=list(re.finditer(r'(20\d{2})年(1|7)[至—－~～–-](9)月.{0,15}(?:业绩|经营)',context))
     quarters=list(re.finditer(r'(20\d{2})年(?:度)?(前三季度|第三季度|第一季度|第四季度).{0,15}(?:业绩|经营)',context))
     candidates=[(m.start(),m) for m in ranges+quarters]
     if candidates:
      m=max(candidates,key=lambda x:x[0])[1];year=int(m[1]);quarter={'前三季度':(1,9),'第三季度':(7,9),'第一季度':(1,3),'第四季度':(10,12)}
      first,last=quarter[m[2]] if m[2] in quarter else (int(m[2]),int(m[3]))
      period={'start':str(date(year,first,1)),'end':str(date(year,last,calendar.monthrange(year,last)[1])),
        'quote':m[0],'basis':'NEAREST_EXPLICIT_YEAR_AND_PERIOD_HEADING_BEFORE_TABLE'}
    period_role='TABLE_HEADER' if period else 'SINGLE_TABLE_REPORT_TITLE' if len(r['facts'])==1 and f['reported_period'] else 'PER_TABLE_PERIOD_UNRESOLVED'
    verified=all(checks.values())
    fact={**f,'independent_layout_checks':checks,'independent_layout_verified':verified,
      'accounting_period':period,'effective_period_basis':period_role,
      'reported_title_period_not_applied_to_every_table':True,
      'period_verified':period_role!='PER_TABLE_PERIOD_UNRESOLVED'}
    facts.append(fact)
    if not verified:gaps.append({'reason':'INDEPENDENT_CELL_REPLAY_FAILED','page':f['page'],'checks':checks})
   if r['id'] in ['1209691633','1203087687','1218000634']:
    for n in sorted({f['page'] for f in facts}):
     png=OUT/'forecast-visual-pages'/f"{r['id']}-{n}.png";png.parent.mkdir(exist_ok=True)
     if not png.exists():pdf.pages[n-1].to_image(resolution=130).save(png)
     renders.append({'id':r['id'],'page':n,'path':str(png),'sha256':hashfile(png)})
  literal=OUT/'forecast-literal-facts-v3'/p.name
  rec={'id':r['id'],'at':datetime.now().astimezone().isoformat(),'title':r['title'],'source':r['source'],'raw_sha256':r['raw_sha256'],
   'table_parser_source':str(p),'table_parser_sha256':hashfile(p),'facts':facts,'gaps':gaps,
   'literal_source':str(literal),'literal_sha256':hashfile(literal),'literal_fact_count':len(read(literal)['facts']),
   'no_numeric_fact_not_zero':True,'training_eligible':False,'all_version_and_accounting_gates_kept':True}
  save(dest,rec);totals.append(rec)
 save(OUT/('forecast-acceptance-result'+suffix+'.json'),{'at':datetime.now().astimezone().isoformat(),'scope':len(totals),
  'with_grid_facts':sum(bool(r['facts']) for r in totals),'grid_facts':sum(len(r['facts']) for r in totals),
  'layout_verified_facts':sum(f['independent_layout_verified'] for r in totals for f in r['facts']),
  'period_verified_facts':sum(f['period_verified'] for r in totals for f in r['facts']),
  'with_table_or_literal':sum(bool(r['facts']) or r['literal_fact_count']>0 for r in totals),
  'replay_gaps':dict(Counter(g['reason'] for r in totals for g in r['gaps'])),'renders':renders,'new_fits':0})

if __name__=='__main__':run()
