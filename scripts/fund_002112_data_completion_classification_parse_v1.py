"""按 PDF 表格边框恢复合并单元格，输出带页码及坐标的历史分类。"""
import hashlib
import json
import re
import time
from datetime import date,timedelta
from pathlib import Path
import pdfplumber

OUT=Path(__file__).resolve().parents[1]/'.local-runs/fund-exposure-002112/data-completion/20260930-v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def clean(s):return re.sub(r'\s+','',s or '')

def run():
    folder=OUT/'industry-classification'
    while True:
        for p in (folder/'documents').glob('*.json'):
            dest=folder/'parsed-v2'/p.name
            if dest.exists():continue
            d=read(p);records=[];issues=[]
            for att in d['attachments']:
                receipt=att['receipt']
                if not att.get('pages'):issues.append('ORIGINAL_UNPARSED');continue
                raw=Path(receipt['path']);assert hashlib.sha256(raw.read_bytes()).hexdigest()==receipt['sha256']
                section=category=name=None;source_cells={}
                with pdfplumber.open(raw) as pdf:
                    for num,page in enumerate(pdf.pages,1):
                        tables=page.find_tables()
                        if not tables:issues.append(f'PAGE_{num}_NO_TABLE')
                        for table in tables:
                            for index,row in enumerate(table.extract()):
                                if len(row)!=5:
                                    issues.append(f'PAGE_{num}_UNEXPECTED_COLUMNS');continue
                                if clean(row[3]) in ('上市公司股票代码','上市公司代码'):continue
                                rawrow=row;row=[clean(v) for v in row]
                                if row[0]:section=row[0];source_cells['section']={'page':num,'row':index,'text':row[0]}
                                if row[1]:category=row[1];source_cells['category']={'page':num,'row':index,'text':row[1]}
                                if row[2]:name=row[2];source_cells['name']={'page':num,'row':index,'text':row[2]}
                                if not re.fullmatch(r'\d{6}',row[3]):
                                    if any(row):issues.append(f'PAGE_{num}_NONSTOCK_ROW_'+str(index))
                                    continue
                                group=re.search(r'[（(]([A-S])[）)]',section or '')
                                valid=bool(group and re.fullmatch(r'\d{2}',category or '') and name and row[4])
                                records.append({'stock_code':row[3],'stock_name':row[4],'section':section,'category_code':category,
                                    'industry_code':group[1]+category if valid else None,'industry_name':name,
                                    'page':num,'row':index,'stock_cell_bbox':table.rows[index].cells[3],'original_cells':rawrow,
                                    'merged_cell_sources':dict(source_cells),'structure_verified':valid,
                                    'raw_sha256':receipt['sha256']})
            counts={}
            for r in records:counts[r['stock_code']]=counts.get(r['stock_code'],0)+1
            duplicates=[c for c,n in counts.items() if n>1]
            save(dest,{'source':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'title':d['title'],
                'published_date':d.get('publication_day'),'available_at':str(date.fromisoformat(d['publication_day'])+timedelta(days=1))+'T08:00:00+08:00' if d.get('publication_day') else None,
                'records':records,'issues':issues,'duplicate_stock_codes':duplicates,
                'complete_table_verified':bool(records and not issues and not duplicates and all(r['structure_verified'] for r in records)),
                'publication_reconstructed_not_first_seen':True,'classification_system':'CSRC_2012','index_membership_equivalence':False})
            print(json.dumps({'classification':d['title'],'rows':len(records),'issues':len(issues),'duplicates':len(duplicates)},ensure_ascii=False),flush=True)
        if (folder/'collection-result.json').exists():break
        time.sleep(5)

if __name__=='__main__':run()
