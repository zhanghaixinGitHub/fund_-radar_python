"""对跨行标签及数字含字间距的财务表，按原 PDF 表格边框复核同一行。"""
import hashlib
import json
import re
import sys
from pathlib import Path
import pdfplumber
from scripts.fund_002112_data_completion_financial_layout_v1 import OUT,ROOT,read,save,norm

def run():
    version=sys.argv[1] if len(sys.argv)>1 else 'v3'
    suffix='' if version=='v3' else '-'+version
    results=[]
    for p in (OUT/('financial-layout-'+version)).glob('*.json'):
        old=read(p)
        if old['all_extracted_rows_layout_verified']:continue
        d=read(old['fact_source']);source=read(d['source']);r=source['receipt'];raw=Path(r['path']) if r.get('path') else ROOT/r['file']
        assert hashlib.sha256(raw.read_bytes()).hexdigest()==r['sha256'];checks=[]
        with pdfplumber.open(raw) as pdf:
            for check in old['checks']:
                if check['layout_verified']:checks.append(check);continue
                matches=[];pg=pdf.pages[check['page']-1]
                for table in pg.find_tables():
                    for index,row in enumerate(table.extract()):
                        cells=[norm(c or '') for c in row]
                        if len(cells)!=len(check['source_values'])+1:continue
                        if not re.fullmatch(r'归属于(?:上市公司|母公司)(?:股东|所有者)的?净利润',cells[0]):continue
                        if cells[1:]!=[norm(c) for c in check['source_values']]:continue
                        matches.append({'row':index,'original_cells':row,'cell_rectangles':table.rows[index].cells,'table_bbox':table.bbox})
                checks.append({**check,'layout_verified':len(matches)==1,'grid_matches':matches,'grid_based_exact_cell_proof':True})
        result={**old,'previous_layout_file':str(p),'checks':checks,'all_extracted_rows_layout_verified':all(c['layout_verified'] for c in checks)}
        save(OUT/('financial-grid'+suffix)/p.name,result);results.append(result)
    save(OUT/('financial-grid-result'+suffix+'.json'),{'rechecked':len(results),'verified':sum(r['all_extracted_rows_layout_verified'] for r in results),
        'unresolved':[r['document_id'] for r in results if not r['all_extracted_rows_layout_verified']]})

if __name__=='__main__':run()
