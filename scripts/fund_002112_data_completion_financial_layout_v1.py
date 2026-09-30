"""按原 PDF 字符坐标验证财务数字列，防止纯文本跨行错列。

仅接受同一行、从左至右、逐字一致的数字和近邻归母净利润标签。
未命中时保留具体缺口，不把可见金额猜成其他单位或比较期间。
"""
import hashlib
import json
import re
import sys
from pathlib import Path
import pdfplumber

OUT=Path(__file__).resolve().parents[1]/'.local-runs/fund-exposure-002112/data-completion/20260930-v1'
ROOT=OUT.parents[1]
def read(p):
    d=json.loads(Path(p).read_text(encoding='utf-8-sig'))
    return d['payload'] if isinstance(d,dict) and set(d)=={'payload','hash'} else d
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def norm(s):return re.sub(r'\s+','',s).replace('−','-').replace('，',',')

def run():
    version=sys.argv[1] if len(sys.argv)>1 else ''
    suffix='-'+version if version else ''
    totals=[]
    for p in sorted((OUT/('financial-table-facts'+suffix)).glob('*.json')):
        dest=OUT/('financial-layout'+suffix)/p.name
        if dest.exists():totals.append(read(dest));continue
        d=read(p)
        if not d['facts']:continue
        old=OUT/'financial-layout'/p.name
        if version=='v3' and (OUT/'financial-layout-v2'/p.name).exists():old=OUT/'financial-layout-v2'/p.name
        if version=='v4' and (OUT/'financial-layout-v3'/p.name).exists():old=OUT/'financial-layout-v3'/p.name
        if version=='v5' and (OUT/'financial-layout-v4'/p.name).exists():old=OUT/'financial-layout-v4'/p.name
        if suffix and old.exists():
            prior=read(old)
            if not prior.get('error') and [x['raw_row'] for x in prior['checks']]==[x['raw_row'] for x in d['facts']]:
                rec={**prior,'fact_source':str(p),'fact_source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                    'reused_exact_layout_evidence':str(old),'reused_layout_sha256':hashlib.sha256(old.read_bytes()).hexdigest()}
                save(dest,rec);totals.append(rec);continue
        original=read(d['source']);r=original['receipt'];raw=Path(r['path']) if r.get('path') else ROOT/r['file']
        rec={'document_id':d['id'],'fact_source':str(p),'fact_source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'raw_sha256':r['sha256'],'checks':[]}
        try:
            assert hashlib.sha256(raw.read_bytes()).hexdigest()==r['sha256']
            with pdfplumber.open(raw) as pdf:
                for fact in d['facts']:
                    page=pdf.pages[fact['page']-1];words=page.extract_words(x_tolerance=1.5,y_tolerance=2)
                    values=[norm(c['original']) for c in fact['cells']];matches=[]
                    for first in [w for w in words if norm(w['text'])==values[0]]:
                        line=[first]
                        for value in values[1:]:
                            candidates=[w for w in words if norm(w['text'])==value and w['x0']>=line[-1]['x1']-1 and abs(w['top']-first['top'])<4]
                            if not candidates:break
                            line.append(min(candidates,key=lambda w:w['x0']))
                        if len(line)!=len(values):continue
                        labels=[w for w in words if w['x1']<=first['x0']+1 and abs(w['top']-first['top'])<26]
                        label=norm(''.join(w['text'] for w in labels))
                        if '归属' in label and '净利润' in label:
                            matches.append({'cells':line,'label_words':labels,'label_text':label})
                    rec['checks'].append({'page':fact['page'],'raw_row':fact['raw_row'],'source_values':values,
                        'layout_verified':len(matches)==1,'matches':matches,'reason':None if len(matches)==1 else 'EXACT_COORDINATE_LAYOUT_NOT_UNIQUE_OR_NOT_FOUND'})
        except Exception as e:rec['error']=type(e).__name__+':'+str(e)
        rec['all_extracted_rows_layout_verified']=bool(rec['checks']) and all(x['layout_verified'] for x in rec['checks'])
        save(dest,rec);totals.append(rec)
        if len(totals)%100==0:print(json.dumps({'financial_layout_documents':len(totals)}),flush=True)
    save(OUT/('financial-layout-result'+suffix+'.json'),{'reports':len(totals),'verified_reports':sum(d['all_extracted_rows_layout_verified'] for d in totals),
        'verified_rows':sum(c['layout_verified'] for d in totals for c in d['checks']),
        'source_errors':sum(bool(d.get('error')) for d in totals),'entire_report_semantics_not_implied':True})

if __name__=='__main__':run()
