"""原缺件的复杂封面补证：记录原页及页内锚点，不改变原公开日期。"""
import re
import sys
from pathlib import Path
from scripts.fund_002112_data_completion_v1 import OUT,ROOT,MANIFEST,read,save,sha,now
sys.path.insert(0,str(OUT/'ocr-deps'))
from rapidocr_onnxruntime import RapidOCR
from opencc import OpenCC
import pypdfium2 as pdfium

def run():
    cc=OpenCC('t2s');ocr=None
    for m in read(MANIFEST)['historical_missing_originals']:
        id=m['document_id'];dest=OUT/'historical-identity-v2'/f'{id}.json'
        if dest.exists():continue
        d=read(OUT/'historical'/f'{id}.json');old=read(OUT/'historical-audit'/f'{id}.json')
        scan=OUT/'historical-ocr'/f'{id}.json';pages=read(scan)['pages'] if scan.exists() else list(d['pages'])
        images=[]
        if not old.get('identity_verified'):
            with pdfium.PdfDocument(d['receipt']['path']) as pdf:
                for i in range(min(3,len(pages))):
                    # 封面文字可被错误字库映射为无意义字符；无论字数都重新识别封面。
                    p=OUT/'historical-identity-ocr'/id/f'{i+1}.json'
                    if p.exists():record=read(p)
                    else:
                        if ocr is None:ocr=RapidOCR()
                        p.parent.mkdir(parents=True,exist_ok=True);image=p.with_suffix('.png')
                        pg=pdf[i];bmp=pg.render(scale=1.8);im=bmp.to_pil();im.save(image);im.close();bmp.close();pg.close()
                        result,elapsed=ocr(str(image));record={'page':i+1,'image':str(image),'sha256':sha(image),
                            'text':'\n'.join(x[1] for x in result or []),'boxes':result or [],'elapsed':elapsed}
                        save(p,record)
                    # 并存原提取与 OCR，不用 OCR 静默替换整个原件。
                    pages[i]+='\n'+record['text'];images.append(str(p))
        normalized=[re.sub(r'\s+','',cc.convert(p)).lower() for p in pages]
        row=d['row'];stock=row['secCode'];title=row['title_plain'];year=re.search(r'20\d{2}',title)
        terms=['半年度报告','interimreport','semi-annualreport'] if '半年度' in title else ['年度报告','annualreport']
        if '审计报告' in title:terms=['审计报告']
        if '专项说明' in title:terms=['专项说明','监管工作函']
        titlehits=[{'page':i+1,'quote':p[:900]} for i,p in enumerate(normalized[:6]) if (not year or year[0] in p) and any(t in p for t in terms)]
        codehits=[]
        for i,p in enumerate(normalized[:20]):
            for match in re.finditer(r'(?:证券|股票|a股|stock|security|securities)(?:[\u4e00-\u9fff:a-z（）().：]{0,35})'+stock+r'(?!\d)',p):
                codehits.append({'page':i+1,'quote':p[max(0,match.start()-40):match.end()+100],'literal':match[0]})
        passed=bool(old.get('identity_verified') or titlehits and codehits) and not d.get('revision_issues')
        save(dest,{'id':id,'at':now(),'source':str(OUT/'historical'/f'{id}.json'),'source_sha256':sha(OUT/'historical'/f'{id}.json'),
            'raw_sha256':d['receipt']['sha256'],'title':title,'published_date':row['published_date'],
            'prior_verified':old.get('identity_verified'),'code_anchors':codehits,'title_anchors':titlehits,
            'ocr_files':images,'passed':passed,'revision_issues':d.get('revision_issues'),
            'rule':'EXPLICIT_STOCK_LABEL_IN_SAME_DOCUMENT_AND_REPORT_YEAR_TYPE','pages':pages,
            'full_semantics_verified':False,'training_eligible':False})
        print(id,passed,flush=True)

if __name__=='__main__':run()
