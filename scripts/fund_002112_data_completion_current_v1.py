"""55 份当前公告的隔离全文重提取与本地 OCR；不覆盖另一个会话的展示数据。"""
import json
import re
import sys
from pathlib import Path

from scripts.fund_002112_data_completion_v1 import OUT, ROOT, MANIFEST, read, save, sha, now
sys.path.insert(0, str(OUT/'ocr-deps'))
from opencc import OpenCC
from rapidocr_onnxruntime import RapidOCR
import pypdfium2 as pdfium
from app.services.fund_exposure_common import digest
from app.services.fund_materials_store import source_path
from app.services.fund_earnings_batch_v4 import extract_pdf

def run():
    cc=OpenCC('t2s')
    ocr=None
    for spec in read(MANIFEST)['current_announcements_pending']:
        key=spec['id'];dest=OUT/'current'/f'{key}.json'
        if dest.exists():
            continue
        source=source_path(ROOT,'supplement/company-documents/'+digest(key)+'.json')
        doc=read(source); receipt=doc['receipt'];raw=source_path(ROOT,receipt['file'])
        assert sha(raw)==receipt['sha256']
        pages,metadata=extract_pdf(raw.read_bytes(),1200)
        records=[]
        with pdfium.PdfDocument(str(raw)) as pdf:
            for i,txt in enumerate(pages):
                # 空白签章页也保留 OCR 结果，避免旧“部分文本”状态永久挡住整份文书。
                if len(re.sub(r'\s+','',txt))<60:
                    target=OUT/'ocr-pages'/key/f'{i+1}.json'
                    if target.exists():
                        rec=read(target)
                    else:
                        if ocr is None:
                            ocr=RapidOCR()
                        page=pdf[i];bmp=page.render(scale=1.7);im=bmp.to_pil()
                        target.parent.mkdir(parents=True,exist_ok=True)
                        image=target.with_suffix('.png');im.save(image)
                        result,elapsed=ocr(str(image))
                        boxes=[{'box':r[0],'text':r[1],'confidence':float(r[2])} for r in result or []]
                        rec={'page':i+1,'image':str(image),'image_sha256':sha(image),'boxes':boxes,
                            'text':'\n'.join(r['text'] for r in boxes),'elapsed':elapsed,'visually_verified':False}
                        save(target,rec);im.close();bmp.close();page.close()
                    pages[i]=rec['text'];records.append(str(target))
                    print(json.dumps({'ocr_id':key,'page':i+1,'pages':len(pages),'characters':len(pages[i])}),flush=True)
            # 每份保存封面供目视回核。渲染属于新证据，不改原 PDF。
            image=OUT/'current-covers'/f'{key}.png';image.parent.mkdir(exist_ok=True)
            if not image.exists():
                page=pdf[0];bmp=page.render(scale=1.4);im=bmp.to_pil();im.save(image);im.close();bmp.close();page.close()
        save(dest,{'id':key,'at':now(),'source_record':str(source),'source_sha256':sha(source),'receipt':receipt,
            'raw_path':str(raw),'pages':pages,'metadata':metadata,'ocr_pages':records,
            'normalized_pages':[cc.convert(p) for p in pages], 'title':doc['title'],'stock_code':doc['stock_code'],
            'published_at':doc['announced_at_source'],'old_reason':spec['reason'],
            'body_extracted':all(len(re.sub(r'\s+','',p))>0 for p in pages),
            'identity_verified':False,'semantic_verified':False,'historical_training_eligible':False})
        print(json.dumps({'current':key,'pages':len(pages),'ocr_pages':len(records)},ensure_ascii=False),flush=True)

if __name__=='__main__':
    run()
