"""只从已下载政策图片/PDF 提取证据；OCR 数值必须经原图回核后才能准入。"""
import json
import re
import sys
import time
from pathlib import Path
from scripts.fund_002112_data_completion_v1 import OUT,read,save,sha,now
sys.path.insert(0,str(OUT/'ocr-deps'))
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
import pypdfium2 as pdfium
from app.services.fund_earnings_batch_v4 import extract_pdf

def run():
    from scripts.fund_002112_data_completion_ocr_runtime import engine
    ocr=engine()
    while True:
        for p in [*(OUT/'receipts').glob('*.json'),*(OUT/'asset-receipts').glob('*.json')]:
            r=read(p)
            if not r.get('ok') or r.get('group') not in ('article_images','body_media','attachments'):continue
            raw=Path(r['path']);mime=r.get('headers',{}).get('content-type','')
            with raw.open('rb') as stream:magic=stream.read(8)
            if magic.startswith(b'%PDF'):mime='application/pdf'
            if r['group']=='attachments' and 'pdf' not in mime:continue
            dest=OUT/'policy-assets'/p.name
            if dest.exists():continue
            d={'at':now(),'receipt':r,'semantic_verified':False,'training_eligible':False,'pages':[],'ocr_pages':[]}
            try:
                assert sha(raw)==r['sha256']
                if 'pdf' in mime:
                    d['pages'],d['metadata']=extract_pdf(raw.read_bytes(),1200)
                    with pdfium.PdfDocument(str(raw)) as pdf:
                        for i,txt in enumerate(d['pages']):
                            if len(re.sub(r'\s+','',txt))>=60:continue
                            target=OUT/'policy-ocr-pages'/p.stem/f'{i+1}.json'
                            if target.exists():rec=read(target)
                            else:
                                target.parent.mkdir(parents=True,exist_ok=True);image=target.with_suffix('.png')
                                pg=pdf[i];bmp=pg.render(scale=1.7);im=bmp.to_pil();im.save(image);im.close();bmp.close();pg.close()
                                result,elapsed=ocr(str(image));rec={'page':i+1,'image':str(image),'image_sha256':sha(image),
                                    'text':'\n'.join(x[1] for x in result or []),'boxes':result or [],'elapsed':elapsed,'visually_verified':False}
                                save(target,rec)
                            d['pages'][i]=rec['text'];d['ocr_pages'].append(str(target))
                else:
                    with Image.open(raw) as im:
                        # 长图逐段识别，避免被检测器整体缩小后丢失小字。重叠区域保留位置供去重。
                        d['image_size']=im.size
                        for y in range(0,im.height,1160):
                            image=OUT/'policy-ocr-pages'/p.stem/f'{y}.png';target=image.with_suffix('.json')
                            if target.exists():rec=read(target)
                            else:
                                image.parent.mkdir(parents=True,exist_ok=True)
                                tile=im.crop((0,y,im.width,min(im.height,y+1200)));tile.save(image);tile.close()
                                result,elapsed=ocr(str(image));rec={'offset_y':y,'image':str(image),'image_sha256':sha(image),
                                    'text':'\n'.join(x[1] for x in result or []),'boxes':result or [],'elapsed':elapsed,'visually_verified':False}
                                save(target,rec)
                            d['pages'].append(rec['text']);d['ocr_pages'].append(str(target))
                d['parsed']=True
            except Exception as e:d.update(parsed=False,error=type(e).__name__+':'+str(e)[:400])
            save(dest,d);print(json.dumps({'asset':p.stem,'pages':len(d['pages']),'parsed':d['parsed']}),flush=True)
        if (OUT/'public-recovery-complete.json').exists():break
        time.sleep(5)

if __name__=='__main__':run()
