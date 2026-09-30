"""修复已定位的异常字体页及缺文字封面；逐页保留原图、OCR 坐标和旧文本。"""
import re,sys
from pathlib import Path
from scripts.fund_002112_data_completion_v1 import OUT,ROOT,read,save,sha,now
sys.path.insert(0,str(OUT/'ocr-deps'))
from scripts.fund_002112_data_completion_ocr_runtime import engine
import pypdfium2 as pdfium

def run():
 cases={r['id']:{x['page'] for x in r['bad_pages']} for r in read(OUT/'historical-font-corruption-audit.json')['rows']}
 cases.setdefault('1209711429',set()).add(1);cases.setdefault('1214258263',set()).add(1)
 ocr=engine();results=[]
 for id,numbers in cases.items():
  dest=OUT/'historical-ocr'/f'{id}.json'
  if dest.exists():results.append({'id':id,'reused_existing_ocr':str(dest),'sha256':sha(dest)});continue
  record=read(OUT/'historical-semantic-closure'/f'{id}.json');d=read(record['source']);r=d['receipt'];raw=Path(r['path']) if r.get('path') else ROOT/r['file']
  assert sha(raw)==r['sha256'];pages=list(d['pages']);evidence=[]
  with pdfium.PdfDocument(str(raw)) as pdf:
   for number in sorted(numbers):
    ep=OUT/'historical-ocr-pages'/id/f'{number}.json'
    if ep.exists():ev=read(ep)
    else:
     ep.parent.mkdir(parents=True,exist_ok=True);png=ep.with_suffix('.png');pg=pdf[number-1];bmp=pg.render(scale=1.8);im=bmp.to_pil();im.save(png);im.close();bmp.close();pg.close()
     boxes,elapsed=ocr(str(png));ev={'page':number,'image':str(png),'image_sha256':sha(png),'text':'\n'.join(x[1] for x in boxes or []),
      'boxes':boxes or [],'elapsed':elapsed,'original_extracted_text':pages[number-1],'visually_verified':False}
     save(ep,ev)
    pages[number-1]=ev['text'];evidence.append(str(ep))
  rec={'id':id,'at':now(),'source':record['source'],'source_sha256':sha(record['source']),'raw_sha256':r['sha256'],
    'pages':pages,'ocr_pages':evidence,'mode':'MIXED_ORIGINAL_TEXT_AND_EXACTLY_LOCATED_PAGE_REPAIR','original_bytes_preserved':True,'numeric_semantics_not_automatically_verified':True}
  save(dest,rec);results.append({'id':id,'repaired_pages':sorted(numbers),'evidence':str(dest)});print(id,sorted(numbers),flush=True)
 save(OUT/'historical-page-repair-result.json',{'at':now(),'rows':results,'new_network_requests':0})

if __name__=='__main__':run()
