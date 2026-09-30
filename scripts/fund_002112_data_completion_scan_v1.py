"""本地读取历史扫描件全部页面；每页可恢复，识别文字不自动升级成事实通过。"""
import json
import sys
from scripts.fund_002112_data_completion_v1 import OUT,ROOT,read,save,sha,now
sys.path.insert(0,str(OUT/'ocr-deps'))
from rapidocr_onnxruntime import RapidOCR
import pypdfium2 as pdfium

def run():
    from scripts.fund_002112_data_completion_ocr_runtime import engine as make_engine
    engine=make_engine()
    inventory_file = sys.argv[1] if len(sys.argv) > 1 else OUT/'historical-scan-inventory.json'
    for item in read(inventory_file)['documents']:
        target=OUT/'historical-ocr'/(item['id']+'.json')
        if target.exists():continue
        original=read(item['source']);r=original['receipt'];raw=__import__('pathlib').Path(r['path']) if r.get('path') else ROOT/r['file']
        assert sha(raw)==r['sha256']
        pages=[];evidence=[]
        with pdfium.PdfDocument(str(raw)) as pdf:
            for i in range(len(pdf)):
                dest=OUT/'historical-ocr-pages'/item['id']/f'{i+1}.json'
                if dest.exists():rec=read(dest)
                else:
                    dest.parent.mkdir(parents=True,exist_ok=True);image=dest.with_suffix('.png')
                    page=pdf[i];bmp=page.render(scale=1.7);im=bmp.to_pil();im.save(image)
                    result,elapsed=engine(str(image));boxes=[{'box':r[0],'text':r[1],'confidence':float(r[2])} for r in result or []]
                    rec={'page':i+1,'image':str(image),'image_sha256':sha(image),'boxes':boxes,'text':'\n'.join(x['text'] for x in boxes),'elapsed':elapsed,'visually_verified':False}
                    save(dest,rec);im.close();bmp.close();page.close()
                pages.append(rec['text']);evidence.append(str(dest))
                print(json.dumps({'historical_ocr':item['id'],'page':i+1,'total':len(pdf)}),flush=True)
        save(target,{'at':now(),'source':item['source'],'source_sha256':sha(item['source']),'receipt':r,'row':original['row'],
            'pages':pages,'ocr_page_evidence':evidence,'original_preserved':True,'semantic_verified':False,'training_eligible':False})

if __name__=='__main__':run()
