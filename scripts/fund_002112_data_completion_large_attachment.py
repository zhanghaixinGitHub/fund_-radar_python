"""逐页处理已核实为 1710 页的 DRG 附件，保留先前页数限额停止记录。"""
import re
import pypdfium2 as pdfium
from scripts.fund_002112_data_completion_v1 import OUT,read,save,sha,now

def run():
    wanted='4b4e6228e717cb18433bf2e3e097a5b02537e4bb2741b9d2c237bbf387fda699'
    for rp in (OUT/'asset-receipts').glob('*.json'):
        receipt=read(rp)
        if receipt['sha256']!=wanted:continue
        old=OUT/'policy-assets'/rp.name
        assert read(old)['error']=='ValueError:PDF_PAGE_LIMIT'
        assert sha(receipt['path'])==wanted
        pages=[];short=[]
        with pdfium.PdfDocument(receipt['path']) as pdf:
            assert len(pdf)==1710
            for i in range(len(pdf)):
                pg=pdf[i];text=pg.get_textpage();value=text.get_text_range();text.close();pg.close();pages.append(value)
                if len(re.sub(r'\s+','',value))<30:short.append(i+1)
            metadata=pdf.get_metadata_dict()
        save(OUT/'policy-assets-v2'/rp.name,{'at':now(),'receipt':receipt,'previous_stop':str(old),'previous_stop_sha256':sha(old),
            'page_count':len(pages),'pages':pages,'metadata':metadata,'short_or_blank_pages':short,'parsed':True,'streamed_page_by_page':True,
            'semantic_numeric_table_layout_verified':False,'training_eligible':False})
        save(OUT/'large-attachment-result.json',{'at':now(),'pages':len(pages),'raw_sha256':wanted,'parsed_path':str(OUT/'policy-assets-v2'/rp.name),'blank_or_short_page_count':len(short),'old_stop_preserved':True})

if __name__=='__main__':run()
