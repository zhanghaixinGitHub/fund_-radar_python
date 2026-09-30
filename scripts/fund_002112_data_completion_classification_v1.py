"""按已发现的证监会历史目录补分类原件；仅取截止日前的季度版本。"""
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from scripts.fund_002112_data_completion_v1 import OUT,read,save,sha,now,fetch
from app.services.fund_earnings_batch_v4 import extract_pdf

def run():
    folder=OUT/'industry-classification'
    specs={r['url']:r for p in folder.glob('listing-*.json') for r in read(p)['entries']
        if not r['title'].startswith('2015') or '4季度' in r['title']}
    contract=folder/'contract.json'
    if not contract.exists():save(contract,{'at':now(),'scope':list(specs.values()),'purpose':'2016-2023截止前可用行业分类，2015Q4是2016早期窗口的最近公开版本','same_existing_alternatives_budget':120,'current_classification_forbidden':True})
    for url,spec in specs.items():
        key=hashlib.sha256(url.encode()).hexdigest();dest=folder/'documents'/f'{key}.json'
        if dest.exists():continue
        r=fetch(url,'alternatives',3_000_000);rec={**spec,'receipt':r,'attachments':[],'training_eligible':False}
        if r.get('ok'):
            soup=BeautifulSoup(Path(r['path']).read_bytes(),'html.parser');text=soup.get_text('\n',strip=True)
            rec['article_text']=text;rec['publication_day']=re.search(r'20\d{2}-\d{2}-\d{2}',spec['context'])[0]
            rec['metadata']={m.get('name'):m.get('content') for m in soup.find_all('meta',content=True)}
            for a in soup.find_all('a',href=True):
                if not re.search(r'\.(pdf|xlsx?|docx?)(?:[?#]|$)',a['href'],re.I):continue
                u=urljoin(url,a['href']);ar=fetch(u,'alternatives',35_000_000);att={'label':a.get_text(strip=True),'receipt':ar}
                if ar.get('ok'):
                    try:att['pages'],att['metadata']=extract_pdf(Path(ar['path']).read_bytes(),1200)
                    except Exception as e:att['parse_error']=type(e).__name__+':'+str(e)
                rec['attachments'].append(att)
        save(dest,rec);print(json.dumps({'classification':spec['title'],'files':len(rec['attachments'])},ensure_ascii=False),flush=True)
    save(folder/'collection-result.json',{'at':now(),'documents':len(list((folder/'documents').glob('*.json'))),'expected':len(specs),'originals_only_not_classification_approval':True})

if __name__=='__main__':run()
