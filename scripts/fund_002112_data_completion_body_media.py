"""补已冻结正文中的明确图片资源；原图优先，已取缩略图对应原件不重复下载。"""
import hashlib,re,json
from urllib.parse import urljoin
from pathlib import Path
from bs4 import BeautifulSoup
from scripts.fund_002112_data_completion_v1 import OUT,read,save,sha,now,fetch
from scripts.fund_002112_data_completion_public_audit_v2 import decode

def run():
    scope=OUT/'public-body-media-scope-v2.json'
    if not scope.exists():
        urls={};bydoc={}
        for p in (OUT/'public-audit-v5').glob('*.json'):
            d=read(p)
            if not d.get('selector'):continue
            if d.get('rendered_source'):container=BeautifulSoup(read(d['rendered_source'])['article'].get('content',''),'html.parser')
            else:
                soup=BeautifulSoup(decode(Path(d['receipt']['path']).read_bytes())[0],'html.parser');container=soup.select_one(d['selector'])
            if container is None:continue
            links=[]
            for img in container.find_all('img'):
                src=img.get('data-src') or img.get('src');parent=img.find_parent('a',href=True)
                if parent and re.search(r'\.(?:jpg|jpeg|png)(?:[?#]|$)',parent['href'],re.I):src=parent['href']
                if not src or src.startswith('data:'):continue
                u=urljoin(d['receipt'].get('url',d['url']),src)
                if not u.startswith(('http://','https://')):continue
                if any(v in u.lower() for v in ['logo','qrcode','loading','headpic','share-icon','blank.gif']):continue
                links.append(u);key=hashlib.sha256(u.encode()).hexdigest()
                if not (OUT/'receipts'/f'{key}.json').exists():urls.setdefault(u,[]).append(p.name)
            if links:bydoc[p.name]=list(dict.fromkeys(links))
        save(scope,{'at':now(),'urls':urls,'by_document':bydoc,'old_scope_retained':True,'old_image_consumption_preserved':1116,'model_budget_not_changed':True})
    for u in read(scope)['urls']:
        r=fetch(u,'body_media',25_000_000)
        print(json.dumps({'url':u,'ok':r.get('ok'),'status':r.get('status'),'reason':r.get('reason')}),flush=True)
    save(OUT/'body-media-complete.json',{'at':now(),'scope':str(scope),'scope_sha256':sha(scope)})

if __name__=='__main__':run()
