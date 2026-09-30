"""重放已取 HTML，处理明确迁移链接、字符集、图片正文及附件；不重试访问拒绝。"""
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from scripts.fund_002112_data_completion_v1 import OUT, read, save, sha, now, fetch
from scripts.fund_002112_data_completion_public_v1 import inspect_html

def obtain(url,group,size):
    try:return fetch(url,group,size)
    except ValueError as error:return {'url':url,'ok':False,'reason':str(error)}

def run():
    while True:
        for p in (OUT/'public').glob('*.json'):
            dest=OUT/'public-recovered'/p.name
            if dest.exists():continue
            base=read(p);r=base['receipt'];chain=[]
            for _ in range(2):
                if r.get('status') not in (301,302,303,307,308):break
                next_url=urljoin(r['url'],r.get('headers',{}).get('location',''))
                if not next_url.startswith(('http://','https://')) or next_url==r['url']:break
                chain.append({'url':r['url'],'status':r['status'],'location':next_url})
                r=obtain(next_url,'redirects',12_000_000)
            rec={**base,'at_recovery':now(),'initial_evidence':str(p),'redirect_chain':chain,'effective_receipt':r}
            if r.get('ok',True) and r.get('path'):
                raw=Path(r['path']).read_bytes();assert sha(r['path'])==r['sha256']
                data=inspect_html(raw,{**base,'url':r['url']});rec.update(data)
                soup=BeautifulSoup(raw,'html.parser');content=soup.select_one('#zoom')
                image_records=[]
                if content and 'nhsa.gov.cn' in r['url']:
                    for img in content.find_all('img',src=True):
                        a=img.find_parent('a',href=True)
                        href=a['href'] if a and re.search(r'\.(jpg|png|jpeg)(?:[?#]|$)',a['href'],re.I) else img['src']
                        url=urljoin(r['url'],href)
                        if not (urlparse(url).hostname or '').endswith('nhsa.gov.cn'):continue
                        ir=obtain(url,'article_images',25_000_000)
                        image_records.append({'url':url,'receipt':ir,'alt':img.get('alt'),'title':img.get('title')})
                rec['body_images']=image_records
                # PDF/Word/Excel 附件再次按明确链接读取；成功回执按 URL 复用，不扣重复消费。
                attachments=[]
                for att in data['attachments']:
                    host=urlparse(att['url']).hostname or ''
                    if host==urlparse(r['url']).hostname or host.endswith('.gov.cn'):
                        ar=obtain(att['url'],'attachments',64_000_000)
                        attachments.append({**att,'receipt':ar,'semantic_verified':False})
                    else:attachments.append({**att,'reason':'CROSS_PUBLISHER_REVIEW_REQUIRED'})
                rec['attachments']=attachments
                meta=rec['metadata']; expected={re.sub(r'\s+','',x['title']) for x in base['aliases']}
                actual=re.sub(r'\s+','',rec.get('title') or '')
                rec['title_exact_or_site_prefix']=any(t and t in actual for t in expected)
                rec['publisher_day_matches_catalog']=bool(rec.get('publisher_date') and any(rec['publisher_date'][:10]==x['display_date'] for x in base['aliases']))
                rec['body_text_sha256']=hashlib.sha256(rec.get('text','').encode()).hexdigest()
            save(dest,rec)
        done=len(list((OUT/'public-recovered').glob('*.json')))
        print(json.dumps({'public_recovered':done}),flush=True)
        if (OUT/'public-collection-result.json').exists() and done==1475:break
        time.sleep(5)

if __name__=='__main__':run()
