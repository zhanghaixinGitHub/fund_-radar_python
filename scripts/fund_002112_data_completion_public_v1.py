"""补齐已冻结公共栏目逐条正文与明确附件，保留语义及历史关联缺口。"""
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from scripts.fund_002112_data_completion_v1 import OUT, read, save, sha, now, inventory, fetch

def inspect_html(raw, spec):
    """保留正文、原始标题、元数据及原文段落；不由关键词推定政策影响。"""
    soup = BeautifulSoup(raw, 'html.parser')
    meta = {str(m.get('name') or m.get('property')): m.get('content') for m in soup.find_all('meta') if m.get('content')}
    content = next((soup.select_one(s) for s in ['#zoom', '#zoomcon', '.TRS_Editor', '#UCAP-CONTENT', '.article-content', '#content', 'article'] if soup.select_one(s)), None)
    extraction = 'publisher_content_container'
    if content is None:
        content = soup.body or soup
        extraction = 'whole_body_requires_review'
    attachments = []
    for a in content.find_all('a', href=True):
        href = urljoin(spec['url'], a['href'])
        label = a.get_text(' ', strip=True)
        if re.search(r'\.(pdf|docx?|xlsx?|zip|rar)(?:[?#]|$)',href,re.I) or '附件' in label:
            attachments.append({'url':href, 'label':label})
    for t in content(['script','style','nav','header','footer']):
        t.decompose()
    text = content.get_text('\n', strip=True)
    title = meta.get('ArticleTitle') or meta.get('og:title') or (soup.title.get_text(strip=True) if soup.title else None)
    publisher_date = meta.get('PubDate') or meta.get('pubdate') or meta.get('publishdate') or meta.get('DCTERMS.issued')
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    clues = []
    for n,p in enumerate(paragraphs):
        if re.search(r'施行|实施|废止|修订|有效期|发文字号|医保发|医保办发|自.{0,30}起|亿元|万元|%|采购|支付标准',p):
            clues.append({'paragraph':n+1, 'text':p})
    return {'text':text,'title':title,'publisher_date':publisher_date,'metadata':meta,
        'extraction':extraction,'attachments':attachments,'fact_paragraphs':clues,
        'body_obtained':len(text)>120,'semantic_verified':False,'historical_relation_verified':False,
        'version_history_complete':False,'training_eligible':False}

def run():
    specs = inventory()['public']
    for n,spec in enumerate(specs,1):
        import hashlib
        key=hashlib.sha256(spec['url'].encode()).hexdigest()
        dest=OUT/'public'/f'{key}.json'
        if dest.exists():
            continue
        result={**spec,'at':now(),'body_obtained':False,'training_eligible':False}
        receipt=spec['cached'][0] if spec['cached'] else fetch(spec['url'],'public',8_000_000)
        result['receipt']=receipt
        if receipt.get('ok',True):
            try:
                raw=Path(receipt['path']).read_bytes()
                assert sha(receipt['path'])==receipt['sha256']
                result.update(inspect_html(raw,spec))
                checked=[]
                for item in result['attachments']:
                    host=urlparse(item['url']).hostname or ''
                    parent=urlparse(spec['url']).hostname or ''
                    if host==parent or host.endswith(('.gov.cn','.nhsa.gov.cn')):
                        try:
                            ar=fetch(item['url'],'attachments',64_000_000)
                        except ValueError as e:
                            ar={'ok':False,'reason':str(e)}
                        checked.append({**item,'receipt':ar,'content_semantics_verified':False})
                    else:
                        checked.append({**item,'reason':'CROSS_PUBLISHER_ATTACHMENT_REQUIRES_IDENTITY_REVIEW'})
                result['attachments']=checked
            except Exception as e:
                result['reason']=type(e).__name__+':'+str(e)[:300]
        else:
            result['reason']=receipt.get('reason')
        save(dest,result)
        if n%10==0:
            print(json.dumps({'public_processed':n,'total':len(specs),'last_body':result['body_obtained']},ensure_ascii=False),flush=True)
    rows=[read(p) for p in (OUT/'public').glob('*.json')]
    save(OUT/'public-collection-result.json',{'at':now(),'scope':len(specs),'processed':len(rows),
        'body_obtained':sum(x['body_obtained'] for x in rows),
        'attachments':sum(len(x.get('attachments',[])) for x in rows),
        'failures':dict(Counter(x.get('reason') for x in rows if not x['body_obtained'])),
        'full_semantics_verified':False,'new_fits':0})

if __name__=='__main__':
    run()
