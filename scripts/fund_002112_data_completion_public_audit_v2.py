"""逐条复核冻结公共材料：标题、刊发时间、正文、附件及可用时点。

同一 URL 的后续成功回执优先用于重新读取，不抹去中断和失败记录。
只认原站元数据或正文明确日期，不从目录日期反推正文发布时间。
"""
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from scripts.fund_002112_data_completion_v1 import OUT,read,save,sha,now

def norm(text):
    text=re.sub(r'^【[^】]+】','',text or '')
    return re.sub(r'[^\u4e00-\u9fffA-Za-z0-9]','',text).lower()

def decode(raw):
    declared=re.search(br'charset\s*=\s*["\x27]?([a-zA-Z0-9_-]+)',raw[:10000])
    codecs=[declared[1].decode() if declared else 'utf-8','utf-8','gb18030']
    for enc in dict.fromkeys(codecs):
        try:return raw.decode(enc),enc
        except (UnicodeError,LookupError):pass
    return raw.decode('utf-8',errors='replace'),'utf-8-replacement-unverified'

def day(text):
    m=re.search(r'(20\d{2})[-年/.](\d{1,2})[-月/.](\d{1,2})',str(text or ''))
    if not m:return None
    try:return date(*map(int,m.groups())).isoformat()
    except ValueError:return None

def effective_asset(item):
    """按 URL 读取已落盘回执，保留原失败，同时核对解析输入的原件摘要。"""
    canonical=item['url'];normalization=None
    np=OUT/'public-asset-url-normalization.json'
    if np.exists() and (read(np)['old_url']==canonical or canonical.rstrip()==read(np)['canonical_url']):
        normalization=read(np);canonical=normalization['canonical_url']
    key=hashlib.sha256(canonical.encode()).hexdigest()
    rp=OUT/'receipts'/f'{key}.json'
    receipt=read(rp) if rp.exists() else item.get('receipt',{})
    parsed=None
    for folder in ['policy-assets-v2','office-attachments','policy-assets']:
        p=OUT/folder/f'{key}.json'
        if p.exists() and read(p).get('parsed'):
            parsed=p;break
    ap=OUT/'attachment-format-audit'/f'{key}.json'
    if parsed is None and ap.exists():
        archive=read(ap)
        if archive.get('format')=='ZIP_ARCHIVE':
            members=[]
            for m in archive.get('members',[]):
                mk=__import__('pathlib').Path(m['receipt_path']).stem
                found=next((OUT/f/f'{mk}.json' for f in ['policy-assets-v2','office-attachments','policy-assets'] if (OUT/f/f'{mk}.json').exists() and read(OUT/f/f'{mk}.json').get('parsed')),None)
                members.append(found)
            if members and all(members):parsed=ap
    role=None
    rolepath=OUT/'public-asset-role-audit.json'
    if rolepath.exists():role=next((r for r in read(rolepath)['rows'] if r['url']==item['url']),None)
    return {'url':item['url'],'label':item.get('label'),'receipt':receipt,'url_normalization':normalization,
            'content_role_evidence':role,'required_body_asset':role is None,
            'original_receipt':item.get('receipt'),'parsed_path':str(parsed) if parsed else None,
            'parsed_sha256':sha(parsed) if parsed else None}

def run():
    version=sys.argv[1] if len(sys.argv)>1 else 'v2'
    rows=[]
    for p in sorted((OUT/'public-recovered').glob('*.json')):
        dest=OUT/('public-audit-'+version)/p.name
        if dest.exists():rows.append(read(dest));continue
        d=read(p);r=d.get('effective_receipt',d['receipt']);key=hashlib.sha256(r.get('url',d['url']).encode()).hexdigest()
        settled=OUT/'receipts'/f'{key}.json'
        if settled.exists() and read(settled).get('ok'):r=read(settled)
        alternative=OUT/'public-original-alternatives'/p.name
        if alternative.exists() and read(alternative).get('receipt',{}).get('ok'):
            r=read(alternative)['receipt']
        rec={'source':str(p),'source_sha256':sha(p),'url':d['url'],'aliases':d['aliases'],'receipt':r,
             'at':now(),'original_failures_preserved':True,'training_eligible':False,'gaps':[]}
        if alternative.exists():rec['official_alternative']=read(alternative)
        rendered_path=OUT/'public-rendered'/p.name
        rendered=read(rendered_path) if rendered_path.exists() else None
        # 旧分享页超时不否定后续正常公开 API 原件；两个回执均留档。
        if rendered and rendered.get('parsed') and rendered.get('receipt',{}).get('path') and rendered.get('receipt',{}).get('status',200)==200:
            rec['original_share_receipt']=r;r=rendered['receipt'];rec['receipt']=r
        if not r.get('path') or r.get('status',200)!=200:
            rec['gaps'].append({'reason':'BODY_UNAVAILABLE','details':r.get('reason'),'http_status':r.get('status')})
            save(dest,rec);rows.append(rec);continue
        raw=Path(r['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha256']
        html,encoding=decode(raw);soup=BeautifulSoup(html,'html.parser')
        if rendered and rendered.get('parsed'):
            from html import escape
            article=rendered['article']
            html='<html><head><title>'+escape(article.get('topic',''))+'</title><meta name="pubdate" content="'+escape(article.get('releasedate',''))+'"></head><body><article>'+article.get('content','')+'</article></body></html>'
            soup=BeautifulSoup(html,'html.parser');encoding='utf-8-official-public-api'
            rec['rendered_source']=str(rendered_path);rec['rendered_source_sha256']=sha(rendered_path)
            rec['body_version_updated_at']=article.get('lastupdatetime')
            rec['source_display_date_field']=rendered.get('source_display_date_field','releasedate')
            rec['first_publication_not_separately_exposed']=rendered.get('first_publication_not_separately_exposed',False)
        meta={str(m.get('name') or m.get('property')).lower():m.get('content','') for m in soup.find_all('meta') if m.get('content')}
        titles=[meta.get('articletitle'),meta.get('og:title'),soup.title.get_text() if soup.title else None]
        titles += [e.get_text(' ',strip=True) for e in soup.select('h1,.artTit,.article-title,.title')[:6]]
        host=urlparse(d['url']).hostname
        if host=='paper.ce.cn':titles += [e.get_text(' ',strip=True) for e in soup.select('td.font01')]
        if host=='paper.people.com.cn':
            headings=soup.select('.article-box h1,.article-box h2,.article-box h3')
            if headings:titles.append(''.join(e.get_text('',strip=True) for e in headings))
        expected=[norm(a['title']) for a in d['aliases']]
        titleproof=next((t for t in titles if t and any((len(x)>=2 and x==norm(t)) or len(x)>=10 and (x in norm(t) or len(norm(t))>=12 and norm(t) in x) for x in expected)),None)
        host=urlparse(r.get('url',d['url'])).hostname
        host_selectors={'health.people.com.cn':['.artDet'], 'www.hubpd.com':['.content .txt'],
            'wap.peopleapp.com':['#newsContent'],'www.peopleapp.com':['#newsContent'],'m.news.cctv.com':['.cnt_bd'],
            'epaper.gmw.cn':['#articleContent'],'www.xinhuanet.com':['#detail','#p-detail'],
            'www.chinanews.com.cn':['.left_zw'],
            'm.bjnews.com.cn':['.article-cen'],'china.cnr.cn':['.article-body'],'finance.cnr.cn':['.article-body'],
            'wjw.xinjiang.gov.cn':['.d_m_centre'],'paper.ce.cn':['#ozoom'],'paper.people.com.cn':['#ozoom','#articleContent']}
        selectors=host_selectors.get(host,[])+['#zoom','#zoomcon','.TRS_Editor','#UCAP-CONTENT','.article-content','#content','article','.rm_txt_con','.artBody','.content_area','.text_con','#rwb_zw','.pages_content','#Content','.正文']
        container=next((soup.select_one(s) for s in selectors if soup.select_one(s)),None)
        selector=next((s for s in selectors if soup.select_one(s)),None)
        if container is None:container=soup.body or soup
        for t in container.select('script,style,nav,header,footer'):t.decompose()
        body=container.get_text('\n',strip=True)
        dates=[(k,day(meta.get(k))) for k in ['pubdate','publishdate','firstpublishedtime','dcterms.issued','date','weixin:article_create_time','article:published_time','ptime'] if day(meta.get(k))]
        if host in ['paper.ce.cn','paper.people.com.cn']:
            selector_date='span.default' if host=='paper.ce.cn' else '.article-box span.date'
            printed=[('PRINTED_NEWSPAPER_ISSUE_DATE',day(e.get_text('',strip=True))) for e in soup.select(selector_date) if day(e.get_text('',strip=True))]
            if printed:
                rec['website_template_date_metadata_preserved']=dates;dates=printed
        if not dates:
            for e in soup.select('.time,.date,.pubtime,.publish_time,.h-time,.pages-date,#pubtime_baidu,.source,#pubtime,.artInfo,.origin,.artOri,.extra,.info,.lai'):
                text=e.get_text(' ',strip=True)
                if day(text):dates.append(('visible_publication_element',day(text)));break
        catalogdays={a['display_date'] for a in d['aliases']}
        pub=next((v for _,v in dates if v in catalogdays),dates[0][1] if dates else None)
        dateproof=bool(pub and any(0<=(date.fromisoformat(v)-date.fromisoformat(pub)).days<=7 for v in catalogdays))
        if rendered and rendered.get('parsed') and (rendered.get('body_id_verified') or rendered.get('public_api_id_bound_to_frozen_url')) and rec.get('source_display_date_field')!='updatedAt' and titleproof and pub and all(pub<=v for v in catalogdays):
            # 官方 API 精确文章编号已经绑定原目录链接；目录转载日不是原稿刊发日。
            # 明确保存两个日期，最早研究可用日仍取较晚者，不把转载间隔当作版本冲突。
            dateproof=True
            rec['source_and_catalog_repost_dates_separated']={'source_publication':pub,'catalog_dates':sorted(catalogdays),
                'source_article_id_bound':bool(rendered.get('body_id_verified') or rendered.get('public_api_id_bound_to_frozen_url')),
                'available_date_never_moved_earlier_than_catalog':True}
        images=[]
        for item in d.get('body_images',[]):
            images.append(effective_asset(item))
        extra_scope=OUT/'public-body-media-scope-v5.json'
        if not extra_scope.exists():extra_scope=OUT/'public-body-media-scope-v4.json'
        if not extra_scope.exists():extra_scope=OUT/'public-body-media-scope-v3.json'
        if not extra_scope.exists():extra_scope=OUT/'public-body-media-scope-v2.json'
        if extra_scope.exists():
            known={x['url'] for x in images}
            for url in read(extra_scope).get('by_document',{}).get(p.name,[]):
                if url not in known:images.append(effective_asset({'url':url}));known.add(url)
        attachments=[]
        for item in d.get('attachments',[]):
            attachments.append(effective_asset(item))
        # 只提取正文明确写出的法定时间和引用，不将“计划、征求意见”当生效。
        paragraphs=[t for t in body.splitlines() if t.strip()]
        facts=[]
        for i,t in enumerate(paragraphs):
            if re.search(r'自.{0,50}(?:施行|实施|起执行)|废止|有效期|医保[办发函].{0,12}\d{1,4}号|发文字号',t):
                facts.append({'paragraph':i+1,'quote':t,'classification':'DRAFT' if '征求意见' in ''.join(t or '' for t in titles) else 'LITERAL_CLAUSE','effect_inferred':False})
        required_images=[i for i in images if i.get('required_body_asset',True)]
        imagebody=bool(required_images and all(i['parsed_path'] for i in required_images))
        bodyok=bool(selector and (len(body)>80 or host=='www.nhsa.gov.cn' and len(body)>10 or imagebody) and 'utf-8-replacement' not in encoding)
        if host in ['tv.cctv.com','app.cctv.com','w.yangshipin.cn','vod-finance.cctv.cn']:
            bodyok=False;rec['gaps'].append({'reason':'VIDEO_PAGE_SUMMARY_IS_NOT_FULL_TRANSCRIPT'})
        if not titleproof:rec['gaps'].append({'reason':'TITLE_NOT_MATCHED','observed_titles':titles})
        if not dateproof:rec['gaps'].append({'reason':'PUBLICATION_DAY_NOT_VERIFIED','observed':dates})
        if not bodyok:rec['gaps'].append({'reason':'BODY_CONTAINER_OR_TRANSCRIPTION_UNVERIFIED'})
        if any(not x['parsed_path'] for x in attachments+required_images):rec['gaps'].append({'reason':'ATTACHMENT_OR_IMAGE_NOT_PARSED'})
        updated=day(rec.get('body_version_updated_at'))
        if updated and pub and updated>pub:
            rec['gaps'].append({'reason':'BODY_UPDATED_AFTER_ORIGINAL_PUBLICATION','updated_date':updated,'original_date':pub})
        if rec.get('first_publication_not_separately_exposed'):
            rec['gaps'].append({'reason':'SOURCE_DISPLAYS_UPDATED_TIME_ONLY_FIRST_PUBLICATION_NOT_SEPARATELY_PROVEN'})
        rec.update({'title_proof':titleproof,'published_date':pub,'date_evidence':dates,'title_identity_verified':bool(titleproof),
            'publication_day_verified':dateproof,'conservative_available_at':str(date.fromisoformat(max([pub or '0001-01-01',updated or '0001-01-01',*catalogdays]))+timedelta(days=1))+'T08:00:00+08:00',
            'encoding':encoding,'selector':selector,'text':body,'text_sha256':hashlib.sha256(body.encode()).hexdigest(),
            'body_verified':bodyok,'images':images,'attachments':attachments,'literal_policy_clauses':facts,
            'source_body_identity_time_verified':bool(titleproof and dateproof and bodyok),
            'version_complete':False,'full_policy_effect_verified':False,'numeric_ocr_requires_original_check':imagebody,
            'market_direction_inferred':False,'first_seen_historical_proven':False})
        save(dest,rec);rows.append(rec)
    save(OUT/('public-audit-result-'+version+'.json'),{'at':now(),'rows':len(rows),
        'source_body_identity_time_verified':sum(bool(r.get('source_body_identity_time_verified')) for r in rows),
        'reason_counts':dict(Counter(g['reason'] for r in rows for g in r['gaps']))})

if __name__=='__main__':run()
