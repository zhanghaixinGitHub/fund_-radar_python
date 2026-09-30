"""读取冻结链接对应的新华社公开 H5 正文；匿名协议来自现场已显示正文的官方脚本。

不读取账户或浏览器凭证，不调用登录、评论、订阅或点赞接口。每个冻结文章
仅请求一次正文，先写请求账；旧失败回执不覆盖，网络消费单独累计。
"""
import hashlib,json,re,time,sys
from pathlib import Path
import httpx
from bs4 import BeautifulSoup
from scripts.fund_002112_data_completion_v1 import OUT,read,save,sha,now

def run():
    supplemental=len(sys.argv)>1 and sys.argv[1]=='v2'
    rows=[]
    for p in (OUT/('public-audit-v8' if supplemental else 'public-audit-v4')).glob('*.json'):
        d=read(p)
        pattern=r'h\.xinhuaxmt\.com/vh512/share/\d+' if supplemental else r'xhpfmapi\.(?:zhongguowangshi|xinhuaxmt)\.com/vh512/share/\d+'
        if re.search(pattern,d['url']) and (not supplemental or not d.get('source_body_identity_time_verified')):rows.append((p,d))
    scope=OUT/('public-client-api/xinhua-exact-scope-v2.json' if supplemental else 'public-client-api/xinhua-exact-scope.json')
    if not scope.exists():save(scope,{'at':now(),'rows':[{'audit':str(p),'url':d['url']} for p,d in rows],'old_consumption_preserved':True,'new_model_budget':False,'maximum_requests':len(rows)})
    for p,d in rows:
        id=re.search(r'/share/(\d+)',d['url'])[1];request=OUT/'public-client-api/requests'/f'xh-{id}.json';receipt=OUT/'public-client-api/receipts'/f'xh-{id}.json'
        if receipt.exists():r=read(receipt)
        elif request.exists():r={'error':'INTERRUPTED_REQUEST_PRESERVED'}
        else:
            stamp=str(int(time.time()*1000));key=hashlib.new('sm3',b'H5').hexdigest()
            sign=hashlib.new('sm3',f'Key={key}&Timestamp={stamp}&Token=&Request=sign='.encode()).hexdigest()
            url=f'https://h.xinhuaxmt.com/1017/n/newsapi/h5/news-detail/{id}?sign='
            save(request,{'at':now(),'url':url,'anonymous_public_h5':True,'frozen_url':d['url'],'credential_used':False,'timestamp':stamp,'scope':str(scope)})
            try:
                with httpx.Client(timeout=45,trust_env=False) as c:response=c.get(url,headers={'Timestamp':stamp,'Signature':sign,'Device-Access-Id':''})
                raw=OUT/'public-client-api/raw'/f'xh-{id}.json';raw.write_bytes(response.content)
                r={'status':response.status_code,'path':str(raw),'sha256':sha(raw),'bytes':len(response.content)}
            except Exception as e:r={'error':type(e).__name__+':'+str(e)[:200]}
            save(receipt,r);time.sleep(.4)
        result={'at':now(),'source_audit':str(p),'source_audit_sha256':sha(p),'url':d['url'],'receipt':r,'parsed':False,'gaps':[]}
        try:
            data=read(r['path']);assert str(data['code'])=='0',data.get('message')
            # 只解码 JSON 对象，不执行返回的 JavaScript 包装。
            value=data['data'];article=json.JSONDecoder().raw_decode(value[value.index('{'):])[0] if isinstance(value,str) else value
            assert str(article['id'])==id
            result.update(parsed=True,article=article,title=article.get('topic'),published=article.get('releasedate'),last_updated=article.get('lastupdatetime'),
                          body=BeautifulSoup(article.get('content',''),'html.parser').get_text('\n',strip=True),body_id_verified=True)
        except Exception as e:result['gaps'].append(type(e).__name__+':'+str(e))
        dest=OUT/'public-rendered'/p.name
        if not dest.exists():save(dest,result)
        print(json.dumps({'xinhua':id,'parsed':result['parsed'],'gaps':result['gaps']}),flush=True)

if __name__=='__main__':run()
