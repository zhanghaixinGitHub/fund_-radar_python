"""按经济日报公开网页的匿名访问约定读取冻结文章，不使用任何用户登录凭证。"""
import json,re,time
from urllib.parse import urlparse,parse_qs
import httpx
from scripts.fund_002112_data_completion_v1 import OUT,read,save,sha,now
from bs4 import BeautifulSoup

def run():
    script=OUT/'raw/23a4d2c432ac8e22cc2de7c8ee05e6862570065faefa05a49b210735f2804a68'
    # 官方脚本为公开分享页明示默认未登录标识，不能替换为用户账户值。
    marker=re.search(r'userId:null,token:"([^"]+)"',script.read_text('utf-8'))[1]
    wanted=[]
    for p in (OUT/'public-audit-v5').glob('*.json'):
        d=read(p)
        if urlparse(d['url']).hostname=='proapi.jingjiribao.cn':wanted.append((p,d))
    scope=OUT/'public-client-api/jjrb-exact-scope.json'
    if not scope.exists():save(scope,{'at':now(),'articles':[d['url'] for p,d in wanted],'maximum_requests':len(wanted),
        'official_public_share_page_verified_in_browser':True,'protocol_source':str(script),'protocol_source_sha256':sha(script),'uses_user_credentials':False,'previous_failed_probe_preserved':True})
    for p,d in wanted:
        id=parse_qs(urlparse(d['url']).query)['id'][0];request=OUT/'public-client-api/requests'/f'jjrb-public-{id}.json';receipt=OUT/'public-client-api/receipts'/f'jjrb-public-{id}.json'
        if receipt.exists():r=read(receipt)
        elif request.exists():r={'error':'INTERRUPTED_REQUEST_PRESERVED'}
        else:
            save(request,{'at':now(),'url':'https://proapi.jingjiribao.cn/api/news/detail','method':'POST','json':{'id':id},'scope':str(scope),'public_anonymous_protocol':True})
            try:
                with httpx.Client(timeout=45,trust_env=False) as c:response=c.post('https://proapi.jingjiribao.cn/api/news/detail',json={'id':id},headers={'token':marker})
                raw=OUT/'public-client-api/raw'/f'jjrb-public-{id}.json';raw.write_bytes(response.content)
                r={'status':response.status_code,'path':str(raw),'sha256':sha(raw),'bytes':len(response.content)}
            except Exception as e:r={'error':type(e).__name__+':'+str(e)[:160]}
            save(receipt,r);time.sleep(.4)
        result={'at':now(),'source_audit':str(p),'url':d['url'],'receipt':r,'parsed':False,'gaps':[]}
        try:
            response=read(r['path']);assert str(response['code'])=='0',response.get('message')
            article=response['data'];result.update(parsed=True,original_article=article)
            result['article']={'topic':article.get('title'), 'content':article.get('content'),'releasedate':article.get('publishTime') or article.get('releaseTime') or article.get('releaseDate') or article.get('ctime'),'lastupdatetime':article.get('updateTime')}
        except Exception as e:result['gaps'].append(type(e).__name__+':'+str(e))
        target=OUT/'public-rendered'/p.name
        if not target.exists():save(target,result)
        print(json.dumps({'jjrb':id,'parsed':result['parsed'],'gaps':result['gaps']}),flush=True)

if __name__=='__main__':run()
