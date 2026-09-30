"""按正常公开移动页观察到的只读内容接口，补齐同一冻结文章的正文。"""
import re,hashlib,sys
from urllib.parse import urlparse
from scripts.fund_002112_data_completion_v1 import OUT,read,save,sha,now,fetch

def run():
 supplemental=len(sys.argv)>1 and sys.argv[1]=='v2';suffix='-v2' if supplemental else ''
 scope=[]
 for p in (OUT/('public-audit-v9' if supplemental else 'public-audit-v7')).glob('*.json'):
  r=read(p)
  if urlparse(r['url']).hostname!=('dw.chinanews.com' if supplemental else 'm.chinanews.com') or r.get('source_body_identity_time_verified'):continue
  m=re.search(r'id=(\d+)' if supplemental else r'/(\d+)\.shtml',r['url'])
  if m:scope.append({'source':str(p),'id':m[1],'api_url':f'https://dw.chinanews.com/cns/app/v1/wapDetail/content/{m[1]}.json?language=chs'})
 save(OUT/('public-client-api/chinanews-exact-scope'+suffix+'.json'),{'at':now(),'rows':scope,'observed_in_browser':'https://m.chinanews.com/wap/detail/chs/zw/9372796.shtml',
  'public_api_requires_no_account':True,'uses_remaining_alternatives_budget':True,'request_limit_not_reset':True})
 results=[]
 for item in scope:
  r=fetch(item['api_url'],'alternatives');d=read(r['path']).get('data',{}) if r.get('ok') else {};source=__import__('pathlib').Path(item['source']);old=read(source)
  rec={'at':now(),'source_audit':str(source),'source_audit_sha256':sha(source),'url':old['url'],'receipt':r,
   'parsed':bool(str(d.get('id'))==item['id'] and d.get('title') and d.get('content') and d.get('pubtime')),
   'article':{'topic':d.get('title'),'content':d.get('content'),'releasedate':d.get('pubtime'),'lastupdatetime':d.get('freshTime')},
   'original_article':d,'public_api_id_bound_to_frozen_url':str(d.get('id'))==item['id'],'video_not_downloaded_or_transcribed':bool(d.get('video'))}
  save(OUT/'public-rendered'/source.name,rec);results.append({'id':item['id'],'parsed':rec['parsed']})
 save(OUT/('public-client-api/chinanews-result'+suffix+'.json'),{'at':now(),'rows':results})

if __name__=='__main__':run()
