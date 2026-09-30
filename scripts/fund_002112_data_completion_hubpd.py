"""复用公开分享页无登录正文接口；不调用评论、赞、举报、日志等写入口。"""
import re
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime,timezone,timedelta
from scripts.fund_002112_data_completion_v1 import OUT,read,save,sha,now,fetch

def run():
 scope=[]
 for p in (OUT/'public-audit-v7').glob('*.json'):
  r=read(p)
  if urlparse(r['url']).hostname!='www.hubpd.com' or r.get('source_body_identity_time_verified'):continue
  m=re.search(r'contentId=(\d+)',r['url'])
  if m:scope.append({'source':str(p),'id':m[1],'api_url':f'https://api-m.hubpd.com/content?contentId={m[1]}&andHot=true'})
 save(OUT/'public-client-api/hubpd-exact-scope.json',{'at':now(),'rows':scope,'observed_in_browser':'https://www.hubpd.com/#/detail?contentId=6341068275338332665',
  'no_account_cookie_or_userId':True,'uses_remaining_alternatives_budget':True,'original_failed_shells_kept':True})
 results=[]
 for item in scope:
  r=fetch(item['api_url'],'alternatives');d=read(r['path']).get('data',{}) if r.get('ok') else {};source=Path(item['source']);old=read(source)
  when=datetime.fromtimestamp(d['updatedAt'],timezone(timedelta(hours=8))).isoformat() if d.get('updatedAt') else None
  rec={'at':now(),'source_audit':str(source),'source_audit_sha256':sha(source),'url':old['url'],'receipt':r,
   'parsed':bool(str(d.get('trueId'))==item['id'] and d.get('title') and d.get('detail') and when),
   'article':{'topic':d.get('title'),'content':d.get('detail'),'releasedate':when,'lastupdatetime':when},
   'original_article':d,'source_display_date_field':'updatedAt','first_publication_not_separately_exposed':True,
   'public_api_id_bound_to_frozen_url':str(d.get('trueId'))==item['id']}
  save(OUT/'public-rendered'/source.name,rec);results.append({'id':item['id'],'parsed':rec['parsed']})
 save(OUT/'public-client-api/hubpd-result.json',{'at':now(),'rows':results})

if __name__=='__main__':run()
