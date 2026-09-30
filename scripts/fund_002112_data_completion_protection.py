"""收尾只读复核：旧文件仅做字节摘要，不解析封存答案；数据库只返回摘要。"""
import json,subprocess,tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from sqlalchemy import create_engine,text
from sqlalchemy.engine import URL
from app.core.config import get_settings
from scripts.fund_002112_data_completion_v1 import OUT,ROOT,PY,WEB,JAVA,read,save,sha,now

def runtime(destination='runtime-after.json'):
 cfg={}
 for line in (JAVA/'.env').read_text(encoding='utf-8-sig').splitlines():
  if '=' in line and not line.strip().startswith('#'):
   k,v=line.split('=',1);cfg[k.strip()]=v.strip()
 core=URL.create('postgresql+psycopg',username=cfg['FUND_CORE_DB_USERNAME'],password=cfg['FUND_CORE_DB_PASSWORD'],host=cfg.get('FUND_CORE_DB_HOST','localhost'),port=int(cfg.get('FUND_CORE_DB_PORT','54329')),database=cfg.get('FUND_CORE_DB_NAME','fund_core'))
 result={'at':now(),'read_only':True}
 for name,url in [('ai',get_settings().ai_database_url),('core',core)]:
  engine=create_engine(url,hide_parameters=True,connect_args={'connect_timeout':5,'options':'-c default_transaction_read_only=on -c statement_timeout=15000'})
  assert engine.url.host in {'localhost','127.0.0.1'}
  with engine.connect() as c:
   def rows(q):return [dict(r) for r in c.execute(text(q)).mappings()]
   d={'locks':rows("select classid,objid,mode from pg_locks where locktype='advisory' and granted")}
   if name=='ai':
    d['models']=rows('select model_id::text as id,content_hash as hash from direction_1d_model order by model_id')
    d['active_jobs']=rows("select kind,state,count(*) as count from direction_1d_job where state in ('QUEUED','RUNNING') group by kind,state")
    d['latest_nav']=rows("select max(nav_date)::text as latest,count(*) as count from nav_daily where fund_code='002112'")
    from app.services.direction_1d_training import MODEL_ROOT
    d['model_files']=[{'id':r['id'],'actual':sha(MODEL_ROOT/(r['id']+'.json')),'matches_registry':sha(MODEL_ROOT/(r['id']+'.json'))==r['hash']} for r in d['models']]
   else:d['forecasts']=rows("select forecast_id::text as id,content_hash as hash,encode(sha256(convert_to(payload_json,'UTF8')),'hex') as actual from direction_1d_forecast order by forecast_id")
   result[name]=d
  engine.dispose()
 before=read(OUT/'runtime-before.json')
 result['models_unchanged']=result['ai']['models']==before['ai']['models']
 result['old_predictions_unchanged']=all(r in result['core']['forecasts'] for r in before['core']['forecasts'])
 result['old_prediction_count']=len(before['core']['forecasts'])
 result['new_prediction_count']=len(result['core']['forecasts'])-len(before['core']['forecasts'])
 assert result['models_unchanged'] and result['old_predictions_unchanged'] and all(r['hash']==r['actual'] for r in result['core']['forecasts'])
 save(OUT/destination,result)

def protection():
 before=read(OUT/'protection-before.json');changed=[];missing=[];checked=0;research_changed=[]
 def check_original(item):
  path,expected=item;p=Path(path)
  return path,expected,sha(p) if p.is_file() else None
 with ThreadPoolExecutor(max_workers=4) as pool:
  for path,expected,actual in pool.map(check_original,before['files'].items(),chunksize=32):
   p=Path(path)
   if actual is None:missing.append(path);print(json.dumps({'missing':path}),flush=True);continue
   checked+=1
   if actual!=expected:
    item={'path':path,'before':expected,'after':actual};changed.append(item);print(json.dumps({'changed':item},ensure_ascii=False),flush=True)
    if ROOT in p.parents:research_changed.append(item)
   if checked%20000==0:print(json.dumps({'checked':checked}),flush=True)
 workspaces=[]
 for w in before['workspaces']:
  repo=Path(w['path']);head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
  status=subprocess.check_output(['git','-C',str(repo),'status','--porcelain=v1','-uall'],text=True,encoding='utf-8')
  check=subprocess.run(['git','-C',str(repo),'diff','--check'],capture_output=True,text=True,encoding='utf-8')
  workspaces.append({'path':str(repo),'head':head,'head_unchanged':head==w['head'],'status':status,'diff_check_exit':check.returncode,'diff_check':check.stdout+check.stderr})
 budget=read(ROOT/'information-research/20260928-v1/budget-authorized.json');fits=list((ROOT/'information-research/20260928-v1/fit-ledger').glob('*.json'))
 auto=Path('C:/Users/a/.codex/automations/002112/automation.toml');a=tomllib.loads(auto.read_text(encoding='utf-8'))
 rec={'at':now(),'checked_original_files':checked,'original_files':len(before['files']),'missing':missing,'changed':changed,
  'research_changes':research_changed,'historical_original_files_preserved':not research_changed and not [p for p in missing if str(ROOT) in p],
  'workspaces':workspaces,'old_requests_consumed':len(list((ROOT/'closure/20260929-v1/company-bodies/requests').glob('*.json'))),
  'cumulative_fits':budget['previous_actual_fits']+len(fits),'fit_ledger_sha256':{str(p):sha(p) for p in fits},'new_fits':0,
  'automation':{'name':a.get('name'),'status':a.get('status'),'path':str(auto),'sha256':sha(auto)},
  'parallel_chat':{'id':'01a0f020-79fd-71a0-901e-cacdd76bacfa','observed_status':'idle','latest_turn':'01a0f066-ddf8-7b32-92a1-165e33d5c448','changes_not_reverted':True}}
 save(OUT/'protection-after.json',rec)
 assert all(w['head_unchanged'] for w in workspaces)
 print(json.dumps({k:v for k,v in rec.items() if k not in ['workspaces','fit_ledger_sha256']},ensure_ascii=False),flush=True)

if __name__=='__main__':
 if not (OUT/'runtime-after.json').exists():runtime()
 protection()
