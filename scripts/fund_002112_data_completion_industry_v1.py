"""复用已授权指数源补 002112 历史行业背景，不将现行业分类倒灌为历史关联。"""
import json
import time
from datetime import date
from decimal import Decimal
from sqlalchemy import create_engine,text
from scripts.fund_002112_data_completion_v1 import PY,ROOT,OLD,OUT,read,save,sha,now
from app.core.config import get_settings
from app.services.direction_market_data import client,metadata

CODES=('000986.SH','000987.SH','000989.SH','000991.SH','000993.SH','399929.SZ','399930.SZ',
       '399931.SZ','399935.SZ','399936.SZ','399937.SZ','399986.SZ','399998.SZ')

def run():
    target=OUT/'industry';target.mkdir(exist_ok=True)
    plan=target/'contract.json'
    if not plan.exists():
        meta=metadata(CODES)
        save(plan,{'at':now(),'codes':CODES,'years':list(range(2016,2024)),'max_requests':104,
            'source':meta,'scope_reason':'有限行业背景池；覆盖已有002112披露持仓涉及的生产、材料、能源、医疗、信息通信、银行及公用事业；历史公司行业归属另行验收',
            'no_broad_market_proxy':True,'no_current_classification_backfill':True,'no_fit':True})
    cache=PY/'.local-runs/direction-1d-sector-data-20260912'
    with client() as api:
        for code in CODES:
            for year in range(2016,2024):
                result_path=target/f'{code}-{year}.json'
                if result_path.exists():continue
                old=cache/f'daily-{code}-{year}.json'
                if old.exists() and read(old).get('status')=='DOWNLOADED':
                    d=read(old)
                    save(result_path,{'code':code,'year':year,'prices':d['prices'],'cache':str(old),'cache_sha256':sha(old),'new_requests':0})
                    continue
                request=target/'requests'/f'{code}-{year}.json'
                if request.exists():continue
                save(request,{'at':now(),'api':'index_daily','code':code,'start':f'{year}0101','end':f'{year}1231'})
                time.sleep(.35)
                d={'at':now(),'code':code,'year':year,'new_requests':1,'prices':[]}
                try:
                    rows=api.list_index_daily(code,start_date=date(year,1,1),end_date=date(year,12,31))
                    if any(r.index_code!=code or r.trade_date.year!=year or r.close_price<=0 for r in rows):raise ValueError('IDENTITY_DATE_VALUE_INVALID')
                    d['prices']=[{'date':str(r.trade_date),'close':str(r.close_price)} for r in rows]
                    d['status']='DOWNLOADED' if rows else 'EMPTY_PRESERVED'
                except Exception as e:
                    d.update(status='FAILED',error_type=type(e).__name__)
                save(result_path,d)
                print(json.dumps({'code':code,'year':year,'rows':len(d['prices']),'status':d['status']}),flush=True)
                if d['status']=='FAILED':break
    # 只读取交易日，不读取封存的价格答案或标签。
    from app.services.trading_calendar import load_calendar
    sessions={str(d) for d in load_calendar().sessions if date(2016,1,1)<=d<=date(2023,12,31)}
    summary=[]
    for code in CODES:
        allrows=[];sources=[]
        for p in target.glob(code+'-*.json'):
            d=read(p);allrows.extend(d['prices']);sources.append({'path':str(p),'sha256':sha(p)})
        dates=[r['date'] for r in allrows]
        summary.append({'code':code,'rows':len(allrows),'unique':len(set(dates)),'first':min(dates,default=None),
            'last':max(dates,default=None),'missing_dates':sorted(sessions-set(dates)),
            'noncalendar_dates':sorted(set(dates)-sessions),'sources':sources,'historical_company_classification_verified':False})
    save(target/'coverage.json',{'at':now(),'indices':summary,'new_requests':len(list((target/'requests').glob('*.json'))),
        'training_eligible':False,'reason':'行业行情独立验收；历史成分/分类及披露持仓关联未完成前不能生成行业权重输入'})

if __name__=='__main__':run()
