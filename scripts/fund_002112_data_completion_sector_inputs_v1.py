"""为独立行业背景假设生成无标签输入，严格使用目标日前已收盘的数据。

行业指数自身是研究输入，不冒充基金的行业持仓组合；不读取收益答案、
封存标签或进行任何拟合，停发指数保留原缺口，绝不延长或补零。
"""
from bisect import bisect_left
from decimal import Decimal
from scripts.fund_002112_data_completion_v1 import OUT,read,save,sha,now

def run():
    coverage=read(OUT/'industry/coverage-validated-v2.json')['indices']
    catalog=read(OUT/'industry/contract.json')['source']['catalog'];series={};sources=[]
    for item in coverage:
        code=item['code'];prices={}
        for src in item['sources']:
            assert sha(src['path'])==src['sha256']
            for r in read(src['path'])['prices']:
                assert r['date'] not in prices
                prices[r['date']]=Decimal(r['close'])
            sources.append(src)
        dates=sorted(prices);assert all(v>0 for v in prices.values())
        series[code]=(dates,prices)
    targets=[r['target'] for r in read(OUT/'historical-fund-relations.json')['rows']]
    complete=[c for c,(days,_) in series.items() if len(days)==1945 and days[-1]=='2023-12-29' and catalog[c]['list_date']<'2016-01-01']
    assert len(complete)==8
    rows=[]
    for target in targets:
        facts={}
        for code,(days,prices) in series.items():
            ix=bisect_left(days,target)-1;expiry=catalog[code]['expiry_date']
            if ix<20 or (expiry and target>expiry):
                facts[code]={'values':None,'reason':'INDEX_EXPIRED' if expiry and target>expiry else 'INSUFFICIENT_PAST_HISTORY'};continue
            used=days[ix];assert used<target
            values={'close':str(prices[used]),'return_1d':str(prices[used]/prices[days[ix-1]]-1),
                    'return_5d':str(prices[used]/prices[days[ix-5]]-1),'return_20d':str(prices[used]/prices[days[ix-20]]-1)}
            facts[code]={'values':values,'last_price_date':used,'lookback_start':days[ix-20],
                'currency_unit':'INDEX_POINTS','return_unit':'DECIMAL_RATIO','fund_industry_weight_applied':False}
        rows.append({'fund_code':'002112','target_date':target,'as_of':target+'T08:00:00+08:00','indices':facts,
            'eight_series_complete':all(facts[c]['values'] is not None for c in complete)})
    assert len(rows)==1422 and all(r['eight_series_complete'] for r in rows)
    data=OUT/'independent-sector-inputs.json'
    save(data,{'at':now(),'rows':rows,'catalog':catalog,'sources':sources,'complete_universe':complete,
        'hypothesis_scope':'INDEPENDENT_SECTOR_BACKGROUND_WITHOUT_FUND_INDUSTRY_WEIGHTS',
        'historical_source_vintages_unavailable':True,'data_source_revision_risk':'Historical index closes obtained now; publisher methodology or vendor correction history is not reconstructed',
        'no_target_or_label_fields':True,'fits':0})
    save(OUT/'independent-sector-input-acceptance.json',{'at':now(),'file':str(data),'sha256':sha(data),'rows':1422,'complete_series':complete,
        'source_dates_strictly_before_target':True,'no_imputation':True,'original_expired_series_retained':True,
        'input_arithmetic_and_coverage_verified':True,'fit_authorized_this_turn':False,
        'historical_vintage_strict_gate_passed':False,'reason':'VENDOR_HISTORICAL_RESTATING_AND_METHODOLOGY_VINTAGES_NOT_PROVEN',
        'fund_weighted_industry_hypothesis_ready':False})

if __name__=='__main__':run()
