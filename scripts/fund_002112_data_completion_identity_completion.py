"""补核报告标题等价表达；目录与正文矛盾时记录真实文书性质，不按目录猜事实。"""
import re,sys
from collections import Counter
from scripts.fund_002112_data_completion_v1 import OUT,read,save,sha,now
sys.path.insert(0,str(OUT/'ocr-deps'))
from opencc import OpenCC
from scripts.fund_002112_data_completion_identity_final import norm

def normalized(s,cc):
    s=norm(cc.convert(s))
    digits={'〇':'0','○':'0','零':'0','一':'1','二':'2','三':'3','四':'4','五':'5','六':'6','七':'7','八':'8','九':'9'}
    s=re.sub(r'[二一][〇○零一二三四五六七八九]{3}(?=年)',lambda m:''.join(digits[x] for x in m[0]),s)
    return s.replace('上半年度','半年度').replace('上半年','半年度').replace('中期报告','半年度报告').replace('报告全文','报告').replace('(摘要)','摘要')

def run():
    cc=OpenCC('t2s');profiles=read(OUT/'historical-issuer-name-profiles.json');rows=[]
    for id in read(OUT/'historical-identity-additional-result.json')['unresolved']:
        a=read(OUT/'historical-audit'/f'{id}.json');d=read(a['source']);row=d['row'];scan=OUT/'historical-ocr'/f'{id}.json'
        pages=[normalized(t,cc) for t in (read(scan)['pages'] if scan.exists() else d.get('pages',[]))[:20]]
        title=normalized(row['title_plain'],cc);code=row['secCode'];anchors=[]
        for n,t in enumerate(pages):
            for m in re.finditer(r'(?:证券|股票|公司)代码[:：a-z\u4e00-\u9fff(),.]{0,60}'+code+r'(?!\d)',t):anchors.append({'page':n+1,'quote':m[0]})
        legal=[]
        for pr in profiles.get(code,[]):
            if pr['published_date']>row['published_date']:continue
            for name in pr['names']:
                if any(normalized(name['name'],cc) in t for t in pages[:8]):legal.append({'profile_source':pr['source'],'profile_sha256':pr['raw_sha256'],'name':name['name']});break
            if legal:break
        titleanchors=[];year=re.search(r'20\d{2}',title)
        if year and '报告' in title and not re.search('审计|确认|意见|董事|监事|专项',title):
            period='半年度' if '半年度' in title else '第一季度' if '一季度' in title else '第三季度' if '三季度' in title else '年度'
            terms={'半年度':['半年度报告','interimreport','semiannualreport','semi-annualreport'],
                   '第一季度':['第一季度报告','一季度报告','firstquarter','reportofq1'],
                   '第三季度':['第三季度报告','三季度报告','thirdquarter','reportofq3'],
                   '年度':['年度报告','annualreport']}[period]
            for n,t in enumerate(pages[:8]):
                if year[0] in t and any(w in t for w in terms):
                    if '摘要' in title and not any(w in t for w in ['摘要','summary']):continue
                    titleanchors.append({'page':n+1,'quote':t[:850],'year':year[0],'period':period});break
        canonical=re.sub(r'^h股公告[-:：]?','',title)
        if '关于' in canonical:canonical=canonical[canonical.index('关于'):]
        for n,t in enumerate(pages[:8]):
            if len(canonical)>6 and canonical in t:titleanchors.append({'page':n+1,'quote':canonical})
        passed=bool(titleanchors and (anchors or legal))
        rec={'id':id,'at':now(),'identity_verified':passed,'source':a['source'],'raw_sha256':d.get('receipt',{}).get('sha256'),
             'code_anchors':anchors,'legal_entity':legal,'title_anchors':titleanchors,'gaps':[],'training_eligible':False}
        if not (anchors or legal):rec['gaps'].append('BODY_CODE_OR_PRIOR_LEGAL_NAME_NOT_BOUND')
        if not titleanchors:rec['gaps'].append('CATALOG_BODY_DOCUMENT_TITLE_OR_PERIOD_DIFFERENT')
        save(OUT/'historical-identity-completion'/f'{id}.json',rec);rows.append(rec)
    save(OUT/'historical-identity-completion-result.json',{'at':now(),'reviewed':len(rows),'newly_verified':sum(r['identity_verified'] for r in rows),
        'unresolved':[r['id'] for r in rows if not r['identity_verified']],'gap_counts':dict(Counter(g for r in rows for g in r['gaps']))})

if __name__=='__main__':run()
