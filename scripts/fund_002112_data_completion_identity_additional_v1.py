"""处理正文省略版本后缀、繁体及英文载体，身份通过不等于状态或版本通过。"""
import re
import sys
from collections import defaultdict
from scripts.fund_002112_data_completion_v1 import OUT,read,save,sha,now
sys.path.insert(0,str(OUT/'ocr-deps'))
from opencc import OpenCC
from scripts.fund_002112_data_completion_identity_final import norm

def run():
    cc=OpenCC('t2s');profiles=defaultdict(list)
    unresolved=set(read(OUT/'historical-identity-final-result.json')['unresolved'])
    wanted={read(OUT/'historical-audit'/f'{i}.json')['company'] for i in unresolved}
    for p in (OUT/'historical-audit').glob('*.json'):
        a=read(p)
        if a['company'] not in wanted or not read(OUT/'historical-identity-final'/p.name)['identity_verified']:continue
        d=read(a['source']);namehits=[]
        for n,t in enumerate(d.get('pages',[])[:5],1):
            for line in t.splitlines():
                line=norm(cc.convert(line))
                m=re.fullmatch(r'([\u4e00-\u9fff]{3,35}(?:股份有限公司|有限责任公司))(?:20\d{2}年.*)?',line)
                if m:namehits.append({'name':m[1],'page':n,'quote':line})
        if namehits:profiles[a['company']].append({'id':a['id'],'source':a['source'],'raw_sha256':d['receipt']['sha256'],'published_date':a['published_date'],'names':namehits})
    save(OUT/'historical-issuer-name-profiles.json',dict(profiles));results=[]
    for id in unresolved:
        a=read(OUT/'historical-audit'/f'{id}.json');d=read(a['source']);row=d['row'];title=norm(cc.convert(row['title_plain']));code=row['secCode']
        pages=[norm(cc.convert(t)) for t in d.get('pages',[])[:20]];canonical=title
        annotations=re.findall(r'\((?:已取消|已废止|修订版|修订稿|修订后|修订|更新后|更新|更正后)\)',canonical)
        canonical=re.sub(r'\((?:已取消|已废止|修订版|修订稿|修订后|修订|更新后|更新|更正后)\)','',canonical)
        canonical=re.sub(r'^h股公告[-:：]?','',canonical)
        if '关于' in canonical and '公告' in canonical:canonical=canonical[canonical.index('关于'):]
        if row.get('secName') and canonical.startswith(norm(row['secName'])+':'):canonical=canonical[len(norm(row['secName']))+1:]
        # 证券代码表头和其行值允许相隔其他列标题，但证券代码须逐字一致。
        codehits=[]
        for n,t in enumerate(pages):
            for m in re.finditer(r'(?:证券代码|股票代码|stockcode|securitiescode|securitycode)[:：a-z\u4e00-\u9fff(),.]{0,70}'+code+r'(?!\d)',t):codehits.append({'page':n+1,'quote':m[0]})
        titlehits=[{'page':n+1,'quote':canonical} for n,t in enumerate(pages[:6]) if len(canonical)>6 and canonical in t]
        if not titlehits and '英文' in title:
            year=re.search(r'20\d{2}',title)
            terms=['semi-annualreport','semiannualreport','januarytojune','interimreport'] if '半年度' in title else ['firstquarter','januarytomarch'] if '一季度' in title else ['thirdquarter','januarytoseptember'] if '三季度' in title else ['annualreport']
            titlehits=[{'page':n+1,'quote':t[:650],'explicit_period_terms':terms} for n,t in enumerate(pages[:3]) if year and year[0] in t and any(term in t for term in terms)]
        legal=[]
        for profile in profiles.get(code,[]):
            if profile['published_date']>row['published_date']:continue
            for name in profile['names']:
                if any(name['name'] in t for t in pages[:5]):legal.append({'historical_profile':profile,'matched_name':name['name']});break
            if legal:break
        passed=bool(titlehits and (codehits or legal))
        rec={'id':id,'at':now(),'source':a['source'],'raw_sha256':d['receipt']['sha256'],
            'identity_verified':passed,'code_anchors':codehits,'title_anchors':titlehits,'prior_dated_legal_entity_proof':legal,
            'catalog_only_suffixes':annotations,'suffixes_not_used_as_historical_status':True,
            'source_version_and_full_semantics_not_implied':True}
        save(OUT/'historical-identity-additional'/f'{id}.json',rec);results.append(rec)
    save(OUT/'historical-identity-additional-result.json',{'at':now(),'reviewed':len(results),'newly_verified':sum(r['identity_verified'] for r in results),
        'unresolved':[r['id'] for r in results if not r['identity_verified']]})

if __name__=='__main__':run()
