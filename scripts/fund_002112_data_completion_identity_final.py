"""全清单身份补核：精确正文证券代码及等价标题，不做模糊名称匹配。"""
import re
import unicodedata
from scripts.fund_002112_data_completion_v1 import OUT,read,save,sha,now

def norm(s):return re.sub(r'\s+','',unicodedata.normalize('NFKC',s)).casefold()

def run():
    results=[]
    for p in (OUT/'historical-audit').glob('*.json'):
        old=read(p);passed=old['identity_verified'];anchors=[];extra=None
        fixed=OUT/'historical-missing-audit-v4'/p.name
        if fixed.exists():passed=read(fixed)['identity_verified'];extra=str(fixed)
        if not passed:
            d=read(old['source']);row=d['row'];code=row['secCode'];title=norm(row['title_plain']);pages=[norm(t) for t in d.get('pages',[])[:6]]
            canonical=title
            short=norm(row.get('secName') or '')
            if short and canonical.startswith(short+':'):canonical=canonical[len(short)+1:]
            # 公告标题以发行人名称开头时，明确的“关于…”文书名仍须在正文完整出现。
            if '关于' in canonical and canonical.index('关于')>0 and '公告' in canonical:
                canonical=canonical[canonical.index('关于'):]
            codehits=[{'page':n+1,'quote':m[0]} for n,t in enumerate(pages) for m in re.finditer(r'(?:证券|股票)代码[:：]?'+re.escape(code)+r'(?!\d)',t)]
            titlehits=[{'page':n+1,'quote':canonical} for n,t in enumerate(pages) if len(canonical)>=7 and canonical in t]
            if codehits and titlehits:passed=True;anchors=codehits+titlehits
        rec={'id':old['id'],'company':old['company'],'prior_identity_verified':old['identity_verified'],
            'identity_verified':bool(passed),'prior_evidence':str(p),'prior_evidence_sha256':sha(p),
            'missing_original_audit':extra,'new_exact_body_anchors':anchors,'full_semantics_implied':False,'at':now()}
        save(OUT/'historical-identity-final'/p.name,rec);results.append(rec)
    save(OUT/'historical-identity-final-result.json',{'at':now(),'scope':len(results),'identity_verified':sum(r['identity_verified'] for r in results),
        'newly_verified':sum(r['identity_verified'] and not r['prior_identity_verified'] for r in results),
        'unresolved':[r['id'] for r in results if not r['identity_verified']]})

if __name__=='__main__':run()
