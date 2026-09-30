"""缺失的 46 份原件验收：严格公告绑定与第三方文书真实对象核验。"""
import re
from datetime import datetime,timezone,timedelta
from scripts.fund_002112_data_completion_v1 import OUT,ROOT,OLD,read,save,sha,now
from pathlib import Path
from app.services.fund_materials_store import source_path

NAMES={'688001':'苏州华兴源创科技股份有限公司','688122':'西部超导材料科技股份有限公司',
       '603260':'合盛硅业股份有限公司','601899':'紫金矿业集团股份有限公司','301281':'山东科源制药股份有限公司'}
def run():
    results=[];historical_catalogs={}
    wanted={p.stem for p in (OUT/'historical-identity-v2').glob('*.json')}
    catalogs=list((OLD/'catalogs').glob('*.json'))+list((ROOT/'information-research').glob('*/catalogs/*.json'))+list((ROOT/'information-research').glob('*/company-catalogs/*.json'))
    for cp in catalogs:
        cat=read(cp)
        if not isinstance(cat,dict):continue
        for row in cat.get('rows',[]):
            if str(row.get('announcementId')) in wanted:historical_catalogs[str(row['announcementId'])]=(cp,cat)
    for p in sorted((OUT/'historical-identity-v2').glob('*.json')):
        completed=OUT/'historical-missing-audit-v3'/p.name
        if completed.exists():results.append(read(completed));continue
        d=read(p);src=read(d['source']);row=src['row'];code=row['secCode']
        if d['id'] in historical_catalogs:catpath,cat=historical_catalogs[d['id']]
        else:
            catpath=source_path(ROOT,'supplement/company-announcements/'+code+('.SH' if code.startswith('6') else '.SZ')+'.json');cat=read(catpath)
        matches=[]
        for r in cat['receipts']:
            raw=Path(r['path']) if r.get('path') else source_path(ROOT,r['file']);assert sha(raw)==r['sha256']
            for item in read(raw).get('announcements',[]):
                if str(item.get('announcementId'))==d['id'] and item.get('secCode')==code and item.get('adjunctUrl')==row['adjunctUrl'] and item.get('announcementTime')==row['announcementTime']:
                    matches.append({'catalog_receipt':r,'row':item})
        declared=datetime.fromtimestamp(row['announcementTime']/1000,timezone(timedelta(hours=8))).date().isoformat()
        name=NAMES.get(code);anchors=[]
        scan=OUT/'historical-ocr'/p.name
        pages=read(scan)['pages'] if scan.exists() else d['pages']
        for n,t in enumerate(pages[:20],1):
            t=re.sub(r'\s+','',t)
            if name and name in t:
                offset=t.find(name);anchors.append({'page':n,'literal':name,'quote':t[max(0,offset-60):offset+len(name)+170]})
        identity=bool(d['passed'] or anchors and d['title_anchors'])
        gaps=[]
        if not matches:gaps.append('EXACT_RAW_CATALOG_BINDING_MISSING')
        if declared!=row['published_date']:gaps.append('RAW_CATALOG_DAY_MISMATCH')
        if not identity:gaps.append('ORIGINAL_LEGAL_ENTITY_OR_DOCUMENT_PURPOSE_UNVERIFIED')
        if src['revision_issues']:gaps.append('PDF_VERSION_LATER_THAN_CATALOG')
        result={'at':now(),'id':d['id'],'source':d['source'],'raw_sha256':d['raw_sha256'],
            'original_sha256_verified':sha(src['receipt']['path'])==d['raw_sha256'],
            'catalog_source':str(catpath),'catalog_sha256':sha(catpath),'raw_catalog_matches':matches,
            'identity_verified':identity,'previous_identity_evidence':str(p),'additional_legal_entity_anchors':anchors,
            'document_kind':'THIRD_PARTY_AUDIT_OF_NAMED_ISSUER' if '审计报告' in row['title_plain'] else 'AUDITOR_REGULATORY_REPLY' if '专项说明' in row['title_plain'] else 'ISSUER_REPORT',
            'published_date':declared,'pdf_metadata':src['metadata'],'pdf_revision_issues':src['revision_issues'],
            'source_declared_historical_publication_verified':not gaps,'first_seen_archive_proven':False,
            'original_acquisition_stop_preserved':src.get('old_stop'),'gaps':gaps,'full_semantics_verified':False,
            'ocr_source':str(scan) if scan.exists() else None,'training_eligible':False}
        save(OUT/'historical-missing-audit-v3'/p.name,result);results.append(result)
    save(OUT/'historical-missing-audit-result-v3.json',{'at':now(),'scope':46,'rows':len(results),
        'source_identity_declared_time_verified':sum(x['source_declared_historical_publication_verified'] for x in results),
        'unresolved':[{'id':x['id'],'gaps':x['gaps']} for x in results if x['gaps']],
        'body_font_recovery_and_full_semantics_independently_required':True})

if __name__=='__main__':run()
