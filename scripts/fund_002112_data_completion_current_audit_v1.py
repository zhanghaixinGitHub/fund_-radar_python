"""当前 55 份公告逐项核主体、文书性质和时点；与历史研究输入彻底分开。"""
import re
import unicodedata
from collections import Counter
from datetime import datetime,timedelta
from scripts.fund_002112_data_completion_v1 import OUT,ROOT,read,save,sha,now
from app.services.fund_materials_store import source_path
from app.services.fund_information_history_v1 import pdf_revision

def norm(s):return re.sub(r'\s+','',unicodedata.normalize('NFKC',s)).casefold()

def anchors(pages,patterns,limit=10):
    rows=[]
    for i,p in enumerate(pages,1):
        t=norm(p)
        for pattern in patterns:
            for m in re.finditer(pattern,t):
                rows.append({'page':i,'offset':m.start(),'literal':m[0],'quote':t[max(0,m.start()-70):m.end()+200]})
                if len(rows)>=limit:return rows
    return rows

def classify(d):
    title=d['title'];key=d['id']
    if '翌日披露' in title:return 'SHARE_CHANGE_NEXT_DAY',['翌日披露报表','变动日期','呈交日期'],'股份类别及事件逐项保留，表名本身不证明发生回购'
    if '月報表' in title or '月报表' in title:return 'SHARE_CHANGE_MONTHLY',['证券变动月报表','截至月份','呈交日期'],'月末股本记录，不等于营业业绩'
    if '中期股息' in title:return 'INTERIM_DIVIDEND_UPDATE' if '更新' in title else 'INTERIM_DIVIDEND',['中期股息','股息派发日','每股'],'派息安排及更新，个人税率/资格未知'
    if '法律意见书' in title:return 'LEGAL_OPINION',['法律意见书','综上','综上所述'],'法律意见的发表主体是律所，意见对象是上市公司，不充当公司业绩'
    if '核查意见' in title:return 'INCENTIVE_ELIGIBILITY_REVIEW',['核查意见','公示结果','公示时间'],'只保留公司公开核查结论，不推断个人账户资格'
    if '草案' in title:return 'CONDITIONAL_GOVERNANCE_DRAFT',['生效','上市','本政策|本制度|本规范|章程'],'保留草案和H股发行上市后适用条件，不当作已生效'
    if '章程' in title:return 'ARTICLES_OF_ASSOCIATION',['章程','生效'],'章程文本，生效条件须按正文保留'
    if '权益变动报告书' in title:return 'HOLDER_INTEREST_CHANGE',['简式权益变动报告书','被动稀释','5%以下'],'持有人权益变化，不作公司回购事实'
    if '业绩说明会' in title:return 'INVESTOR_MEETING_NOTICE',['业绩说明会','网上','时间'],'会议安排，不是盈利预告'
    if '验资报告' in title:return 'CAPITAL_VERIFICATION',['验资报告','认购','募集资金'],'认购资金验资，不视为营业收入'
    if '发行情况报告书' in title:return 'PLACEMENT_REPORT',['发行情况报告书','发行价格','发行数量'],'发行事项，不视为经营业绩'
    if '鉴证报告' in title:return 'FUND_REPLACEMENT_ASSURANCE',['鉴证报告','自筹资金','募集资金'],'资金置换鉴证，不视为新增盈利'
    if key=='1225580470':return 'ACQUISITION_TARGET_SIMULATED_AUDIT',['苏州安捷讯光电科技股份有限公司','模拟财务报表','编制基础'],'被收购标的模拟报表，不能作为光库本身的已实现业绩'
    if key=='1225580471':return 'PRO_FORMA_REVIEW',['备考合并财务报表','审阅报告','该交易尚未完成'],'备考口径，不能当成收购已完成或实际合并业绩'
    if '可持续' in title:return 'SUSTAINABILITY_REPORT',['可持续发展报告书|sustainabilityreport','2025'],'ESG报告和英文载体，不计为独立盈利事件'
    if '半年度' in title or '中期报告' in title:return 'INTERIM_FINANCIAL_REPORT_TRANSLATION' if '英文' in title else 'HK_INTERIM_FINANCIAL_REPORT',['中期报告|semi-annualreport','2026','截至2026年6月30日|2026年1-6月|june30,2026'],'单独保留会计口径和译文发布日期，不继承早期中文版日期'
    if '会议文件' in title:return 'SHAREHOLDER_MEETING_AGENDA',['会议','议案'],'待审议材料不等于议案已经通过'
    if '换届' in title:return 'BOARD_AND_MANAGEMENT_APPOINTMENT',['换届','聘任'],'组织任免公告'
    return 'UNCLASSIFIED',[], '文书性质未核实'

def run():
    profiles=read(ROOT/'supplement/stock-context.json')['stock_basic'];results=[]
    for p in sorted((OUT/'current').glob('*.json')):
        d=read(p);code=d['stock_code'];stock=next((v for v in profiles.values() if v['symbol']==code),None)
        suffix='SH' if code.startswith('6') else 'SZ'
        catpath=source_path(ROOT,'supplement/company-announcements/'+code+'.'+suffix+'.json');cat=read(catpath)
        entries=[r for r in cat['rows'] if str(r.get('announcementId'))==d['id']]
        binding=bool(len(entries)==1 and entries[0]['secCode']==code and entries[0]['adjunctUrl'] in d['receipt']['url'])
        raw_matches=[]
        for receipt in cat['receipts']:
            raw=source_path(ROOT,receipt['file'])
            if sha(raw)!=receipt['sha256']:raise ValueError('CATALOG_RAW_DIGEST_MISMATCH')
            for row in read(raw).get('announcements',[]):
                if str(row.get('announcementId'))==d['id']:
                    raw_matches.append({'receipt':receipt,'row':row})
        binding=binding and any(all(m['row'].get(k)==entries[0].get(k) for k in ('announcementId','secCode','adjunctUrl','announcementTime')) for m in raw_matches)
        pages=d['normalized_pages'];name=stock['fullname'] if stock else ''
        identity=anchors(pages[:15],[re.escape(norm(name))],3) if name else []
        related=None
        if d['id']=='1225580470':
            other=read(OUT/'current/1225580471.json')
            related={'id':other['id'],'sha256':other['receipt']['sha256'],'anchors':anchors(other['normalized_pages'],['公司基本情况及拟实施的资产重组方案','该交易尚未完成'],4)}
            identity=anchors(pages[:5],['苏州安捷讯光电科技股份有限公司'],3)
        elif not identity and code=='002384':
            identity=anchors(pages[:15],[r'suzhoudongshanprecisionmanufacturingco\.,ltd\.','stockcode:002384','stockcode002384'],4)
        kind,terms,boundary=classify(d);facts=anchors(pages,terms,14)
        kind_anchors=anchors(pages,terms[:1],2)
        revision=pdf_revision(d['metadata'],d['published_at'][:10])
        missing=[]
        if not binding:missing.append('CATALOG_ID_STOCK_OR_ATTACHMENT_MISMATCH')
        if not identity:missing.append('LEGAL_ENTITY_ANCHOR_MISSING')
        if not kind_anchors or kind=='UNCLASSIFIED':missing.append('DOCUMENT_PURPOSE_BODY_ANCHOR_MISSING')
        if revision:missing.append('PDF_VERSION_AFTER_CATALOG_DAY')
        body=any(len(norm(x))>100 for x in pages)
        if not body:missing.append('BODY_NOT_READABLE')
        rec={'id':d['id'],'at':now(),'title':d['title'],'stock_code':code,'source_file':str(p),'source_sha256':sha(p),
            'raw_sha256':d['receipt']['sha256'],'catalog_file':str(catpath),'catalog_sha256':sha(catpath),'catalog_rows':entries,
            'raw_catalog_matches':raw_matches,
            'catalog_binding_verified':binding,'legal_entity':name,'identity_anchors':identity,'related_entity_proof':related,
            'document_kind':kind,'kind_anchors':kind_anchors,'fact_anchors':facts,'business_boundary':boundary,
            'source_announced_at':d['published_at'],'pdf_metadata':d['metadata'],'version_issues':revision,
            'source_publication_verified':not missing,'current_document_nature_verified':bool(identity and kind_anchors),
            'numeric_table_semantics_verified':False,'current_publication_gaps':missing,
            'source_claimed_time_not_first_seen':True,'historical_training_eligible':False,
            'personal_account_conditions':'UNKNOWN','ocr_detail_files':d['ocr_pages']}
        dest=OUT/'current-audit-v2'/p.name
        if not dest.exists():save(dest,rec)
        results.append(rec)
    save(OUT/'current-audit-result-v2.json',{'at':now(),'scope':55,'audited':len(results),
        'source_publication_verified':sum(x['source_publication_verified'] for x in results),
        'current_document_nature_verified':sum(x['current_document_nature_verified'] for x in results),
        'kinds':dict(Counter(x['document_kind'] for x in results)),
        'unresolved':[{'id':x['id'],'gaps':x['current_publication_gaps']} for x in results if not x['source_publication_verified']],
        'all_excluded_from_historical_training':True})

if __name__=='__main__':run()
