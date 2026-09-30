"""从定期报告原始行和有序表头提取归母利润，保留比较列及修订版本。

列布局只接受有明确表头的已知模板。金额列遇缺字符或百分比、追溯调整列、
无法核对的单位立即留缺口，不移动单元格凑数。所有事实仍不自动训练准入。
"""
import json
import re
import sys
from collections import Counter,defaultdict
from decimal import Decimal,InvalidOperation
from scripts.fund_002112_data_completion_v1 import OUT,OLD,read,save,sha,now
from scripts.fund_002112_data_completion_semantics_v1 import doc_for
from scripts.fund_002112_data_completion_facts_v1 import purpose

CELL=r'[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?'
LABEL=r'归\s*属\s*于\s*(?:上\s*市\s*公\s*司|母\s*公\s*司)\s*(?:股\s*东|所\s*有\s*者)\s*(?:的\s*)?净\s*利\s*润'
ROW=re.compile(LABEL+r'\s*(?:[（(]\s*(?P<unit>百万元|千元|万元|亿元|元)\s*[）)])?\s*(?P<cells>'+CELL+r'(?:\s+'+CELL+r'){1,7})(?![\d.%])')
def compact(s):return re.sub(r'\s+','',s)
def amount(s):return Decimal(s.replace(',','').replace('−','-'))

def parse(pages,title):
    year=re.search(r'20\d{2}',title)
    if not year:
        digits={'零':'0','〇':'0','Ｏ':'0','○':'0','一':'1','二':'2','三':'3','四':'4','五':'5','六':'6','七':'7','八':'8','九':'9'}
        # 标题明确写出的汉字年份是字形转换，不从其他报表日期推定报告年。
        normalized_title=re.sub(r'[二〇零一三四五六七八九Ｏ○]{4}(?=年)',lambda m:''.join(digits[c] for c in m[0]),title)
        year=re.search(r'20\d{2}',normalized_title)
    if not year:return [],['REPORT_YEAR_MISSING']
    year=year[0];rows=[];gaps=[]
    annual='年度报告' in title and '半年度' not in title
    half='半年度' in title
    q1='一季度' in title or '第一季度' in title
    q3='三季度' in title or '第三季度' in title
    if not any([annual,half,q1,q3]):return [],['UNSUPPORTED_REPORT_PERIOD']
    for num,page in enumerate(pages[:25],1):
        for match in ROW.finditer(page):
            before=page[max(0,match.start()-1400):match.start()];header=compact(before).replace('％','%')
            if not any(s in header for s in ['主要会计','主要财务','上年同期','本年比上年','本报告期比上年']):continue
            cells=re.findall(CELL,match['cells'])
            adjusted=any(x in header for x in ['调整前','调整后','重述前','重述后'])
            unit=match['unit'];unitproof=match[0] if unit else None
            if unit is None:
                u=re.findall(r'单位[：:]?(?:人民币)?(百万元|千元|万元|亿元|元)(?:币种[：:]?人民币)?',header)
                u+=re.findall(r'以人民币(百万元|千元|万元|亿元|元)(?:列示|列报)',header)
                u+=re.findall(r'[（(]人民币(百万元|千元|万元|亿元|元)[）)]',header)
                if len(set(u))==1:unit=u[0];unitproof=before
            if unit is None:
                gaps.append({'page':num,'reason':'MONEY_UNIT_NOT_PROVEN','row':match[0]});continue
            columns=None
            percent_header=bool(re.search(r'(?:增减|变动|增减变动)(?:幅度|比例)?[（(]%[）)]',header))
            # 上交所常把百分号放在表头，数字单元格不重复写；保留原数字和表头单位。
            third_percent=len(cells)>=3 and (cells[2].endswith('%') or percent_header)
            if '报告期分季度' in header or ('第一季度' in header[-450:] and '第二季度' in header[-450:]):
                gaps.append({'page':num,'reason':'QUARTER_BREAKOUT_NOT_ANNUAL_COMPARISON','row':match[0]});continue
            raw_orders=re.findall(r'(?:调整|重述)[前后]',header)
            orders=[v.replace('重述','调整') for v in raw_orders]
            # 同比列常另标“调整后”，不把该注记误认为一个金额比较列。
            if len(orders)==3 and orders==['调整前','调整后','调整后']:orders=orders[:2]
            elif len(orders)==5 and orders==['调整前','调整后','调整后','调整前','调整后']:orders=orders[:2]+orders[3:]
            yearly_header=bool(re.search(year+r'年'+str(int(year)-1)+r'年',header))
            cumulative_header='年初至报告期末' in header and '上年初至上年报告期末' in header
            if adjusted and annual and yearly_header and len(cells)==6 and len(orders)>=4 and orders[-4:-2]==orders[-2:] and set(orders[-2:])=={'调整前','调整后'} and (cells[3].endswith('%') or percent_header):
                order=orders[-2:]
                columns=[{'period':year+'-FY','kind':'AMOUNT'}]+[{'period':str(int(year)-1)+'-FY','kind':'AMOUNT','adjustment':v} for v in order]+[{'kind':'YOY_PERCENT'}]+[{'period':str(int(year)-2)+'-FY','kind':'AMOUNT','adjustment':v} for v in order]
            elif adjusted and (half or q1 or q3) and len(cells)==4 and len(orders)>=2 and set(orders[-2:])=={'调整前','调整后'} and (cells[3].endswith('%') or percent_header) and (cumulative_header or '本报告期' in header and '上年同期' in header and not q3):
                period='YTD9' if q3 else 'H1' if half else 'Q1'
                columns=[{'period':year+'-'+period,'kind':'AMOUNT'}]+[{'period':str(int(year)-1)+'-'+period,'kind':'AMOUNT','adjustment':v} for v in orders[-2:]]+[{'kind':'YOY_PERCENT'}]
            elif adjusted and q3 and len(cells)==8 and '本报告期' in header and '年初至报告期末' in header and len(orders)>=4 and orders[-4:-2]==orders[-2:] and set(orders[-2:])=={'调整前','调整后'} and cells[3].endswith('%') and cells[7].endswith('%'):
                columns=[]
                for period in ['Q3','YTD9']:
                    columns += [{'period':year+'-'+period,'kind':'AMOUNT'}]+[{'period':str(int(year)-1)+'-'+period,'kind':'AMOUNT','adjustment':v} for v in orders[-2:]]+[{'kind':'YOY_PERCENT'}]
            elif adjusted:
                gaps.append({'page':num,'reason':'ADJUSTMENT_COLUMNS_REQUIRE_EXPLICIT_LAYOUT','row':match[0]});continue
            elif annual and yearly_header and len(cells) in (3,4) and third_percent:
                columns=[{'period':year+'-FY','kind':'AMOUNT'},{'period':str(int(year)-1)+'-FY','kind':'AMOUNT'},{'kind':'YOY_PERCENT'}]
                if len(cells)==4:columns.append({'period':str(int(year)-2)+'-FY','kind':'AMOUNT'})
            elif (half or q1) and ('本报告期' in header and '上年同期' in header or '年初至报告期末' in header and '上年初至上年报告期末' in header) and len(cells)==3 and third_percent:
                period='H1' if half else 'Q1';columns=[{'period':year+'-'+period,'kind':'AMOUNT'},{'period':str(int(year)-1)+'-'+period,'kind':'AMOUNT'},{'kind':'YOY_PERCENT'}]
            elif (half or q1) and '本报告期' in header and re.search(r'本报告期比上年同期增减(?:变动)?(?:幅度)?',header) and len(cells)==2 and (cells[1].endswith('%') or percent_header):
                # 新版摘要仅列本期和同比，不反推没有披露的上期金额。
                period='H1' if half else 'Q1';columns=[{'period':year+'-'+period,'kind':'AMOUNT'},{'kind':'YOY_PERCENT'}]
            elif q3 and '本报告期' in header and '年初至报告期末' in header and len(cells)==4 and cells[1].endswith('%') and cells[3].endswith('%'):
                columns=[{'period':year+'-Q3','kind':'AMOUNT'},{'kind':'YOY_PERCENT'},{'period':year+'-YTD9','kind':'AMOUNT'},{'kind':'YOY_PERCENT'}]
            elif q3 and len(cells)==4 and percent_header and (('本报告期' in header and '年初至报告期末' in header) or re.search(year+r'年7至9月.*'+year+r'年1至9月',header)):
                columns=[{'period':year+'-Q3','kind':'AMOUNT'},{'kind':'YOY_PERCENT'},{'period':year+'-YTD9','kind':'AMOUNT'},{'kind':'YOY_PERCENT'}]
            elif q3 and '年初至报告期末' in header and '上年初至上年报告期末' in header and len(cells)==3 and third_percent:
                columns=[{'period':year+'-YTD9','kind':'AMOUNT'},{'period':str(int(year)-1)+'-YTD9','kind':'AMOUNT'},{'kind':'YOY_PERCENT'}]
            if columns is None or any(cells[i].endswith('%') for i,c in enumerate(columns) if c['kind']=='AMOUNT'):
                gaps.append({'page':num,'reason':'COLUMN_HEADER_OR_CARDINALITY_UNRESOLVED','row':match[0],'header':before});continue
            values=[{'original':v,'column':columns[i]} for i,v in enumerate(cells)]
            # 百分比只用于独立算术检查，绝不反推原文未披露的比较金额。
            check=None
            if len(cells)>=3 and columns[1]['kind']=='AMOUNT':
                prior_index=next((i for i,c in enumerate(columns) if c.get('adjustment')=='调整后'),1)
                percent_index=next(i for i,c in enumerate(columns) if c['kind']=='YOY_PERCENT')
                if amount(cells[prior_index])==0:
                    gaps.append({'page':num,'reason':'ZERO_COMPARATIVE_AMOUNT_PERCENT_NOT_RECONCILED','row':match[0]});continue
                computed=(amount(cells[0])-amount(cells[prior_index]))/abs(amount(cells[prior_index]))*100
                check={'computed':str(computed),'reported':cells[percent_index],'tolerance_percentage_points':'0.015',
                    'prior_column':prior_index,'percentage_column':percent_index,
                    'passed':abs(computed-amount(cells[percent_index].removesuffix('%')))<=Decimal('.015')}
                if not check['passed']:
                    gaps.append({'page':num,'reason':'PERCENTAGE_RECONCILIATION_FAILED','row':match[0],'check':check});continue
            currency='CNY' if '人民币' in header else None
            cumulative_check=None
            if len(cells)==8 and q3:
                prior_index=next(i for i in (5,6) if columns[i].get('adjustment')=='调整后')
                if amount(cells[prior_index])==0:
                    gaps.append({'page':num,'reason':'ZERO_COMPARATIVE_AMOUNT_PERCENT_NOT_RECONCILED','row':match[0]});continue
                computed=(amount(cells[4])-amount(cells[prior_index]))/abs(amount(cells[prior_index]))*100
                cumulative_check={'computed':str(computed),'reported':cells[7],'tolerance_percentage_points':'0.015',
                    'prior_column':prior_index,'percentage_column':7,'passed':abs(computed-amount(cells[7].removesuffix('%')))<=Decimal('.015')}
                if not cumulative_check['passed']:
                    gaps.append({'page':num,'reason':'CUMULATIVE_PERCENTAGE_RECONCILIATION_FAILED','row':match[0],'check':cumulative_check});continue
            rows.append({'metric':'NET_PROFIT_ATTRIBUTABLE_TO_PARENT','unit':unit,'currency':currency,'cells':values,
                'page':num,'raw_row':match[0],'raw_header':before,'row_offset':match.start(),
                'unit_proof':unitproof,'arithmetic_check':check,'percentage_unit_from_header':percent_header,'reported_period_basis':'REPORT_TITLE_AND_EXPLICIT_TABLE_HEADER',
                'cumulative_arithmetic_check':cumulative_check,
                'literal_and_template_verified':True,'visual_layout_verified':False,
                'comparative_previous_publication_verified':False,'accounting_standard':None,
                'full_semantics_verified':False})
    return rows,gaps

def run():
    version=sys.argv[1] if len(sys.argv)>1 else ''
    suffix='-'+version if version else ''
    allrows=[]
    for n,r in enumerate(read(OLD/'admission-audit-v3/20260929-172820.json')['rows'],1):
        if r['category']!='PERFORMANCE':continue
        dest=OUT/('financial-table-facts'+suffix)/f"{r['document_id']}.json"
        if dest.exists():allrows.append(read(dest));continue
        d,p=doc_for(r);title=d['row']['title_plain'];kind=purpose(title,r['category'])
        if kind not in ('FINANCIAL_REPORT','FINANCIAL_REPORT_TRANSLATION'):continue
        scan=OUT/'historical-ocr'/f"{r['document_id']}.json"
        if scan.exists():pages=read(scan)['pages'];textsource=str(scan)
        else:pages=d.get('pages',[]);textsource=str(p)
        facts,gaps=parse(pages,title)
        rec={'id':r['document_id'],'company':r['company'],'title':title,'published_date':r['published_date'],
            'source':str(p),'source_sha256':sha(p),'raw_sha256':d.get('receipt',{}).get('sha256'),
            'text_source':textsource,'facts':facts,'gaps':gaps,'revision_issues':r['revision_issues_preserved'],
            'training_eligible':False,'at':now()}
        save(dest,rec);allrows.append(rec)
        if len(allrows)%100==0:print(json.dumps({'financial_docs':len(allrows)}),flush=True)
    save(OUT/('financial-table-result'+suffix+'.json'),{'at':now(),'reviewed_reports':len(allrows),'reports_with_table_facts':sum(bool(x['facts']) for x in allrows),
        'table_facts':sum(len(x['facts']) for x in allrows),'all_report_failures_retained':True,'no_automatic_training_admission':True})

if __name__=='__main__':run()
