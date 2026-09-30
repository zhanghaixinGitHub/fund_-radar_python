"""逐项固定 41 份重大合同的事实边界，后续更正仅从其自身披露时点起可用。

金额只绑定其原句及业务范围；中标、签署、生效、履约、收款分别记录。
合同金额从不当作已实现收入，匿名客户、豁免金额及未满足条件原样保留。
"""
import re
from collections import Counter
from scripts.fund_002112_data_completion_v1 import OUT,read,save,sha,now

# 人工核读本轮保存原文后的语义判定；锚点必须在对应文件中逐字找到。
REVIEW={
'1210198119':('AWARDED_SIGNATURE_UNKNOWN','中标总金额为33,617.148070万元','中标金额，未由此认定已签约或收入。'),
'1210758701':('CUMULATIVE_ORDERS_UNDER_PRIOR_FRAMEWORK','并使合同于2019年8月24日起生效','框架不约定总额；当前披露采购订单，保留原披露及后续更正的各自时点。'),
'1210760613':('CORRECTION_OF_ORDER_SCOPE','2021年3月18日至2021年8月13日','明确系既有框架进展及上市后累计订单6.77亿元，不是新框架总额。'),
'1210900908':('SIGNED_CONDITIONAL','30,186.51万元人民币','项目合同含税额；签字盖章条件与实际履行分开。'),
'1211064813':('SIGNED_ESTIMATED_PROJECT','合同估算总投资45,407.57万元','联合体项目总投资及施工暂估额，不等于发行人单独收入。'),
'1211457521':('SIGNED_ORDER_BASED_SUPPLY','实际采购数量及销售金额以特斯拉发出的采购订单为准','2022至2024供应安排，未披露确定采购总金额。'),
'1212192592':('SIGNED_PRICING_AGREEMENT','未对产品采购量进行保证','匿名海外客户；采购数量和金额依后续订单。'),
'1212592020':('SIGNED_CONDITIONAL','人民币2.15亿元','经销合同含税金额，付款和签章条件保留。'),
'1212625912':('SIGNED_EFFECTIVE_DECLARED','合同虽已正式签署并正式生效','三份合同合计9577.973万元，不再叠加合计与分项。'),
'1212751183':('SIGNED_AMOUNT_EXEMPT','因合同金额涉及商业秘密','合同金额豁免披露是明确事实，不填零；签约生效与履行不同。'),
'1212818315':('SIGNED_CURRENT_GAS_PROJECT','411,935,438.83美元','当前天然气项目；593584975美元属于此前原油项目，不能当本次金额。'),
'1212827348':('CUMULATIVE_ORDERS_UNDER_FRAMEWORK','上述合同未约定合同金额','289023613.25元为订单区间累计，框架没有约定总额。'),
'1212963361':('SIGNED_CONCESSION_INVESTMENT','885,517,600元','联合体特许经营项目总投资，合作期30年，不直接归为发行人销售收入。'),
'1213274495':('SIGNED_LONG_TERM_SUPPLY','约6.55亿欧元','2024至2030供应合同，金额币种欧元。'),
'1213790172':('SIGNED_PRICE_DEPENDENT_ESTIMATE','预计销售总额约385亿元人民币','以当时硅料均价估算的长单，月议订单价，未承诺实现收入。'),
'1213917253':('SIGNED_PRICE_DEPENDENT_ESTIMATE','预计销售总额约644.10亿元人民币','以当时硅料均价估算的长单，未考虑未来市场价格变化。'),
'1213917254':('SIGNED_PRICE_DEPENDENT_ESTIMATE','预计销售总额约560亿元人民币','以当时硅料均价估算的长单，与同日另一合同分开。'),
'1214420164':('PERFORMANCE_DELAY_AND_UNCERTAINTY','整体合同履行进度不达预期','是否继续履行存在不确定性；此日不能提前使用后来终止消息。'),
'1214447343':('SIGNED_ESTIMATED_GAS_SUPPLY','合同金额约人民币20亿元','初步测算含税额，最终按实际结算。'),
'1214450102':('SIGNED_SHAREHOLDER_APPROVAL_PENDING','并在乙方股东大会审议通过变更公司注册地址','当时股东大会尚待召开，不认定已经生效。'),
'1214469404':('TERMINATION_AGREEMENT','签署了《合同终止协议》','解除合同；退款是否已到账需另有证据。'),
'1214532449':('SIGNED_EFFECTIVE_DECLARED','合同虽已正式签署并正式生效','2640万元氯化锂合同，风险和收款条件仍保留。'),
'1214556877':('SIGNED_SALE','5,040.00万元人民币','含税销售合同；对方按原文匿名，不推测名称。'),
'1214578450':('SIGNED_PRICE_DEPENDENT_ESTIMATE','预计销售总额约1,033.56亿元人民币','按当时硅料均价估算，含权益采购量，实际订单月议。'),
'1214602238':('EFFECTIVE_AFTER_APPROVAL_CONFIRMED','按照合同约定，本合同已生效','本次披露确认9月15日股东大会批准；不回写8月30日已生效。'),
'1214642039':('SIGNED_ESTIMATED_GAS_SUPPLY','人民币4.60亿元','合同期初步测算，未含扩建增量，最终实际结算。'),
'1214695279':('TERMINATED_REFUND_OVERDUE_WITH_PROMISE','未能如约向公司退还合作意向金','分期偿还是承诺，不等于实际收款。'),
'1216116225':('SIGNED_CONDITIONAL_SALE','3.9434亿元人民币','含税合同总额，生效与履约条件保留。'),
'1217431128':('CONTRACT_AMOUNT_AMENDED','由3,450万美元调整至3,640万美元','原金额和变更后金额分别保存，从本公告起使用变更。'),
'1217451953':('SIGNED_CONDITIONAL','27,986.32万元人民币','六大管道项目合同含税金额。'),
'1217490068':('SIGNED_WITH_OPTIONAL_VOLUME','客户对其中3GW有选择执行权','约7GW包含客户可选择3GW；不得全额当作无条件确定订单。'),
'1217501279':('COUNTERPARTY_IDENTITY_SUPPLEMENT','客户目前已同意公司对外披露客户信息','后披露EDF-RE US主体，不能倒灌到前一公告匿名客户。'),
'1217619369':('AWARDED_AND_SIGNED_NOT_COMPLETED','并完成合同签订，项目尚在实施过程中','27210万元为中标合计，不等于完成收入。'),
'1217681238':('SIGNED_ESTIMATE_WITH_LATER_BATCH_ORDERS','暂估金额为（含税）120,350,650.00元','第二、三批数量、型号、价格及交付取决于后续订单。'),
'1217858442':('SIGNED_CONDITIONAL','42,332.39万元人民币','EPCO项目含税总额，不当成单年度收入。'),
'1217968318':('SIGNED_EFFECTIVE_DECLARED','合同生效时间：2023年9月27日','约10亿美元造船合同；汇率测算口径与美元原金额分开。'),
'1218004272':('AWARDED_NOT_SIGNED','合同尚未签订','64312.29万元中标合计，当时未签约。'),
'1218042104':('SIGNED_PREPAYMENT_CONDITION','自公司收到预付款之日起生效','35888万元为两合同合计，未取得预付款到账证据。'),
'1218047695':('SIGNED_CONSORTIUM_ALLOCATION_PENDING','根据联合体分工进一步明确双方合同金额','58888.89万元为联合体总额，发行人份额尚待补充协议。'),
'1218048988':('SIGNED_PRICE_BY_ACTUAL_SETTLEMENT','人民币105,827,697.00元','含税合同总额，最终按实际采购结算。'),
'1218523004':('SIGNED_SHIPBUILDING','总金额约14.6亿美元','造船合同原币金额，不能与当年营业收入等同。'),
}

def run():
    rows=[]
    for id,(stage,key,note) in REVIEW.items():
        dest=OUT/'contract-scope-review'/f'{id}.json'
        if dest.exists():rows.append(read(dest));continue
        f=read(OUT/'historical-facts-v2'/f'{id}.json');d=read(f['source']);pages=[re.sub(r'\s+','',t) for t in d['pages']]
        hits=[{'page':n+1,'offset':t.index(key),'quote':t[max(0,t.index(key)-100):t.index(key)+len(key)+180]} for n,t in enumerate(pages) if key in t]
        assert hits,(id,key)
        clauses=[]
        for n,t in enumerate(pages):
            for m in re.finditer(r'[^。]{0,60}(?:合同金额|合同总金额|合同价格|中标总金额|合同价款|预计销售总额|合同生效|甲方[：:]|乙方[：:]|供方[：:]|需方[：:]|签署时间|履行期限|合同签署概况)[^。]{0,350}',t):
                q=m[0];scope='CURRENT_OR_RELATED_CLAUSE'
                if id=='1212818315' and '593,584,975' in q:scope='OTHER_PREVIOUS_CRUDE_OIL_PROJECT'
                elif id=='1217431128' and '3,450' in q:scope='BEFORE_AND_AFTER_AMENDMENT'
                elif '预计' in q or '暂估' in q or '测算' in q:scope='ESTIMATE_NOT_GUARANTEED'
                elif '未约定' in q or '商业秘密' in q:scope='EXPLICITLY_UNDISCLOSED_OR_ORDER_BASED'
                clauses.append({'page':n+1,'quote':q,'amount_scope':scope,'not_realized_revenue':True})
        rec={'id':id,'at':now(),'company':f['company'],'published_date':f['published_date'],'raw_sha256':f['raw_sha256'],'source':f['source'],
             'event_stage':stage,'review_note_zh':note,'stage_anchors':hits,'contract_clauses':clauses,'stage_and_scope_verified':True,
             'all_party_allocations_or_payments_proven':False,'numeric_facts_require_own_units_and_currency':True,'training_eligible':False}
        save(OUT/'contract-scope-review'/f'{id}.json',rec);rows.append(rec)
    save(OUT/'contract-scope-review-result.json',{'at':now(),'frozen_contract_documents':41,'reviewed':len(rows),'stage_counts':dict(Counter(r['event_stage'] for r in rows)),
        'later_announcements_not_applied_backwards':True,'no_contract_amount_used_as_revenue':True})

if __name__=='__main__':run()
