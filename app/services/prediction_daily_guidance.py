"""每日执行的审阅指引：约束实际分析过程，不自动决定方向或改变严格规则。"""

VERSION = "DAILY_V2_BASELINE_SENSITIVITY_20261116"
DOCUMENT = "docs_zhx/implementation/daily-prediction-review-guidance-2026-11-16.md"


def review_guidance():
    """每次返回独立快照，供请求和资料冻结回执留证；不含基金名单或旧方向。"""
    return {
        "version": VERSION,
        "document": DOCUMENT,
        "purpose": "动态读取本人持仓，以本期真实资料实际生成四周期分析并正式读回。",
        "requirements": {
            "current_window": "核官方窗口、真实基准净值及公告可见时间；缺净值先补，未到位就等待。"
            "收盘后不得倒填旧一日预测，不用场内价格、估算净值或代理汇率代替真实基准。",
            "evidence": "按用途有界补原文并阅读，保留取得时间、来源版本与引文；"
            "报告持仓、当前篮子、现金存量和当期现金流分别说明，不冒充实时全仓。",
            "baseline": "先独立写基准和最强反证，再判断净方向。部分覆盖、未知精确幅度或未来风险"
            "不自动阻断低把握方向；历史股息、低估值和正现金也不自动推出上涨。",
            "sensitivity": "分别列当前资料缺口EVIDENCE_GAP与未来反转风险FUTURE_RISK。"
            "若合理的当前未知可改变方向仍暂不判断；不能为通过校验把UNKNOWN改成WEAKER。"
            "已知负现金、信用恶化和资金占用应纳入基准反证，不能包装成待补资料。",
            "periods": "一日看目标估值时段剩余传导，五日看多日承接，二十日看经营政策兑现，"
            "半年看持续收入现金成本。逐周期独立取舍，不把单日涨跌乘期限或复制方向。",
            "conditional": "半年列基准、有利、不利三情景及失效条件；官方终点未核可作条件展望。"
            "不伪造2027日历，不将条件意见或暂不判断算作已核日期的预测。",
            "publication": "本轮完成sources、prepare、实际撰写、evaluate、reviewed-trial-ref和正式发布读回。"
            "禁止按基金代码套用旧结论，不用旧116项核对、草稿或资料冻结冒充新增预测。",
            "continuation": "同材料同窗口已完成的严格审阅不重复造数；新一期重新分析。"
            "个别基金外部等待不阻断其他基金，仍可解析阅读编译发布的工作必须在本轮推进。",
            "reporting": "分别报告新带日期方向、半年条件方向、已审阅暂不判断、资料等待、"
            "尚未审阅和草稿；先报本轮新增，再报沿用旧成果，新增为零明确写零。",
        },
        "automatic_decision": False,
        "changes_strict_validation": False,
    }
