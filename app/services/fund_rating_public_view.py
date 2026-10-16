"""公开资料评级的用户依据：显示事实、日期及限制，绝不透出分数、权重和验证任务。"""

from app.services.fund_rating_public_policy import (
    ETF_BASIS,
    GLOBAL_BASIS,
    INDEX_BASES,
    MARKET_BASES,
    MARKET_BASIS,
    policy,
)
from app.services.fund_rating_rules import DIMENSIONS, GRADES, decimal, grade

LABELS = {
    "return_1y": ("近一年总回报", "%"),
    "return_3y": ("近三年年化总回报", "%"),
    "drawdown": ("三年最大回撤幅度", "%"),
    "downside": ("年化下行波动", "%"),
    "calmar": ("年化收益与最大回撤之比", ""),
    "rolling_positive": ("滚动一年正收益窗口占比", "%"),
    "monthly_positive": ("正收益月份占比", "%"),
    "team_months": ("本届团队连续管理", "月"),
    "team_changes": ("三年正式团队变更", "次"),
    "strategy_continuity": ("报告期投资策略", ""),
    "governance_state": ("管理人监管事项", ""),
    "issuer_hhi": ("股票发行人集中度", ""),
    "industry_hhi": ("股票行业集中度", ""),
    "restricted_ratio": ("报告期受限股票占净资产", "%"),
    "cash_ratio": ("报告期货币资金占净资产", "%"),
    "holder_risk_band": ("报告期单一持有人披露", ""),
    "net_assets": ("基金净资产", "亿元"),
    "cost_1y": ("一万元持有一年标准费用率", "%"),
}
NOTES = {
    "return": "根据真实净值与分红折算事件计算历史总回报，不作同类排名。",
    "risk": "较高回撤和下行波动会降低评价，过去的跌幅不代表未来损失上限。",
    "efficiency": "衡量同一三年区间内的年化收益与最大回撤，了解承受历史回撤所获得的回报。",
    "stability": "同时观察25个滚动一年窗口和36个月的正收益比例；窗口有重叠，不表示未来上涨概率。",
    "management": "依据正式团队任免、报告期策略及监管事项；已经整改的监管事项仍计入评价，不等同于经理能力判断。",
    "holdings": "集中度越高，分散程度越低。按完整股票持仓及报告行业分类衡量，报告持仓不是实时持仓。",
    "liquidity": (
        "使用报告期现金、受限股票、规模及大额持有人披露；低于20%仅表示披露风险档，不是精确持有人比例或赎回到账承诺。"
    ),
    "cost": "按一万元持有365天、无涨跌计算综合运作及适用交易费用；不使用个人折扣，不从历史收益重复扣费。",
}


def public_evidence(member: dict, computed: dict | None, as_of) -> dict:
    bundle, metrics = member["bundle"], member["metrics"]
    etf = bundle.get("basis") == ETF_BASIS
    index = bundle.get("basis") in INDEX_BASES
    config = policy(bundle["basis"])
    labels = dict(LABELS)
    notes = dict(NOTES)
    if bundle.get("basis") in {*MARKET_BASES, *INDEX_BASES}:
        labels["strategy_continuity"] = ("报告期重大投资策略变化", "")
        notes["return"] = "根据已核验的复权单位净值计算历史总回报，分红折算已包含，不作同类排名。"
        notes["management"] = (
            "经理任职资料与管理人原件核对，并结合报告披露的重大策略变化和监管事项；不等同于经理能力判断。"
        )
    if etf or index:
        labels.update(
            tracking_error_ratio=("三年年化跟踪误差", "%"),
            tracking_bias_ratio=("三年日均绝对跟踪偏离", "%"),
            net_assets=("基金产品净资产（A/C合计）", "亿元"),
            restricted_ratio=("穿透受限股票占联接基金净资产", "%"),
        )
        notes.update(
            management="按官方业绩基准核对跟踪表现，并结合报告策略和管理人监管披露；合同跟踪目标不是收益保证。",
            holdings="穿透目标ETF的完整股票和报告行业结构；主题行业集中是实际风险，报告持仓不代表实时持仓。",
            liquidity="依据报告货币资金、穿透受限股票和产品规模；未公开最大个人持有人比例，不能据此判断为零或承诺赎回到账。",
        )
        if index:
            labels["net_assets"] = ("基金产品净资产（各类份额合计）", "亿元")
            labels["restricted_ratio"] = ("报告期末受限资产占净资产", "%")
            notes["holdings"] = "根据完整股票持仓与行业表衡量实际集中风险；指数的主题集中不代表管理人偏离目标。"
            notes["liquidity"] = (
                "依据报告期末受限资产、货币资金和产品规模；未公开最大持有人精确比例，不承诺赎回到账时间。"
            )
    sources = [{"title": s["title"], "url": s["url"], "publishedOn": s["published_on"]} for s in bundle["sources"]]
    period = f"{bundle['history']['dates'][0]} 至 {as_of}"
    dimensions = []
    for key, name in DIMENSIONS.items():
        g = grade(computed["dimensions"][key]) if computed else None
        values = []
        for metric, _ in config["terms"][key]:
            label, unit = labels[metric]
            n = decimal(metrics[metric.removesuffix("_ratio")] if metric.startswith("tracking_") else metrics[metric])
            value = f"{n * 100 if unit == '%' else n / 100_000_000 if unit == '亿元' else n:.2f}"
            if metric == "strategy_continuity":
                value = (
                    ("未发生重大改变" if bundle.get("basis") in {*MARKET_BASES, *INDEX_BASES} else "未发生改变")
                    if n == 1
                    else "发生改变"
                )
            if metric == "governance_state":
                value = {0: "有事项，整改未完成", 1: "报告披露受到监管措施，已整改", 2: "报告明确无事项"}[int(n)]
            if metric == "holder_risk_band":
                value = {
                    0: "未出现达到或超过20%的单一持有人",
                    1: "存在20%至50%的单一持有人",
                    2: "存在达到或超过50%的单一持有人",
                }[int(n)]
            if metric in ("issuer_hhi", "industry_hhi"):
                value = f"{n:.4f}"
            date_label = (
                member["facts"]["holdings"]
                if key in ("holdings", "liquidity") or metric in ("strategy_continuity", "governance_state")
                else period
            )
            if key == "cost":
                date_label = f"费用资料核对 {member['facts']['fees']}"
            if metric == "return_1y":
                date_label = f"截至 {as_of} 的近一年"
            values.append({"name": label, "value": value, "unit": unit, "period": date_label})
        dimensions.append(
            {
                "key": key,
                "name": name,
                "grade": g,
                "gradeLabel": GRADES[g] if g else None,
                "explanation": notes[key],
                "metrics": values,
                "sources": sources,
            }
        )

    def points(good):
        return [
            {"dimension": d["key"], "text": f"{d['name']}：{d['gradeLabel']}，详见对应历史表现与披露依据。"}
            for d in dimensions
            if (d["grade"] in {"GOOD", "EXCELLENT"} if good else d["grade"] == "WEAK")
        ][:3]

    limitations = [
        "本评级衡量历史表现与公开披露，不预测未来收益；不同类型的同名等级不能直接比较。",
        f"持仓、现金和受限情况截至 {member['facts']['holdings']}，不代表评价日实时资产或五日收付款。",
        "联接基金最大个人持有人精确比例未公开，不能从个人持有100%推断为零或低于20%。"
        if etf or index
        else "最大持有人精确比例未公开，持有人评价采用报告披露的区间信息。",
        "该评价口径尚未完成大范围同类及未来表现验证。",
    ]
    if computed:
        from app.services.fund_rating_public_validation import sensitivity

        if sensitivity(metrics, computed, config=config)["changed_grades"]:
            limitations.insert(
                0, "综合评价接近相邻等级分界；收益表现与风险、分散程度存在明显差异，需结合八维依据理解。"
            )
    return {
        "summary": f"根据已核验的历史表现和公开披露，八维综合评价为{GRADES[computed['grade']]}。"
        if computed
        else "公开八维原件已齐，评价规则尚未通过验证。",
        "comparison": {
            "category": bundle["category_label"],
            "currency": "CNY",
            "productCount": 0,
            "scope": "境内ETF联接基金的公开资料评价，采用统一评价区间，不作同类排名"
            if etf
            else "人民币境外主动股票基金的公开资料评价，采用统一评价区间，不作同类排名"
            if bundle.get("basis") == GLOBAL_BASIS
            else "境内主动权益基金的公开资料评价，采用统一评价区间，不作同类排名"
            if bundle.get("basis") == MARKET_BASIS
            else "境内主动混合基金的公开资料评价，采用统一评价区间，不作同类排名",
            "period": period,
        },
        "dimensions": dimensions,
        "strengths": points(True),
        "weaknesses": points(False),
        "dataDates": member["facts"],
        "limitations": limitations,
    }
