"""八维业务评价的纯文字生成；只解释事实，不计算评分或推断缺失数据。"""

from copy import deepcopy
from decimal import Decimal, InvalidOperation

REVISION = "RATING_NARRATIVE_20261112_1"
PROFILES = {
    "INDEX": ("指数基金", "跟踪表现", "指数匹配不等于行业风险分散。"),
    "ACTIVE_EQUITY": ("主动权益基金", "合同仓位与策略执行", "集中度不直接代表选股能力或未来收益。"),
    "BOND": ("债券基金", "合同投资与期限要求", "发行人、信用及期限风险不能由股票集中度代替。"),
    "SHORT_BOND": ("短债基金", "合同投资与期限要求", "短期限和低波动不代表没有信用风险。"),
    "MIXED_BOND": ("混合债券基金", "合同配置要求", "债券和权益风险需分别识别。"),
    "CONVERTIBLE": ("可转债基金", "合同配置要求", "需结合债券信用、转股溢价与权益风险理解持仓。"),
    "MONEY": ("货币基金", "合同期限与流动性要求", "分配收益稳定不代表信用与流动性风险为零。"),
    "QDII_EQUITY": ("境外权益基金", "本基金合同目标", "还需关注境外市场、汇率及估值日期差异。"),
    "QDII_BOND": ("境外债券基金", "本基金合同目标", "还需关注信用、期限、汇率及跨市场流动性。"),
    "QDII_COMMODITY": ("境外商品基金", "本基金商品投资目标", "还需关注商品、展期及交易对手风险。"),
    "FOF": ("基金中基金", "合同配置要求", "底层基金数量不等于资产分散，需结合穿透及重叠情况。"),
    "MIXED": ("混合基金", "合同股债配置要求", "股票、债券及现金分别按实际合同口径核对。"),
    "REIT": ("基础设施基金", "项目运营与分配要求", "项目租金、融资与二级市场交易风险不能用普通开放基金口径替代。"),
}


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value))
    except (ValueError, InvalidOperation):
        return None
    return result if result.is_finite() else None


def pct(value):
    return f"{number(value) * 100:.2f}%"


def _assessment(conclusion, paragraphs=(), limitations=()):
    return {"conclusion": conclusion, "paragraphs": list(paragraphs), "limitations": list(limitations)}


def render(item: dict, facts: dict, family: str) -> dict:
    """输入只含同批事实；保留原有分数、范围、状态和等级。"""
    from app.services.fund_rating_narrative_methods import tidy

    output = deepcopy(item)
    profile = PROFILES.get(family, ("本类型基金", "本基金合同目标", "仅依据本基金已披露的适用资料评价。"))
    passive = family == "INDEX"
    performance = facts.get("performance", {})
    period = performance.get("period", "")
    report_date = facts.get("holding_date", "")
    assessments = deepcopy(facts.get("dimension_assessments", {}))
    metrics = {}
    reference_label = performance.get("reference_label", "合同基准")
    comparison_word = "基准" if reference_label == "合同基准" else "参照"

    def metric(key, name, value, unit, when):
        if value is not None and when:
            metrics.setdefault(key, []).append({"name": name, "value": str(value), "unit": unit, "period": when})

    fund, reference = number(performance.get("fund_return")), number(performance.get("reference_return"))
    fund = fund if fund is not None and fund >= -1 else None
    reference = reference if reference is not None and reference >= -1 else None
    if fund is not None:
        metric("return", "近一年总回报", f"{fund * 100:.2f}", "%", period)
        paragraphs = [] if period else [f"考察期内本份额总回报为{pct(fund)}。"]
        if reference is not None:
            gap = (fund - reference) * 100
            position = "领先" if gap > 0 else "落后" if gap < 0 else "持平"
            relative = f"{position}基准" if gap else "与基准持平"
            if fund < 0:
                conclusion = (
                    "比基准少跌，但实际收益仍为负。"
                    if gap > 0
                    else ("基金出现亏损，跌幅大于基准。" if gap < 0 else "基金与基准均出现亏损，期间回报持平。")
                )
            elif fund > 0:
                conclusion = (
                    "取得正收益，同时领先基准。"
                    if gap > 0
                    else ("取得正收益，但仍落后基准。" if gap < 0 else "取得正收益，与基准期间回报持平。")
                )
            else:
                conclusion = f"期间收益持平，{relative}。"
            conclusion = conclusion.replace("基准", comparison_word)
            paragraphs = []
            metric("return", f"{reference_label}回报", f"{reference * 100:.2f}", "%", period)
            metric("return", f"相对{comparison_word}收益差", f"{gap:+.2f}" if gap > 0 else f"{gap:.2f}", "个百分点", period)
        else:
            conclusion = (
                "考察期取得正收益。" if fund > 0 else "考察期实际收益为负。" if fund < 0 else "考察期收益持平。"
            )
        if fund > -1:
            paragraphs.append(
                f"期初一万元、期间不追加或赎回，按上述总回报折算的期末金额约"
                f"{10000 * (1 + fund):.0f}元；交易费用另按实际条件处理。"
            )
        assessments["return"] = _assessment(
            conclusion, paragraphs, ["相对基准的优势不等于绝对盈利，也不能单凭收益差认定主动选股或择时能力。"]
        )

    drawdown = number(performance.get("drawdown"))
    if drawdown is not None and 0 <= drawdown <= 1:
        paragraphs = ["最大回撤指净值从一个高点到随后低点的最大跌幅。"]
        reference_dd = number(performance.get("reference_drawdown"))
        if reference_dd is not None and 0 <= reference_dd <= 1:
            difference = (reference_dd - drawdown) * 100
            relation = (
                f"小{difference:.2f}" if difference > 0 else f"大{abs(difference):.2f}" if difference < 0 else "相同"
            )
            paragraphs.append(
                "基金的最大跌幅"
                + (f"比基准{relation}个百分点。" if difference else "与基准相同。")
            )
            metric("risk", "同期基准最大回撤", f"{reference_dd * 100:.2f}", "%", period)
        if performance.get("drawdown_peak") and performance.get("drawdown_trough"):
            paragraphs.append(
                f"本次最大回撤对应{performance['drawdown_peak']}至{performance['drawdown_trough']}。"
            )
            paragraphs.append(f"高点一万元折算到低点约{10000 * (1 - drawdown):.0f}元。")
        assessments["risk"] = _assessment(
            "本项记录考察期内从高点到低点的跌幅，不能代表未来损失上限。",
            paragraphs,
            [
                "本项未提供同类回撤排名，不能据此声称同类更抗跌。",
                profile[2] if family not in {"INDEX", "ACTIVE_EQUITY"} else "历史回撤不是未来最大可能亏损。",
            ],
        )
        metric("risk", "基金最大回撤", f"{drawdown * 100:.2f}", "%", period)

    execution = facts.get("execution", {})
    error = number(performance.get("tracking_error"))
    if passive and error is not None and error >= 0:
        paragraphs = [
            "跟踪误差衡量基金每日涨跌与基准差异的波动程度，通常越小说明跟踪越稳定。"
        ]
        limit = number(execution.get("tracking_error_limit"))
        if limit is not None and not execution.get("contract_date"):
            paragraphs.append(f"所引合同的年跟踪误差目标为{pct(limit)}以内。")
        transition = execution.get("transition_date")
        post = number(execution.get("post_transition_tracking_error"))
        if transition:
            paragraphs.append(f"基金于{transition}转为ETF联接基金，本次一年考察期跨越转型。")
            if post is not None:
                paragraphs.append(
                    f"单独观察转型后至{facts['as_of_date']}，年化跟踪误差约{pct(post)}"
                    + ("，低于所引合同目标。" if limit is not None and post < limit else "。")
                )
        comparable_error = post if transition else error
        conclusion = (
            "已观察期间的跟踪误差低于所引合同目标。"
            if limit is not None and comparable_error is not None and comparable_error < limit
            else "基金跟踪差异已有实际观测，需结合合同条件理解。"
        )
        assessments["efficiency"] = _assessment(
            conclusion,
            paragraphs,
            [
                "不足完整一年的区间采用年化值，不等于已经完成完整年度验收；跟踪误差单项不代表全部合同要求。",
                "跟踪稳定不等于基金低风险，指数下跌时仍会承担相应损失。",
            ],
        )
        metric("efficiency", "年化跟踪误差", f"{error * 100:.2f}", "%", period)
        if limit is not None:
            metric(
                "efficiency", "合同年跟踪误差目标", f"{limit * 100:.2f}", "%以内", execution.get("contract_date", "")
            )
    elif not passive and execution.get("description"):
        paragraphs = []
        for field, label in (
            ("stock_total_ratio", "股票占总资产"),
            ("stock_nav_ratio", "股票占净资产"),
            ("cash_nav_ratio", "合同认可现金占净资产"),
        ):
            ratio = number(execution.get(field))
            if ratio is not None:
                metric("efficiency", label, f"{ratio * 100:.2f}", "%", report_date)
        low, high, cash_min = (number(execution.get(k)) for k in ("stock_min", "stock_max", "cash_min"))
        if low is not None and high is not None:
            paragraphs.append(f"所引合同股票仓位范围为{pct(low)}至{pct(high)}；不同资产分母按原合同分别核对。")
        if cash_min is not None:
            paragraphs.append(f"所引合同认可现金下限为净资产的{pct(cash_min)}，其他债券或受限资产不能自动计入。")
        assessments["efficiency"] = _assessment(
            f"按{profile[1]}核对本基金的目标执行。",
            paragraphs or [execution["description"]],
            ["合同配置执行不等于收益保证，也不直接证明选股或择时能力。"],
        )

    wins, months = performance.get("months_at_or_above"), performance.get("month_count")
    if (
        isinstance(wins, int)
        and not isinstance(wins, bool)
        and isinstance(months, int)
        and 0 <= wins <= months
        and months > 0
    ):
        assessments["stability"] = _assessment(
            f"本项看相对{comparison_word}的表现能否在不同月份持续出现。",
            [
                f"达标月份占比{Decimal(wins) / months * 100:.2f}%，另外{months - wins}个月落后{comparison_word}。"
                "这反映相对表现是否持续出现，不能由全年领先推断每个月都占优。"
            ],
            ["达到基准的月份不等于赚钱的月份，这个比例也不是未来上涨概率。"],
        )
        metric("stability", "达到或超过基准的月份", f"{wins}/{months}", "个月", period)

    management = facts.get("management", {})
    if management.get("paragraphs"):
        assessments["management"] = _assessment(
            management.get("conclusion", "已有本基金实际任职与公开履职依据。"),
            management["paragraphs"],
            ["本项基于任职、责任接续和公开治理资料，不能直接等同于投资能力；公开披露及托管核查只覆盖所引期间。"],
        )

    holdings = facts.get("holdings", {})
    matched, total = holdings.get("matched_major_holdings_count"), holdings.get("major_holdings_count")
    if passive and isinstance(total, int) and total > 0 and isinstance(matched, int) and 0 <= matched <= total:
        paragraphs = [f"截至{report_date}，已核主要{total}只股票中有{matched}只是目标指数成分。"]
        fraction = number(holdings.get("target_etf_nav_fraction"))
        if fraction is not None and 0 <= fraction <= 1:
            metric("holdings", "目标ETF占净资产", f"{fraction * 100:.2f}", "%", report_date)
        assessments["holdings"] = _assessment(
            "已核主要持仓与指数匹配，投资方向本身的风险仍然存在。"
            if matched == total
            else "已核主要持仓并非全部属于指数成分。",
            paragraphs,
            [
                "本项衡量已核主要持仓的匹配程度，不是分散程度；主要持仓匹配不代表全部持仓逐日完全一致。",
                (
                    f"主要持仓围绕{facts['index_name']}指数配置，仍会承担这一投资方向共同下跌的风险。"
                    if facts.get("index_name")
                    else "持有多只股票仍可能承担共同下跌与集中暴露的风险。"
                ),
            ],
        )
        metric("holdings", "主要持仓成分匹配", f"{matched}/{total}", "只", report_date)
    elif not passive and number(holdings.get("major_holdings_nav_fraction")) is not None:
        ratio = number(holdings["major_holdings_nav_fraction"])
        if 0 <= ratio <= 1 and family in {"ACTIVE_EQUITY", "QDII_EQUITY"}:
            assessments["holdings"] = _assessment(
                "本项看前十大股票占基金净资产的比重。",
                ["这一比例反映主要持仓的集中程度，集中持仓会使少数股票对基金表现的影响更大。"],
                [profile[2]],
            )
            metric("holdings", "前十大股票占净资产", f"{ratio * 100:.2f}", "%", report_date)

    liquidity = facts.get("liquidity", {})
    if liquidity.get("review_complete") is True:
        alert = liquidity.get("reported_redemption_alert")
        paragraphs = []
        net_assets = number(liquidity.get("total_net_assets"))
        if net_assets is not None and net_assets >= 0:
            metric("liquidity", "基金各类份额合计净资产", f"{net_assets / Decimal(100000000):.2f}", "亿元", report_date)
            paragraphs.append("基金规模是各类份额合计净资产，不是仅当前份额的资产。")
        redeemed = number(liquidity.get("reported_redemption_shares"))
        if redeemed is not None and redeemed >= 0:
            paragraphs.append(
                f"上面的赎回量是本{facts.get('share_class', '')}类在报告期内的累计份额，"
                "不是现金金额或单日赎回量，也不表示每笔申请当天到账。"
            )
        paragraphs.append(
            "已核资料包含实际申赎及托管复核。"
            + ("存在已记录的赎回异常或压力影响。" if alert else "所引资料未标记已证实的赎回异常。")
        )
        if liquidity.get("deferred_redemption"):
            paragraphs.append("合同约定发生巨额赎回时可部分延期办理，实际退出速度受当时资产变现能力等条件影响。")
        assessments["liquidity"] = _assessment(
            "已有实际申赎运作依据，极端情况下退出仍可能受影响。" if not alert else "已记录赎回压力，需要关注实际影响。",
            paragraphs,
            ["本项分数反映所引期间的运作证据，不保证未来随时全部赎回或固定时间到账。"],
        )
        if redeemed is not None and redeemed >= 0:
            metric(
                "liquidity",
                "本份额报告期累计赎回",
                f"{redeemed / Decimal(100000000):.2f}",
                "亿份",
                f"报告期截至{report_date}",
            )

    cost = facts.get("cost", {})
    amount = number(cost.get("annual_cost_per_10000"))
    if amount is not None and amount >= 0:
        fee_date = facts.get("fee_date", "")
        paragraphs = [
            "上面的金额按本份额一万元持有一年估算；实际费用随资产价值、持有条件及实际发生额变化。"
        ]
        rate = number(cost.get("annual_total_rate"))
        if rate is not None:
            paragraphs.append(f"所引资料概要的年化运作综合费率测算为{pct(rate)}，不是几项名义费率的简单相加。")
        if cost.get("no_subscription_fee"):
            paragraphs.append("本份额不收申购费，持有期间仍有运作费用。")
        if cost.get("short_redemption"):
            paragraphs.append(cost["short_redemption"])
        if cost.get("feeder_fee_waiver"):
            paragraphs.append("投资目标ETF的部分不重复收取本基金层面的管理费和托管费。")
        if cost.get("detail_gap"):
            paragraphs.append(cost["detail_gap"])
        assessments["cost"] = _assessment(
            "持续持有有运作成本，交易费用需按实际持有条件区分。",
            paragraphs,
            ["没有相同持有期限和份额条件的比较，不能据此判断费用同类最低或C类一定比A类划算。"],
        )
        metric(
            "cost", "一万元持有一年估算成本", f"{amount:.2f}", "元", f"费用资料{fee_date}；持有一年" if fee_date else ""
        )

    for dimension in output.get("dimensions", []):
        key = dimension["key"]
        if dimension.get("score") is None:
            dimension["assessment"] = facts.get("dimension_assessments", {}).get(key) or _assessment(
                "本维度尚无确定分，保留已披露依据与具体限制。",
                [dimension.get("explanation", "")],
                ["现有范围或资料不能替代完整确定评分。"],
            )
            assessment = dimension["assessment"]
            assessment = tidy(assessment)
            dimension["assessment"] = assessment
            dimension["explanation"] = " ".join(
                [assessment["conclusion"], *assessment["paragraphs"], *assessment["limitations"]]
            )
            continue
        else:
            dimension["assessment"] = assessments.get(key) or _assessment(
                f"本项按{profile[0]}的适用依据评价。",
                [dimension.get("explanation", "")],
                [f"本批次尚未整理出可自动解释本项分数的完整事实；不从分数反推{profile[1]}或能力结论。"],
            )
        additions = facts.get("extra_metrics", {}).get(key, [])
        if metrics.get(key) or additions:
            dimension["metrics"] = metrics.get(key, dimension.get("metrics", [])) + additions
        if facts.get("sources", {}).get(key):
            dimension["sources"] = facts["sources"][key]
        # 旧客户端仍读取explanation，保持同样的事实和边界。
        assessment = dimension["assessment"]
        purpose = facts.get("dimension_purposes", {}).get(key)
        if purpose and assessment["conclusion"] and (
            assessment["conclusion"].startswith(("原", "既有")) or any(
                text.startswith(assessment["conclusion"].rstrip("。")) for text in assessment["paragraphs"]
            )
        ):
            assessment = {**assessment, "conclusion": purpose}
        assessment = tidy(assessment, facts.get("score_methods", {}).get(key))
        dimension["assessment"] = assessment
        dimension["explanation"] = " ".join(
            [assessment["conclusion"], *assessment["paragraphs"], *assessment["limitations"]]
        )
    if fund is not None and drawdown is not None and item.get("status") in {"SCORED", "RATED"}:
        output["summary"] = (
            f"考察期总回报{pct(fund)}，最大回撤{pct(drawdown)}。"
            + assessments["return"]["conclusion"]
            + ("跟踪表现与投资方向本身的风险需要分别理解。" if passive else "历史收益与合同执行需分别理解。")
        )
    elif item.get("status") in {"SCORED", "RATED"} and fund is not None:
        output["summary"] = f"考察期总回报{pct(fund)}。" + assessments["return"]["conclusion"]
    if performance.get("period"):
        output["dataDates"] = {**output.get("dataDates", {}), "nav": facts.get("as_of_date")}
    return output
