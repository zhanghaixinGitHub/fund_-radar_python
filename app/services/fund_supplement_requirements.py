"""每日补数共用项目及逐基金档案中的取数任务。"""

SECTIONS = ("评分", "四周期预测", "买卖建议", "预测结果核验")
CYCLES = ("T1", "T5_V1", "T20_V1", "M6_V1")
CYCLE_NAMES = dict(zip(CYCLES, ("一日", "五日", "二十日", "半年"), strict=True))
RATING_WINDOW = "最近已结束评分月，按正式指标各自的日线、自然月及滚动区间回溯"
NAV_API = "Tushare fund_nav https://tushare.pro/document/2?doc_id=119"
SHARE_API = "Tushare fund_share https://tushare.pro/document/2?doc_id=207"
FX_SOURCE = "中国外汇交易中心 https://www.chinamoney.com.cn/chinese/bkccpr/index.html?tab=2"


def disclosure_directory(code):
    return f"按 {code} 查定期报告、合同、招募说明书及变更公告 https://fundf10.eastmoney.com/jjgg_{code}.html，核对管理人或托管人原件"


def nav_sources(code):
    return f"{NAV_API}，查询 `{code}.OF`；核对本份额正式净值历史 https://fundf10.eastmoney.com/jjjz_{code}.html 及管理人披露"


def section_for(use):
    if use in SECTIONS:
        return use
    if use.startswith("评分"):
        return "评分"
    if use.startswith(("建议", "买卖")):
        return "买卖建议"
    if use.startswith(("预测", "四周期")):
        return "四周期预测"
    return use


def checklist_requirements(code, profile):
    """共用项目复用同一规则；资产、合同基准和周期资料取自该基金档案。"""
    focus = profile.get("supplement_focus", {})
    directory = disclosure_directory(code)
    report = "；".join(filter(None, (profile.get("report_source"), directory)))
    benchmark = profile.get("benchmark") or "待核对本份额生效合同"
    benchmark_source = focus.get("benchmark_source") or "本份额合同列明的指数发布机构、对应利率及授权日线"
    fund = "基金管理人及托管人原件"
    notices = f"{directory}；巨潮资讯 https://www.cninfo.com.cn/ 按基金或证券代码查询"
    own_nav = (f"{nav_sources(code)}；Tushare fund_div https://tushare.pro/document/2?doc_id=120，"
               f"核对本份额分红公告与除息日 https://fundf10.eastmoney.com/fhsp_{code}.html")
    market = focus.get("market_source") or "该基金持仓所在交易所、发行人和行情数据源"

    def row(key, use, label, source, period, verify, *, route="人工取证", horizons=(), dimension=None):
        return dict(id=key, check_key=key, use=use, item=label, source=source, data_window=period,
                    verification=verify, collection=route, horizons=list(horizons), rating_dimension=dimension)

    rows = [
        row("rating_return", "评分", "收益能力 · 本份额真实费后回报、同期合同基准和适用比较组净值",
            f"{own_nav}；{benchmark_source}", RATING_WINDOW,
            "核对份额及分红再投资口径、实际起止估值日、正式比较组和同一日期的基准点位", dimension="return"),
        row("rating_risk", "评分", "风险控制 · 同期日净值、最大回撤及适用的风险暴露",
            f"{own_nav}；{report}；相应债券、股票或境外市场资料", RATING_WINDOW,
            "核对回撤起止、交易日缺口及异常跳点；按基金类型核大小盘、行业、币种、利率或信用暴露",
            dimension="risk"),
        row("rating_execution", "评分", "目标执行 · 生效合同、资产比例明细、实际跟踪或策略执行记录",
            f"{profile.get('contract_source') or '在基金公告目录查本份额基金合同及修订公告'}；{report}；{benchmark_source}",
            RATING_WINDOW, "核对合同基准及生效段，按产品适用口径核跟踪误差、比例门槛或实际策略执行",
            dimension="efficiency"),
        row("rating_stability", "评分", "业绩稳定性 · 月末与滚动区间回报、基准及正式比较组",
            f"{own_nav}；{benchmark_source}", RATING_WINDOW,
            "逐月按实际估值日对齐，区分自然月和交易日，保留不能比较的月份", dimension="stability"),
        row("rating_management", "评分", "管理与策略 · 当前经理任职职责、交接、策略变化和治理",
            f"{notices}；{report}", "评分截止时生效的任职和策略，含当期变化",
            "核对产品、任职和公告生效日，并查实际执行或交接资料", dimension="management"),
        row("rating_holdings", "评分", "持仓结构 · 各类资产、完整或已披露持仓、集中度和底层穿透",
            f"{report}；联接产品另取目标基金报告、对应日期指数成分及权重",
            "评分截止前可用的最近披露，历史对照分别注明报告期",
            "核对证券身份、报告期和总资产或净资产分母，季度前列持仓与半年报完整持仓分别记录",
            dimension="holdings"),
        row("rating_liquidity", "评分", "规模与流动性 · 基金份额、规模、申赎、现金、受限资产和成交",
            f"{report}；{SHARE_API}；实际申赎、受限资产及未公开明细向{fund}的披露渠道取证",
            "评分所用报告期和最近实际可得时点",
            "核对份额单位、基金与本份额口径、披露日、受限资产、集中持有人和赎回条件",
            dimension="liquidity"),
        row("rating_cost", "评分", "费用 · 本份额管理费、托管费、销售服务费、申购赎回费及底层减免",
            f"{directory}，取现行招募说明书、产品资料概要和费率变更公告；费率页交叉核对 https://fundf10.eastmoney.com/jjfl_{code}.html",
            "评分截止日有效版本及统一申购金额、持有期",
            "核对份额类别、费率分档、持有期和生效日，按现行评分口径计算",
            dimension="cost"),
    ]
    if not focus:
        rows.append(row("fund_profile", "评分", "核对本份额个别档案的合同基准、真实投向和四周期来源",
                        f"{notices}；{report}", "该持仓进入每日任务时",
                        "将已核对的基准、相关市场、估值、汇率及各周期资料写入本基金档案"))
    if detail := focus.get("rating_data"):
        rows.append(row("rating_product_detail", "评分", detail, f"{benchmark_source}；{report}", RATING_WINDOW,
                        f"先核对本份额合同口径（{benchmark}）、基准身份和日期，再核原始观测覆盖及分档依据"))

    cycle_defaults = (
        "基准日同口径行情及下一实际估值日开盘前后能进入判断的事件",
        "基准日前近几日的价格与成交，以及后续五个实际估值日催化和反证",
        "后续二十个实际估值日内的政策、公告、经营及资金验证日期",
        "实际基准日起至半年目标的政策、行业和盈利路径，含基准、有利、不利情景",
    )
    for horizon, period in zip(CYCLES, cycle_defaults, strict=True):
        detail = focus.get("cycles", {}).get(horizon) or period
        rows.append(row("prediction_" + horizon.lower(), "四周期预测", f"{CYCLE_NAMES[horizon]} · {detail}",
                        market, period, "对齐市场和基金估值日、公开时点、来源身份和持仓。留可验证的主要依据及反证",
                        horizons=(horizon,)))
    rows.append(row(
        "prediction_exposure", "四周期预测", "最新公开投向、主要持仓或指数权重及代表范围",
        f"{report}；{benchmark_source}；发行人持仓和公司行动见 https://www.cninfo.com.cn/",
        "采取预测时已公开的最新资料，分别记录持仓报告期与实际行情日",
        "核对上期实际持仓、最新部分披露及对应标的。注明成交观测、停复牌、除权和代表范围",
        horizons=CYCLES,
    ))
    for target in profile.get("target_funds", []):
        if target in profile.get("target_market_specialized", ()):
            continue
        rows.append(row(
            "prediction_target_" + target, "四周期预测",
            f"底层 {target} 正式净值、份额、场内成交与跟踪偏离",
            f"{disclosure_directory(target)}；{SHARE_API}；目标基金正式净值、交易所同日收盘和指数发布方",
            "基准估值日及对应比较交易日；历史跟踪按所需窗口",
            "按目标基金本身的代码和估值日核对净值，场内价和正式净值同日才算折溢价；资金变化用份额资料核实",
            horizons=CYCLES,
        ))
    if profile.get("template") == "overseas":
        rows.append(row(
            "prediction_overseas_alignment", "四周期预测",
            "境外市场实际交易日、收市时点、估值折算汇率和已披露对冲",
            f"{market}；{profile.get('contract_source') or directory}；{FX_SOURCE}；跨币种按基金合同列明渠道向管理人或托管人核对",
            "基准及目标的本份额估值日，匹配境外市场当地交易日和对应汇率",
            "区分两地假日、提前收市、已经发生的境外行情、汇率及未知对冲比例",
            horizons=CYCLES,
        ))
    rows.extend([
        row("advice_rating", "买卖建议", "当期八项正式评分与持有诊断",
            "基金雷达正式评分及持有诊断，包括经理、规模、同类排名、基准、回撤、费用和分红",
            "本次建议时点及最近应评月份", "核对当前有效评分来源和适用限制，读回同一基金完整诊断",
            route="系统读回"),
        row("advice_predictions", "买卖建议", "四个周期有效预测及本人持有偏好要求的周期",
            "基金雷达正式预测快照和本人持有偏好", "本次建议对应的基础与目标估值窗口",
            "核对所需周期的方向、有效期、来源版本和缺项，审阅后仍无方向的独立记录",
            route="系统读回", horizons=CYCLES),
        row("advice_ledger", "买卖建议", "本人正份额持仓、可用或冻结份额、组合占比、成本与盈亏、订单和定投",
            "当前账号的我的持仓、订单、定投账本及最近正式估值", "生成建议时",
            "核对份额代码、余额、估值日、待确认订单及定投状态",
            route="系统读回"),
        row("advice_funding", "买卖建议", "本人持有偏好、用款日及金额、可承受损失和资金复查日期",
            "基金雷达「我的持仓」的本人资金安排和持有偏好，变更或到期由本人重新确认", "生成建议时",
            "核对有效或撤销状态、复查是否到期。需要用款时核现金覆盖、赎回条件和到账日期",
            route="系统读回及本人确认"),
        row("advice_conditions", "买卖建议", "本份额当前申购赎回、限额、到账和执行费用",
            f"{notices}；现行产品资料概要", "建议生效的交易和结算窗口",
            "核对本份额申购和赎回分别适用的暂停、金额限制、费用分档及到账日"),
        row("outcome_nav", "预测结果核验", "到期预测的本份额正式结果日净值",
            own_nav, "逐条使用原预测的一日、五日、二十日和半年实际结果日",
            "对齐原窗口、单位或累计净值及分红规则后核验实际涨跌，保存并读回结果",
            route="既有结果核对流程", horizons=CYCLES),
    ])
    # 已有逐基金专项继续保留自己的来源、日期和取证规则。
    default_periods = {"评分": RATING_WINDOW, "四周期预测": "清单所示基准日和该专项对应的实际预测窗口",
                       "买卖建议": "建议生成时及个人安排生效区间"}
    for requirement in profile.get("data_requirements", []):
        rows.append({
            "data_window": default_periods.get(section_for(requirement["use"]), "该专项所要求的实际日期"),
            "verification": "核对所需日期、基金或证券身份、数据口径与原件，缺项保留待核",
            **requirement,
        })
    return rows
