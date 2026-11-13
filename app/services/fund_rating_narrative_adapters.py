"""各类正式评分的展示适配；复用已冻结业务事实，不从分数倒推指标。"""

import re
from decimal import Decimal

from app.services.fund_rating_narrative import number, pct

KEYS = {
    "return": "return",
    "risk": "risk",
    "execution": "efficiency",
    "stability": "stability",
    "management": "management",
    "holdings": "holdings",
    "liquidity": "liquidity",
    "fee": "cost",
}
TITLES = {
    "return": "收益表现",
    "risk": "风险暴露",
    "efficiency": "投资目标执行",
    "stability": "表现持续性",
    "management": "管理与履职",
    "holdings": "持仓结构",
    "liquidity": "规模与退出条件",
    "cost": "持有成本",
}


def business_text(value):
    """只使用底稿中的业务说明；内部路径、摘要、换分档位及研发状态不展示。"""
    if not isinstance(value, str):
        return ""
    sentences = re.split(r"(?<=[。！？])", value)
    forbidden = (
        r"[A-Z]:[\\/]|[a-f0-9]{64}|(?:R\d+|M\d+)(?:规则|口径|协议|量尺|已|统一)|"
        r"sha256|CLI|SQLite|数据库|模块|编译器|探针|候选|原内部权重|40/20/40|50/75/50|"
        r"输入摘要|评分包|重放|回放|固定权重|有理数|状态码"
    )
    return "".join(s for s in sentences if not re.search(forbidden, s, re.I)).strip()


def base_facts(bundle):
    """无新增原件I/O：当前正式底稿和公开说明已通过读取层的来源/批次核验。"""
    from app.services.fund_rating_narrative_methods import PURPOSE

    dates = bundle.get("data_dates", {})
    facts = {
        "as_of_date": bundle["as_of_date"],
        "holding_date": bundle["holding_date"],
        "fee_date": dates.get("fees"),
        "share_class": bundle["share_class"],
        "sources": {},
        "dimension_assessments": {},
        "native_metrics": bundle.get("public_metrics", {}),
    }
    dimensions = bundle.get("reviewed_document", {}).get("dimensions", {})
    explanations = bundle.get("public_explanations", {})
    for public, internal in KEYS.items():
        node = dimensions.get(public)
        reason = business_text(node.get("reason", "")) if isinstance(node, dict) else ""
        if re.fullmatch(r"原\d+/\d+。?", reason):
            reason = ""
        explanation = business_text(explanations.get(public, ""))
        texts = list(dict.fromkeys(t for t in (explanation, reason) if t))
        candidate = node.get("candidate") if isinstance(node, dict) else node
        if number(candidate) is None:
            conclusion = f"{TITLES[internal]}尚无确定分，已知事实与缺项分别保留。"
        else:
            conclusion = PURPOSE[internal]
        facts["dimension_assessments"][internal] = {
            "conclusion": conclusion,
            "paragraphs": texts
            or ["当前正式结果保留了本项分数或范围，但未附可直接展示的原始业务观测；不由分数反推指标。"],
            "limitations": [],
        }
    return facts


def _check_points(computed, dimensions):
    for key, point in computed.items():
        expected, observed = number(dimensions.get(key)), number(point)
        if expected is not None and (observed is None or f"{expected:.2f}" != f"{observed:.2f}"):
            raise ValueError("NARRATIVE_TYPED_SCORE_MISMATCH:" + key)


def _mixed_execution_assessment(execution, rule):
    """按已核事实解释合同要求；评分占比不当作合同投资比例。"""
    states = execution.get("core_constraints", {})
    labels = {"stock": "股票仓位", "cash": "现金比例", "leverage": "总资产比例"}
    groups = []
    for state, wording in ((False, "未达到本次核对采用的要求"), (None, "仍缺少确认依据"), (True, "满足所引要求")):
        names = [label for key, label in labels.items() if states.get(key) is state]
        if names:
            groups.append("、".join(names) + wording)
    result_text = {
        True: "本期满足这一要求。",
        False: "本期未达到这一要求。",
        None: "现有资料还不能确认是否满足。",
    }
    cash_text = result_text[states.get("cash")]
    upper, minimum = number(execution.get("qualified_cash_upper_nav_ratio")), number(rule["cash_min"])
    if states.get("cash") is False and upper is not None and minimum is not None and upper < minimum:
        cash_text = "即使按最宽松的计入范围估算，仍不足这一下限；实际可计入的比例可能更低。"
    paragraphs = [
        f"股票投资范围为基金资产的{pct(rule['stock_min'])}至{pct(rule['stock_max'])}。"
        + result_text[states.get("stock")],
        f"现金及一年内到期的政府债券，扣除相关交易保证金后，合计应不低于净资产的{pct(minimum)}。"
        + cash_text,
        f"基金总资产不得超过净资产的{pct(rule['leverage_max'])}。" + result_text[states.get("leverage")],
    ]
    limitations = ["以上只核对所引的三项要求；比例差异的原因、适用条件和后续调整仍需核实，不能直接据此认定违法违规。"]
    clause = rule.get("claim", {}).get("quote", "")
    if "开放期内" in clause and "封闭期内" in clause:
        limitations.insert(0, "这里采用所引合同的开放期要求；封闭期要求不同，仍需结合基金当时所处阶段判断。")
    return {"conclusion": "；".join(groups) + "。", "paragraphs": paragraphs, "limitations": limitations}


def _remediation_management_assessment(inputs):
    """只解释类型计算器已核过的整改披露，不把公司声明当作监管确认。"""
    review = inputs.get("manager_review", {})
    claims = review.get("claims", [])
    if inputs.get("management_inputs") or review.get("governance_state") != "REMEDIATION_DISCLOSED" or len(claims) != 2:
        return None
    disclosure = re.sub(r"\s+", "", claims[1]["quote"])
    event = re.search(r"受到(?:调查|稽查)或处罚等措施的时间(20\d{2})年(\d{1,2})月(\d{1,2})日", disclosure)
    authority = re.search(r"采取调查或处罚等措施的机构(.+?)受到调查或处罚等措施类型", disclosure)
    measure = re.search(r"受到的具体措施类型(.+?)受到调查或处罚等措施的原因", disclosure)
    reason = re.search(r"受到调查或处罚等措施的原因(.+?)受到处罚的依据", disclosure)
    if not all((event, authority, measure, reason)) or not review.get("current_names"):
        return None
    year, month, day = map(int, event.groups())
    paragraphs = [
        f"截至{inputs['holding_date']}，所引报告列示的本基金经理为{'、'.join(review['current_names'])}。",
        f"{year:04d}-{month:02d}-{day:02d}，{authority[1]}向{inputs['manager_name']}{measure[1]}。"
        f"报告列明的原因是：{reason[1]}。",
        "管理公司在报告中表示已完成整改"
        + ("，并已向监管机构报送整改报告。" if "报送整改报告" in disclosure else "。"),
    ]
    return {
        "conclusion": "现有资料还不能确认监管机构是否认可整改结果。",
        "paragraphs": paragraphs,
        "limitations": [
            "上述事项涉及基金管理公司，不能直接理解为本基金经理个人受到处罚，也不直接代表其投资能力。",
        ],
    }


def _major_holdings_assessment(holding, share_class, score):
    """说明已核持仓占比和本项目计分方法，不改变正式分数。"""
    stocks, bonds = holding.get("stocks"), holding.get("bonds")
    amount, net = number(holding.get("major_risk_concentration_cny")), number(holding.get("net_assets"))
    point = number(score)
    if (
        not isinstance(stocks, list) or not isinstance(bonds, list)
        or amount is None or net is None or net <= 0 or point is None
    ):
        return None
    non_government = [row for row in bonds if "国债" not in row["name"]]
    ratio = amount / net * 100
    calculation = (
        f"本次：100－{ratio:.2f}＝{point:.2f}分。"
        if ratio <= 100 else f"本次扣除后低于0，因此按{point:.2f}分计。"
    )
    paragraphs = [
        f"计算范围：报告披露的前{len(stocks)}只股票，加上前{len(bonds)}只债券中的{len(non_government)}只非国债。"
        f"将它们的市值合计，除以本基金各类份额合计净资产，得到上面的占比；分母不是仅{share_class}类的资产。",
        "本项目按“100减去上述占比”计分，最低为0分。" + calculation,
    ]
    return {
        "conclusion": "本项看选定主要持仓占基金净资产多少；占比越低，得分越高。",
        "paragraphs": paragraphs,
        "limitations": [
            "国债和其他未纳入的持仓仍可能有风险；高分只说明这批选定持仓占比较低，不代表整只基金风险低。",
        ],
    }


def _holder_liquidity_assessment(inputs, raw):
    """说明实际规模、累计申赎和固定给分条件，保留必要的赎回限制。"""
    review = inputs.get("holder_review", {})
    redeemed = number(raw.get("reported_redemption_shares"))
    if (
        review.get("execution_profile") != "TRUSTEE_REDEMPTION_PRICING_AND_REPORT_REVIEW"
        or review.get("actual_effect_review_complete") is not True
        or review.get("confirmed_pressure_effects")
        or redeemed is None or redeemed < 0
    ):
        return None
    paragraphs = [
        f"基金规模是各类份额合计净资产；{inputs['share_class']}类赎回量是报告期累计赎回份额，"
        "单位是“份”，不是现金金额，也不是单日赎回量。",
        "本项目的计分规则：已核对合同赎回安排、该类份额的实际申赎数据和托管核查资料，"
        "且未发现已确认的赎回压力影响时，固定给75分。这个分数没有按逐笔到账时间计算。",
    ]
    mechanism = re.sub(r"\s+", "", review.get("mechanism", {}).get("quote", ""))
    if all(word in mechanism for word in ("巨额赎回", "部分延期赎回")):
        paragraphs.append(
            "合同中的巨额赎回延期条款，是特殊情况下的处理办法，不表示本期已经发生延期；"
            "实际办理仍须按申请渠道和适用的合同条件执行。"
        )
    return {
        "conclusion": "本项反映已披露申赎运作的核对结果。",
        "paragraphs": paragraphs,
        "limitations": ["这些资料不保证未来随时全额赎回或在固定日期到账。"],
    }


def typed_facts(bundle, dimensions, *, root, facts):
    from app.services.fund_rating_typed_disclosure import calculate

    result = calculate(bundle, root=root)
    if result["blocked"]:
        raise ValueError("NARRATIVE_TYPED_EVENT_REVIEW")
    _check_points(result["dimensions"], dimensions)
    raw = result["facts"]
    profile = bundle["typed_disclosure_calculation"]["profile"]
    window = raw.get("window", {})
    performance = raw.get("performance", window if "fund_return" in window else raw)
    facts["performance"] = {
        k: performance[k]
        for k in ("fund_return", "reference_return", "drawdown", "tracking_error", "months_at_or_above")
        if k in performance
    }
    pf = facts["performance"]
    if window.get("from") and window.get("through"):
        pf["period"] = f"{window['from']} 至 {window['through']}"
        if len(window.get("month_ends", [])) == 13:
            pf["month_count"] = 12
    if "maximum_drawdown" in raw:
        pf["drawdown"] = raw["maximum_drawdown"]
    if profile == "QDII_ACTIVE":
        pf["reference_label"] = "同地域人民币参照"
        facts["holdings"] = {"major_holdings_nav_fraction": raw.get("top10_nav_ratio")}
        own_flow = next((r for r in raw.get("share_flows", []) if r.get("share") == bundle["share_class"]), None)
        if own_flow:
            _add(
                facts,
                "liquidity",
                "本份额报告期累计赎回",
                number(own_flow["redeemed"]) / Decimal(100000000),
                "亿份",
                bundle["holding_date"],
            )
        for field, label in (
            ("stock_nav_ratio", "股票占净资产"),
            ("stock_total_ratio", "股票占总资产"),
            ("cash_nav_ratio", "合同认可现金占净资产"),
        ):
            ratio = number(raw.get("execution", {}).get(field))
            if ratio is not None:
                _add(facts, "efficiency", label, ratio * 100, "%", bundle["holding_date"])
    if profile == "BOND_INDEX":
        risk = raw.get("risk", {})
        if "maximum_drawdown" in risk:
            pf["drawdown"] = risk["maximum_drawdown"]
        # 基准缺失时不把仅重建的基金收益包装成可评分的相对收益。
    if profile in {"CONVERTIBLE", "MIXED"}:
        holding = raw.get("holdings", {})
        amount, net = number(holding.get("major_risk_concentration_cny")), number(holding.get("net_assets"))
        if amount is not None and net is not None and net > 0:
            _add(
                facts, "holdings", "主要股票及非国债持仓占基金净资产", amount / net * 100, "%", bundle["holding_date"]
            )
            assessment = _major_holdings_assessment(
                holding, bundle["share_class"], result["dimensions"].get("holdings")
            )
            if assessment:
                facts["dimension_assessments"]["holdings"] = assessment
        cash_upper = number(raw.get("execution", {}).get("qualified_cash_upper_nav_ratio"))
        if cash_upper is not None:
            _add(facts, "efficiency", "合同认可现金占净资产（最高估计）", cash_upper * 100, "%", bundle["holding_date"])
        if profile == "MIXED":
            rule = bundle["typed_disclosure_calculation"]["inputs"]["allocation_contract"]
            facts["dimension_assessments"]["efficiency"] = _mixed_execution_assessment(raw["execution"], rule)
        if result["dimensions"].get("management") is not None:
            management = _remediation_management_assessment(bundle["typed_disclosure_calculation"]["inputs"])
            if management:
                facts["dimension_assessments"]["management"] = management
        if result["dimensions"].get("liquidity") is not None:
            inputs = bundle["typed_disclosure_calculation"]["inputs"]
            liquidity = _holder_liquidity_assessment(inputs, raw)
            if liquidity:
                facts["dimension_assessments"]["liquidity"] = liquidity
                if net is not None and net > 0:
                    _add(
                        facts, "liquidity", "基金各类份额合计净资产",
                        net / Decimal(100000000), "亿元", bundle["holding_date"],
                    )
                _add(
                    facts, "liquidity", f"{bundle['share_class']}类报告期累计赎回",
                    number(raw["reported_redemption_shares"]) / Decimal(100000000),
                    "亿份", f"报告期截至{bundle['holding_date']}",
                )
    if profile == "GOLD":
        for field, label in (
            ("target_holding_nav_fraction", "目标ETF占净资产"),
            ("economic_gold_exposure_fraction", "两层黄金经济敞口"),
            ("net_non_gold_residual_budget_fraction", "非黄金残余预算"),
        ):
            ratio = number(raw.get("holdings", {}).get(field))
            if ratio is not None:
                _add(facts, "holdings", label, ratio * 100, "%", bundle["holding_date"])
    amount = raw.get("annual_cost_per_10000", raw.get("cost", {}).get("cost_cny"))
    if number(amount) is not None:
        facts["cost"] = {"annual_cost_per_10000": amount}
    facts["source_sha256"] = raw.get("source_sha256", [])
    # 目标执行、黄金经济敞口及赎回条款保留本类型已发布业务说明，禁止套股票匹配。
    return facts


def _add(facts, key, name, value, unit, period):
    if number(value) is not None and period:
        facts.setdefault("extra_metrics", {}).setdefault(key, []).append(
            {"name": name, "value": f"{number(value):.2f}", "unit": unit, "period": period}
        )


def r5_facts(bundle, *, facts):
    from app.services.fund_rating_r5 import normalize_packet

    packet = bundle["r5_calculation"]["packet"]
    nodes = normalize_packet(packet)["observations"]
    end = bundle["as_of_date"]
    value = nodes.get("return_1y", {}).get("target")
    if number(value) is not None:
        facts["performance"] = {"fund_return": value, "period": f"截至{end}的近一年"}
    for key, field, label, scale, unit in (
        ("return", "return_3y", "近三年年化总回报", 100, "%"),
        ("risk", "extra_drawdown", "相对基准额外回撤观测", 100, "%"),
        ("risk", "extra_downside", "相对基准下行偏差观测", 100, "%"),
        ("holdings", "issuer_concentration", "发行人集中度", 1, ""),
        ("holdings", "industry_concentration", "行业集中度", 1, ""),
        ("efficiency", "tracking_error", "年化跟踪误差", 100, "%"),
    ):
        value = number(nodes.get(field, {}).get("target"))
        if value is not None:
            _add(facts, key, label, value * scale, unit, f"本批考察期截至{end}")
    risk_paragraphs = []
    for field, label in (("extra_drawdown", "相对基准额外回撤"), ("extra_downside", "相对基准下行偏差")):
        value = number(nodes.get(field, {}).get("target"))
        if value is not None:
            risk_paragraphs.append(f"原评分采用的{label}观测为{value * 100:.2f}%。")
    if risk_paragraphs:
        facts["dimension_assessments"]["risk"] = {
            "conclusion": "风险评价使用相对基准的额外损失和下行波动观测。",
            "paragraphs": risk_paragraphs,
            "limitations": ["这些是原评分的相对风险指标，不能当作本基金净值最大回撤或未来亏损上限。"],
        }
    cost = number(nodes.get("cost_365", {}).get("target"))
    if cost is not None:
        facts["cost"] = {"annual_cost_per_10000": str(cost)}
    for field, title in (("monthly_success", "月度观察"), ("annual_success", "滚动年度观察")):
        rows = nodes.get(field, {}).get("rows", [])
        if rows and all(number(r.get("target")) is not None and number(r.get("reference")) is not None for r in rows):
            wins = sum(number(r["target"]) >= number(r["reference"]) for r in rows)
            _add(
                facts,
                "stability",
                f"{title}达到参照的窗口",
                wins,
                f"/{len(rows)}个",
                f"{rows[0].get('start', rows[0].get('end', end))} 至 {rows[-1].get('end', end)}",
            )
    paragraphs = []
    for key in ("experience", "governance"):
        text = business_text(nodes.get(key, {}).get("reason", ""))
        if key == "experience" and len(text) > 400:
            text = "已核本产品实际基金经理职责及任职履历；任职履历不直接代表未来投资能力。"
        if text:
            paragraphs.append(text)
    continuity = nodes.get("continuity", {})
    names = sorted({span.get("name") for span in continuity.get("intervals", []) if span.get("name")})
    if names:
        paragraphs.insert(0, "已核任职记录涉及" + "、".join(names) + "，分别按实际任职期间评价。")
    if paragraphs:
        facts["management"] = {"paragraphs": paragraphs}
    holder = nodes.get("holders", {})
    if holder.get("actual_execution_verified") and holder.get("share_flow_closed"):
        facts["liquidity"] = {
            "review_complete": True,
            "reported_redemption_shares": holder.get("actual_redemption_shares"),
            "reported_redemption_alert": bool(holder.get("confirmed_pressure_effects")),
        }
    payment = nodes.get("known_payment", {})
    cash = number(payment.get("cash_principal", payment.get("bank_principal")))
    liabilities = number(payment.get("all_book_liabilities"))
    for name, value in (("已核现金本金", cash), ("账面负债", liabilities)):
        if value is not None:
            _add(facts, "liquidity", name, value / Decimal(10000), "万元", bundle["holding_date"])
    return facts


def money_facts(bundle, *, facts):
    raw = bundle.get("money_inputs", {})
    liquidity = raw.get("liquidity", {})
    for name, key, scale, unit in (
        ("30日内资产占比", "assets_30day_ratio", 100, "%"),
        ("30日内负债占比", "liabilities_30day_ratio", 100, "%"),
        ("前十大持有人占比", "top10_holder_fraction", 100, "%"),
        ("普通活期本金", "ordinary_cash_principal", Decimal(".0001"), "万元"),
        ("账面负债", "all_liabilities", Decimal(".0001"), "万元"),
    ):
        value = number(liquidity.get(key))
        if value is not None:
            _add(facts, "liquidity", name, value * scale, unit, bundle["holding_date"])
    return facts
