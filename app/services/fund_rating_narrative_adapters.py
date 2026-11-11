"""各类正式评分的展示适配；复用已冻结业务事实，不从分数倒推指标。"""

import re
from decimal import Decimal

from app.services.fund_rating_narrative import number

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
        explanation = business_text(explanations.get(public, ""))
        texts = list(dict.fromkeys(t for t in (explanation, reason) if t))
        candidate = node.get("candidate") if isinstance(node, dict) else node
        if number(candidate) is None:
            conclusion = f"{TITLES[internal]}尚无确定分，已知事实与缺项分别保留。"
        elif texts:
            first = re.split(r"[。；]", max(texts, key=len))[0]
            conclusion = (
                first + "。" if len(first) <= 100 else f"{TITLES[internal]}已有评分，以下为已核事实和适用范围。"
            )
        else:
            conclusion = f"{TITLES[internal]}已有评分，现有资料尚不足以解释全部影响因素。"
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
                facts, "holdings", "主要股票及非政府债券占全产品净资产", amount / net * 100, "%", bundle["holding_date"]
            )
        cash_upper = number(raw.get("execution", {}).get("qualified_cash_upper_nav_ratio"))
        if cash_upper is not None:
            _add(facts, "efficiency", "合格现金占净资产的已核上界", cash_upper * 100, "%", bundle["holding_date"])
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
