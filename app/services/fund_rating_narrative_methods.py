"""只说明正式结果采用的原有计分方法；不修改评分、权重或来源。"""

import json
import re
from decimal import Decimal
from pathlib import Path

from app.services.fund_rating_narrative import number

REVISION = "PLAIN_ALL_FUNDS_20261112_1"
PUBLIC = {"efficiency": "execution", "cost": "fee"}
LABELS = {
    "return_1y": "一年收益", "return_3y": "三年年化收益",
    "one_year": "一年收益", "three_year_annual": "三年年化收益",
    "extra_drawdown": "相对参照多出的回撤", "extra_downside": "相对参照多出的下跌波动",
    "monthly_success": "月度表现", "annual_success": "滚动年度表现",
    "monthly": "月度表现", "annual": "滚动年度表现",
    "experience": "实际任职职责", "continuity": "责任接续", "governance": "公开治理及履职",
    "allocation": "合同配置", "mandate": "投资范围及策略", "scope": "投资范围", "configuration": "合同配置",
    "issuer_concentration": "发行人集中程度", "industry_concentration": "行业集中程度", "asset_quality": "资产质量",
    "issuer_replication": "发行人权重匹配", "industry_replication": "行业权重匹配",
    "market_capacity": "持仓卖出承接能力", "capacity": "持仓卖出承接能力",
    "restricted": "资产受限情况", "restriction": "资产受限情况",
    "known_payment": "现金与已知付款义务的匹配", "payment": "现金与已知付款义务的匹配",
    "holders": "持有人结构及实际申赎", "maturity_structure": "短期资产与负债的匹配",
    "security_transfer_restriction": "证券受限情况", "cash_liability_match": "现金与负债的匹配",
    "holder_concentration": "持有人集中程度", "numeric_observations": "各期数值要求",
    "investment_categories": "投资品种要求",
    "tracking_error": "跟踪误差", "cost_365": "公开费用",
}
PURPOSE = {
    "return": "本项看本份额的实际收益与适用参照相比怎样。",
    "risk": "本项看已发生的下跌及相对参照的风险差异。",
    "efficiency": "本项按本基金的合同目标和投资要求核对执行情况。",
    "stability": "本项看相对参照的表现能否在不同观察区间持续出现。",
    "management": "本项看实际任职职责、责任接续及公开治理和履职情况。",
    "holdings": "本项按本基金类型检查持仓的集中、匹配或资产结构。",
    "liquidity": "本项看持仓变现、资金用途和实际申赎方面已核实的情况。",
    "cost": "本项看本份额在统一持有情景下的公开费用。",
}


def _point(value):
    if isinstance(value, dict):
        value = value.get("candidate", value.get("point"))
    return number(value)


def _weighted(parts, terms, expected):
    """只展示能与正式分数对上的分项，不靠正式分数倒推分项。"""
    values = [(_point(parts.get(key)), weight, LABELS.get(key, key)) for key, weight in terms]
    if any(value is None for value, _, _ in values):
        return None
    total = sum(value * Decimal(str(weight)) / 100 for value, weight, _ in values)
    if abs(total - expected) > Decimal(".005"):
        return None
    expression = "＋".join(f"{label}{value:.2f}×{weight}%" for value, weight, label in values)
    return f"本项目按原有分项占比合成：{expression}，得{expected:.2f}分；计算使用未四舍五入的原值。"


def _management_parts(bundle):
    from app.services.fund_rating_r5_rules import continuity_score
    from app.services.fund_rating_reviewed_quality import STATES

    node = bundle.get("reviewed_document", {}).get("dimensions", {}).get("management", {})
    if isinstance(node, dict) and node.get("subcomponents"):
        return node["subcomponents"]
    inputs = bundle.get("evidence_v1_calculation", {}).get("inputs", {})
    inputs = bundle.get("typed_disclosure_calculation", {}).get("inputs", inputs)
    raw = inputs.get("management_inputs") or bundle.get("management_calculation", {}).get("inputs")
    if raw:
        result = {key: STATES.get(raw.get(key, {}).get("state")) for key in ("experience", "governance")}
        if raw.get("continuity"):
            result["continuity"] = continuity_score(raw["continuity"])[0]
        return result
    review = inputs.get("manager_review", {})
    if review.get("experience_state") and review.get("governance_state"):
        return {"experience": STATES.get(review["experience_state"]), "continuity": 75,
                "governance": STATES.get(review["governance_state"])}
    reviewed = inputs.get("reviewed_management_facts", {})
    if len(reviewed.get("states", [])) == 3:
        return dict(zip(("experience", "continuity", "governance"),
                        (STATES.get(state) for state in reviewed["states"]), strict=True))
    return {}


def _legacy_components(bundle, dimensions, facts, needed=None):
    """只读取冻结说明明确引用的同产品旧底稿，并核对原点值；路径由D盘读取层处理。"""
    from app.services.material_content import content_digest, source_bytes, source_size

    sources = bundle.get("reviewed_document", {}).get("sources", {})
    if not isinstance(sources, dict):
        return {}
    result = {}
    for name, key in (("holdings", "holdings"), ("management", "management")):
        if needed is not None and key not in needed:
            continue
        ref = sources.get(name)
        point = number(dimensions.get(key))
        if point is None or not isinstance(ref, str) or bundle["fund_code"] not in ref:
            continue
        try:
            if source_size(Path(ref)) > 25 * 1024 * 1024:
                raise ValueError("NARRATIVE_COMPONENT_SIZE")
            content = source_bytes(Path(ref))
        except FileNotFoundError:
            continue
        if len(content) > 25 * 1024 * 1024:
            raise ValueError("NARRATIVE_COMPONENT_SIZE")
        document = json.loads(content)
        if document.get("code") != bundle["fund_code"]:
            raise ValueError("NARRATIVE_COMPONENT_IDENTITY")
        original = number(document.get("dimension_candidate"))
        if original is None or abs(original - point) > Decimal(".005"):
            continue
        if key == "holdings":
            parts = {"issuer_concentration": document.get("issuer_candidate"),
                     "industry_concentration": document.get("industry_candidate"),
                     "asset_quality": document.get("asset_quality_candidate")}
            terms = [("issuer_concentration", 35), ("industry_concentration", 35), ("asset_quality", 30)]
        else:
            parts = {key: document.get(key) for key in ("experience", "continuity", "governance")}
            terms = [("experience", 40), ("continuity", 20), ("governance", 40)]
        text = _weighted(parts, terms, point)
        if text:
            result[key] = text
            facts.setdefault("source_sha256", []).append(content_digest(content))
            if key == "holdings" and isinstance(document.get("stocks_count"), int):
                facts.setdefault("dimension_assessments", {})[key] = {
                    "conclusion": "本项分别看发行人集中、行业集中和资产质量。",
                    "paragraphs": [
                        f"原持仓底稿核对了{document['stocks_count']}只股票。发行人集中看资金是否集中于同一企业，"
                        "行业集中看是否集中于同一行业；资产质量按已有逐项公开依据评价。"
                    ],
                    "limitations": ["这是已核公开持仓和原有分类方法的结果，不代表未来收益或所有风险已排除。"],
                }
            elif key == "management" and isinstance(document.get("report_count"), int):
                facts.setdefault("dimension_assessments", {})[key] = {
                    "conclusion": "本项看实际任职职责、责任接续和公开治理及履职。",
                    "paragraphs": [
                        f"原管理底稿核对了{document['report_count']}份定期报告，再将任职职责、责任接续、"
                        "治理及履职分别计分。"
                    ],
                    "limitations": ["这些分值反映已核资料，不能直接当作基金经理的投资能力等级。"],
                }
    return result


def _standard(bundle, facts, dimensions):
    result = {}
    typed = bundle.get("typed_disclosure_calculation", {})
    profile = typed.get("profile")
    evidence = bundle.get("evidence_v1_calculation", {})
    ordinary = bool(evidence) or profile in {"MIXED", "CONVERTIBLE", "QDII_ACTIVE", "GOLD", "BOND_INDEX"}
    performance = facts.get("performance", {})
    if not ordinary:
        return result
    if number(dimensions.get("return")) is not None and number(performance.get("reference_return")) is not None:
        result["return"] = (
            "本项目以75分为起点，比参照收益高则加分、低则扣分。设收益差为d个百分点，"
            "原换分公式为75＋25×d÷√(d²＋10²)，结果限制在0至100分。"
        )
    drawdown = number(performance.get("drawdown"))
    if number(dimensions.get("risk")) is not None and drawdown is not None:
        result["risk"] = (
            f"本项目用100减去最大回撤的百分数计分：100－{drawdown * 100:.2f}＝{number(dimensions['risk']):.2f}分。"
        )
    wins, months = performance.get("months_at_or_above"), performance.get("month_count")
    if number(dimensions.get("stability")) is not None and isinstance(wins, int) and months:
        result["stability"] = (
            f"本项目按达到或超过参照的月份比例计分：{wins}÷{months}×100＝{number(dimensions['stability']):.2f}分。"
        )
    if number(dimensions.get("efficiency")) is not None:
        if bundle["family"] == "INDEX" and profile != "BOND_INDEX":
            result["efficiency"] = (
                "本项目将年化跟踪误差与固定的1%换分基准比较：得分＝300×1%÷（3×1%＋实际误差）。"
                "误差越小得分越高；这里的1%是计分基准，合同目标另行核对。"
            )
        elif profile == "MIXED":
            result["efficiency"] = (
                "本项目对股票、现金和总资产要求分别计分：满足为75分，未满足为0分，"
                "再按40%、40%、20%合成。这三个百分比是计分占比，不是合同要求的投资比例。"
            )
        elif profile == "QDII_ACTIVE":
            result["efficiency"] = (
                "本项目核对合同股票范围、认可现金及其他已列核心要求；必要条件均有资料支持时固定给75分。"
            )
        elif evidence.get("inputs", {}).get("execution_public_observations"):
            result["efficiency"] = (
                "本项目将合同配置与策略执行按60%、40%合成；已核的连续现金缺口使配置项取25分，"
                "策略执行项为75分，即25×60%＋75×40%＝45分。这不直接构成法律违规定性。"
            )
        else:
            result["efficiency"] = "本项目核对股票范围、认可现金及已列策略要求；必要条件均有资料支持时固定给75分。"
    holding = facts.get("holdings", {})
    if number(dimensions.get("holdings")) is not None:
        if profile == "GOLD":
            result["holdings"] = (
                "本项目将本基金与目标基金两层合计的黄金比例、非黄金比例，与原配置目标逐项比较；"
                "得分＝100×［1－两项绝对偏差之和÷2］，最低为0分。它衡量配置匹配，不衡量资产是否分散。"
            )
        elif profile == "BOND_INDEX":
            result["holdings"] = (
                "本项目用已核符合目标债券类别的金额，除以本项核对的全部债券金额，再乘100计分；"
                "这不等于全部债券逐只成分资格、权重和跟踪情况均已核准。"
            )
        elif profile in {"MIXED", "CONVERTIBLE"}:
            pass  # 这两类已有带实际数值的计分解释。
        elif bundle["family"] == "INDEX":
            result["holdings"] = "本项目按已核主要持仓中属于目标指数成分的金额占比计分：符合金额÷所核主要持仓金额×100。"
        else:
            ratio = number(holding.get("major_holdings_nav_fraction"))
            result["holdings"] = (
                f"本项目用100减去前十大股票占净资产的百分数计分：100－{ratio * 100:.2f}＝"
                f"{number(dimensions['holdings']):.2f}分。高分只说明这批主要股票占比较低。"
                if ratio is not None else "本项目按已核主要股票占净资产的比例，采用100减去该百分数的方法计分。"
            )
    if number(dimensions.get("liquidity")) is not None:
        result["liquidity"] = (
            "本项目核对合同赎回安排、实际申赎及托管等资料；必要资料已核对，"
            "且没有已确认的赎回受影响记录时固定给75分，没有按逐笔到账速度计分。"
        )
    management = number(dimensions.get("management"))
    if management is not None:
        result["management"] = _weighted(
            _management_parts(bundle), [("experience", 40), ("continuity", 20), ("governance", 40)], management
        ) or "本项目按实际任职职责40%、责任接续20%、公开治理及履职40%合成，按各项已有资料确定分值。"
    return result


def _reviewed(bundle, dimensions):
    nodes = bundle.get("reviewed_document", {}).get("dimensions", {})
    result = {}
    terms = {
        "return": [("one_year", 30), ("three_year_annual", 70)],
        "risk": [("extra_drawdown", 70), ("extra_downside", 30)],
        "stability": [("monthly", 30), ("annual", 70)],
        "management": [("experience", 40), ("continuity", 20), ("governance", 40)],
        "liquidity": [("capacity", 40), ("restriction", 20), ("payment", 25), ("holders", 15)],
    }
    money = bundle["family"] == "MONEY"
    if money:
        terms["liquidity"] = [("maturity_structure", 40), ("security_transfer_restriction", 20),
                              ("cash_liability_match", 25), ("holder_concentration", 15)]
        terms["efficiency"] = [("numeric_observations", 60), ("investment_categories", 40)]
    for key, value in dimensions.items():
        point = number(value)
        if point is None:
            continue
        node = nodes.get(PUBLIC.get(key, key), {})
        parts = node.get("subcomponents", {}) if isinstance(node, dict) else {}
        if key == "management" and not parts:
            parts = _management_parts(bundle)
        if key in terms:
            combined = _weighted(parts, terms[key], point)
            if combined:
                result[key] = combined
    if number(dimensions.get("return")) is not None and "return" not in result:
        result["return"] = (
            "本项目分别比较一年收益和三年年化收益，将其与同类适用参照的差距换为分数，再按30%、70%合成。"
            "不是直接把收益百分比当作分数。"
        )
    if number(dimensions.get("risk")) is not None and "risk" not in result:
        result["risk"] = (
            "本项目将季度估值偏离中最不利的一次和典型季度分别与已核参照比较，按70%、30%合成；不使用接近不变净值的零回撤。"
            if money else
            "本项目将相对参照多出的回撤、下跌波动分别换分，再按70%、30%合成；它不等于100减去基金绝对回撤。"
        )
    if number(dimensions.get("stability")) is not None and "stability" not in result:
        result["stability"] = (
            "本项目逐个比较月度及滚动年度收益与适用参照的差距，按原定差距范围换分，再按30%、70%合成。"
            "滚动年度会重叠，不能视为彼此独立的年份。"
        )
    if number(dimensions.get("efficiency")) is not None and "efficiency" not in result:
        if bundle.get("huaan_qdii_benchmark_calculation"):
            result["efficiency"] = (
                "本项目用三年实际年化跟踪误差换分：得分＝300×1%÷（3×1%＋实际误差）。"
                "固定1%用于计分，合同4%目标只用于单独比较是否达到目标。"
            )
        else:
            result["efficiency"] = (
                "本项目将投资范围和配置要求分别按原规则核对，再按原有占比合成；资料支持正常执行的分项取75分。"
            )
    if number(dimensions.get("holdings")) is not None and "holdings" not in result:
        if bundle.get("huaan_qdii_disclosure_calculation"):
            result["holdings"] = (
                "本项目按目标基金报告前十大股票中，属于同日目标指数成分的金额占比计分，符合金额÷所核金额×100。"
            )
    if number(dimensions.get("liquidity")) is not None and "liquidity" not in result and not money:
        if bundle.get("huaan_qdii_disclosure_calculation"):
            result["liquidity"] = (
                "本项目在合同赎回安排、实际公开申赎和托管履职资料核对通过，且没有已确认的赎回受影响记录时固定给75分。"
            )
        else:
            result["liquidity"] = (
                "本项目将持仓卖出承接能力、资产受限情况、现金与已知付款义务的匹配、持有人及实际申赎，"
                "按40%、20%、25%、15%合成；本次冻结结果未逐项展开全部原始数值，不从总分倒推。"
            )
    return result


def _r5(bundle, dimensions, root):
    from app.services.fund_rating_r5 import calculate
    from app.services.fund_rating_reviewed_quality import policy

    raw = bundle["r5_calculation"]
    result = calculate(raw["packet"], raw["anchors"], root=root)
    terms = policy(raw["packet"]["basis"])["terms"]
    methods = {}
    for key, point in dimensions.items():
        if number(point) is None:
            continue
        text = _weighted(result["subscores"], terms[key], number(point))
        if not text:
            raise ValueError("NARRATIVE_METHOD_SCORE_MISMATCH:" + key)
        methods[key] = text
    methods["return"] += "各期限先比较本份额收益与已核参照中间水平的差距，以75分为起点，领先加分、落后扣分。"
    methods["risk"] += "两个分项同时考虑相对参照的差距和实际风险大小，不能当作基金绝对最大回撤。"
    methods["stability"] += "每个窗口超过原定领先范围取100分、落后超出范围取50分，其余取75分，先在各期限内平均。"
    if bundle["family"] == "INDEX":
        methods["efficiency"] += "年化误差越小得分越高；按固定1%计分基准换分，合同目标另行核对。"
    return methods


def attach(bundle, dimensions, facts, *, root):
    """随同批次资料保存解释；页面只读，不现场重算。"""
    if bundle.get("r5_calculation"):
        methods = _r5(bundle, dimensions, root)
    else:
        methods = _standard(bundle, facts, dimensions)
        if not methods:
            methods = _reviewed(bundle, dimensions)
            methods.update(_legacy_components(
                bundle, dimensions, facts, needed={key for key in ("holdings", "management") if key not in methods}
            ))
    cost = number(dimensions.get("cost"))
    if cost is not None:
        amount = number(facts.get("cost", {}).get("annual_cost_per_10000"))
        if amount is not None and abs(max(Decimal(0), 100 - amount / 4) - cost) > Decimal(".005"):
            raise ValueError("NARRATIVE_METHOD_COST_MISMATCH")
        methods["cost"] = (
            "本项目按一万元、持有365天、净值不变的情景估算本份额公开费用，"
            "每4元费用扣1分，即100－费用金额÷4，分数限制在0至100分。"
        )
    for key, value in dimensions.items():
        if number(value) is None:
            methods[key] = "本项尚缺确定分数所需的资料或适用计算依据，保留已知范围，不把未知部分补成0分。"
        elif key not in methods and key not in {"holdings"}:
            methods[key] = f"本项沿用已核分项及原有合成方法，得{number(value):.2f}分；不从这个分数反推缺失的原始数据。"
    facts["score_methods"] = methods
    facts["explanation_revision"] = REVISION
    facts["dimension_purposes"] = dict(PURPOSE)
    if bundle["family"] == "MONEY":
        facts["dimension_purposes"].update({
            "return": "本项看实际分配收益与同类货币基金相比怎样。",
            "risk": "本项看报告估值与另一种核对估值之间的偏离，反映估值差异。",
        })
    return facts


def tidy(assessment, method=None):
    """去掉结论与正文重复的整句，保留数据、条件和不同含义的日期。"""
    conclusion = _plain(assessment["conclusion"]).strip()
    seen = {re.sub(r"\s+", "", conclusion)}
    result = {"conclusion": conclusion}
    for field in ("paragraphs", "limitations"):
        texts = list(assessment.get(field, []))
        has_method = any(method in text for text in texts) if method else False
        has_fixed = method and "固定给75分" in method and any("固定给75分" in text for text in texts)
        if field == "paragraphs" and method and not has_method and not has_fixed:
            texts.append(method)
        result[field] = []
        for text in texts:
            text = _plain(text).strip()
            if conclusion and text.startswith(conclusion):
                text = text[len(conclusion):].lstrip()
            identity = re.sub(r"\s+", "", text)
            if identity and identity not in seen:
                result[field].append(text)
                seen.add(identity)
    return result


def _plain(text):
    """只替换旧底稿的内部简称，数值、日期、必要条件和缺项继续保留。"""
    if re.fullmatch(r"原\d+/\d+[。；]?", text.strip()):
        return ""
    replacements = {
        "CB/EB": "可转债及可交换债", "GICS分类": "行业分类", "GICS": "行业分类", "全仓": "全部持仓",
        "法律产品": "整只基金", "全产品": "整只基金", "补件": "补充核实",
        "尚未闭合": "尚未核对清楚", "未全闭合": "尚未全部核对清楚", "未闭合": "尚未核对清楚",
        "原经验50、连续75保留": "任职职责和责任接续已有记录",
        "原30/70": "按原有两项占比合成", "原70/30": "按原有两项占比合成",
        "不拿三个项目数量套股票HHI": "不能仅凭三个项目的数量判断风险是否分散",
        "36/25": "36个月度、25个滚动年度区间",
        "销售服务费返还沿用R41核准值": "销售服务费返还按此前已核数值处理",
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
    return re.sub(r"按适用的XBRL第5号模板脚注[\d/]+及本基金同格式披露", "按已核公开披露模板的费用定义", text)
