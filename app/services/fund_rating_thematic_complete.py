"""主动主题完整包的19项原件编译及封存绑定；本模块不写数据库或发布。

保留有来源的人工主题/质量判断，期间、净值、持仓、行业、成交及账户金额
均重新读取原件。重大事项状态由完整已审事件底稿导出，绝不默认无事项。
"""

import importlib
import json
import re
from copy import deepcopy
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path

from app.services.fund_rating_active_execution import MONEY, _section
from app.services.fund_rating_cmf_liquidity import COMPATIBLE_TYPES, FOUNDATION_GATES
from app.services.fund_rating_cmf_liquidity import PROTOCOL as HOLDER_PROTOCOL
from app.services.fund_rating_direct_index import _flows
from app.services.fund_rating_management_inputs import calculate as management_calculate
from app.services.fund_rating_market_review import fee_profile
from app.services.fund_rating_r5_rules import holders_score
from app.services.fund_rating_reviewed_quality import DIMENSIONS, policy
from app.services.material_content import content_digest


def component(name):
    return importlib.import_module((__package__ + "." if __package__ else "") + "fund_rating_thematic_" + name)


_source = component("source")
plain, require, iso = _source.plain, _source.require, _source.iso
VERSION = "R5_THEMATIC_COMPLETE_ORIGINALS_20261029_1"
TIME_CONTRACT = "VALUATION_BY_ASOF_DISCLOSURE_BY_RESEARCH_FUNDAMENTALS_PUBLIC_BY_ASOF"
_COMPILED_CACHE = {}


def holders_from_primary(inputs, source, material):
    """原份额表逐列复算；20%是明示上界，未知最大户不补零。"""
    digest = inputs["primary_ref"]
    text, normalized = source.body(digest), source.plain(digest)
    flows, flow_proof = _flows([text])
    section = _section(text, r"8\.1\s*期末基金份额持有人", r"8\.2\s*")
    require(
        all(word in plain(section) for word in ["机构投资者", "个人投资者", "混合A", "混合C"]), "HOLDER_CLASS_COLUMNS"
    )
    # 只在本持有人结构表匹配整行六栏，不从其他章节搜索候选金额。
    row_pattern = (
        r"(?m)^\s*(?:合计\s+)?(\d{1,3}(?:,\d{3})*)\s+("
        + MONEY
        + r")\s+("
        + MONEY
        + r")\s+(\d+\.\d{2})\s+("
        + MONEY
        + r")\s+(\d+\.\d{2})\s*$"
    )
    rows = re.findall(row_pattern, section)
    require(len(rows) == 3 and len(flows) == 2, "HOLDER_STRUCTURE_FULL_TABLE")
    audit = []
    for i, (count, average, institution, ipct, individual, ppct) in enumerate(rows):
        inst, person = D(institution.replace(",", "")), D(individual.replace(",", ""))
        total = inst + person
        count = int(count.replace(",", ""))
        require(total > 0 and count > 0, "HOLDER_STRUCTURE_VALUES")
        require(abs(total / count - D(average.replace(",", ""))) <= D(".005"), "HOLDER_AVERAGE_ROUNDING")
        require(
            abs(inst / total * 100 - D(ipct)) <= D(".005") and abs(person / total * 100 - D(ppct)) <= D(".005"),
            "HOLDER_PERCENT_ROUNDING",
        )
        if i < 2:
            require(total == D(flows[i]["closing"]), "HOLDER_STRUCTURE_FLOW_RECONCILIATION")
        else:
            require(
                inst == sum(D(row["institution"]) for row in audit)
                and person == sum(D(row["individual"]) for row in audit),
                "HOLDER_ALL_CLASS_RECONCILIATION",
            )
            require(count == sum(row["count"] for row in audit), "HOLDER_COUNTS_RECONCILIATION")
        audit.append(
            {
                "share": ["A", "C", "ALL"][i],
                "count": count,
                "institution": str(inst),
                "individual": str(person),
                "total": str(total),
            }
        )
    single = _section(text, r"11\.1\s*报告期内单一投资者", r"11\.2\s*")
    require(
        "本基金本报告期内不存在持有基金份额比例达到或者超过20%的单一投资者的情况" in plain(single),
        "SINGLE_HOLDER_EXPLICIT_ABSENCE",
    )
    trustee = _section(text, r"5\.2\s*托管人对报告期内", r"5\.3\s*托管人")
    require(
        all(
            word in plain(trustee)
            for word in ["报告期", "监督", "复核", "未发现基金管理人有损害基金份额持有人利益的行为"]
        ),
        "ACTUAL_TRUSTEE_HOLDER_EXECUTION",
    )
    operation = re.search(r"本基金的基金管理人采用监控流动性受限资产比例[^。]*。", normalized)
    require(
        operation and all(word in operation[0] for word in ["压力测试", "开放日", "申购赎回", "可用现金头寸"]),
        "ACTUAL_REDEMPTION_MONITORING",
    )
    prospectus = inputs["fee"]["prospectus_ref"]
    source.original(prospectus)
    terms = source.plain(prospectus)
    require(
        plain(inputs["legal_name"]) in terms[:5000]
        and all(word in terms for word in ["T+7日", "巨额赎回", "延缓支付赎回款项"]),
        "HOLDER_APPLICABLE_REDEMPTION_MECHANISM",
    )
    require("契约型开放式" in normalized and "基金份额持有人银行账户" in terms, "HOLDER_LEGAL_OPEN_END_CASH_REDEMPTION")
    large_holder = re.search(
        r"在本基金出现巨额赎回且单个基金份额持有人的赎回申请超过上一开放日基金总份额的20%时.{1,1400}", terms
    )
    require(
        large_holder and all(word in large_holder[0] for word in ["可以延期办理", "取消赎回", "一并办理"]),
        "HOLDER_LARGE_INDIVIDUAL_REDEMPTION_MECHANISM",
    )
    effects = [
        e
        for e in material["events"]
        if e["applicability"] == "FUND_CONFIRMED"
        and e["status"] == "UNRESOLVED"
        and e["severity"] in {"MATERIAL", "SEVERE"}
    ]
    require(not effects, "HOLDER_KNOWN_EFFECT_REQUIRES_CLASSIFICATION")
    node = {
        "kind": "holder_execution_r5",
        "structure_closed": True,
        # 无单一>=20%只是公开上界，仍不能判定个体分散。保留全部相容
        # 结构，并由既有数学helper核两分支基础门槛及相同分值。
        "structure_type": None,
        "structure_condition_protocol": HOLDER_PROTOCOL,
        "compatible_structure_types": deepcopy(COMPATIBLE_TYPES),
        "structure_basis": "OPEN_END_CASH_LEGAL_PRODUCT_AND_COMPLETE_REPORTED_CATEGORY_TABLE",
        "foundation_gate_checks": {name: True for name in FOUNDATION_GATES},
        "share_flow_closed": True,
        "actual_execution_verified": True,
        "redemption_mechanism_verified": True,
        "confirmed_pressure_effects": [],
        "actual_redemption_shares": str(sum(D(row["redeemed"]) for row in flows)),
        "largest_single_holder_pct": None,
        "largest_holder_public_upper_exclusive": "20",
        "source_refs": [digest, prospectus],
    }
    holder_point, condition_proof = holders_score(node)
    return node, {
        "structure_original": section,
        "structure_recomputed": audit,
        "flows": flows,
        "flow_proof": flow_proof,
        "single_holder_original": single,
        "trustee_original": trustee,
        "actual_liquidity_original": operation[0],
        "large_holder_mechanism_original": large_holder[0],
        "structure_condition_point": str(holder_point),
        "structure_condition_proof": condition_proof,
    }


def management_from_originals(inputs, source, *, root):
    """既有管理入口复算后投影三个节点，职责reason必须来自同一原职责quote。"""
    raw = deepcopy(inputs["management"])
    raw["sources"] = {sha: source.sources[sha] for sha in raw["sources"]}
    for sha in raw["sources"]:
        source.original(sha)
    result = management_calculate(
        raw,
        root=root,
        manager_name=inputs["manager_name"],
        as_of=inputs["as_of_date"],
        fund_code=inputs["fund_code"],
        legal_name=inputs["legal_name"],
        primary_report_hash=inputs["primary_ref"],
    )
    require(result["score"] is not None, "MANAGEMENT_RESPONSIBILITY_INCOMPLETE")
    role_quotes = [
        claim["quote"]
        for claim in raw["claims"]
        if claim["source_hash"] in raw["experience"]["source_refs"] and "基金经理" in plain(claim["quote"])
    ]
    require(role_quotes, "EXPERIENCE_ROLE_REASON_ORIGINAL")
    return {
        "experience": {
            "kind": "state",
            **raw["experience"],
            "reason": "本产品原报告明示的经理职责，仅按基本履职档评价：" + "\n".join(role_quotes),
        },
        "continuity": {"kind": "responsibility_timeline", **raw["continuity"]},
        "governance": {
            "kind": "state",
            "state": raw["governance"]["state"],
            "source_refs": sorted(raw["sources"]),
            "reason": raw["governance"]["review_scope"],
        },
    }, result


def compile_observations(inputs, *, root, diagnostics=False):
    """每次重核原字节与日期，仅复用完全相同输入的已闭合确定性计算。"""
    root = Path(root).resolve()
    source = _source.RawArchive(inputs, root)
    key = (
        str(root),
        content_digest(json.dumps(inputs, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()),
    )
    if not diagnostics and key in _COMPILED_CACHE:
        return deepcopy(_COMPILED_CACHE[key])
    result = _compile_observations_uncached(inputs, source=source, root=root, diagnostics=diagnostics)
    if not diagnostics and result["complete"]:
        # 只保留一包最终事实；不持有大体量PDF原字节/解析缓存，也不缓存失败。
        _COMPILED_CACHE.clear()
        _COMPILED_CACHE[key] = deepcopy(result)
    return result


def _compile_observations_uncached(inputs, *, source, root, diagnostics):
    """按独立组件累计已闭合事实；完整模式下任一组件缺证即拒绝出包。"""
    root = Path(root).resolve()
    require(
        inputs.get("version") == VERSION and inputs.get("time_contract") == TIME_CONTRACT,
        "COMPLETE_VERSION_AND_TIME_CONTRACT",
    )
    require(
        all(
            inputs.get(key) == value
            for key, value in {
                "fund_code": "018590",
                "share_class": "C",
                "family": "ACTIVE_EQUITY",
                "basis": "PUBLIC_REVIEWED_ACTIVE_QUALITY_V2",
                "as_of_date": "2026-09-30",
                "holding_date": "2026-06-30",
            }.items()
        ),
        "COMPLETE_LEGAL_SCOPE",
    )
    require(
        not set(inputs) & {"score", "dimensions", "K", "HHI", "Q", "material_event_pending"},
        "COMPLETE_RAW_INPUT_REQUIRED",
    )
    nodes, audit, failures = {}, {}, {}

    def stage(name, function):
        try:
            result = function()
        except (ValueError, KeyError, TypeError) as error:
            if not diagnostics:
                raise
            failures[name] = str(error)
            return None
        audit[name] = result[1] if isinstance(result, tuple) else result
        if isinstance(result, tuple):
            nodes.update(result[0])
        return result

    with localcontext() as context:
        context.prec = 70
        stage("performance", lambda: component("performance").compile_performance(inputs, source))
        stage("structure", lambda: component("structure").compile_structure(inputs, source))
        execution = stage("execution", lambda: component("execution").compile_execution(inputs, source))
        if execution:
            stage("admission", lambda: component("admission").raw_admission(source, inputs, execution[1]))
        material = stage("material", lambda: component("material").compile_material_review(inputs, source))

        stage("management", lambda: management_from_originals(inputs, source, root=root))

        def fees():
            refs = inputs["fee"]
            summary = source.original(refs["summary_ref"])
            source.original(refs["prospectus_ref"])
            profile = fee_profile(
                source.body(refs["summary_ref"]),
                code=inputs["fund_code"],
                share_class=inputs["share_class"],
                legal_name=inputs["legal_name"],
                supporting_text=source.body(refs["prospectus_ref"]),
                published_on=summary["published_on"],
            )
            return {
                "cost_365": {"kind": "standard_cost", "fee_inputs": profile["fees"], "source_refs": list(refs.values())}
            }, profile

        stage("fees", fees)

        def liquidity():
            helper = importlib.import_module((__package__ + "." if __package__ else "") + "fund_rating_payment_domain")
            payment = helper.compile_payment_from_primary(inputs, root=root)
            digest = inputs["primary_ref"]
            balance = _section(source.body(digest), r"6\.1\s*资产负债表", r"6\.2\s*利润表")
            nav = re.search(r"净资产合计\s+(" + MONEY + r")", balance)
            require(nav, "PRIMARY_NAV")
            # payment helper已逐项核准五个受限小节及全资产其余可变现原句；不能只搜一个“无”。
            restricted = {
                "kind": "restricted",
                "restricted_amount": "0",
                "net_assets": nav[1].replace(",", ""),
                "scope_closed": True,
                "source_refs": [digest],
            }
            return {"known_payment": payment["node"], "restricted": restricted}, payment

        stage("accounting_liquidity", liquidity)
        if material:
            stage(
                "holders",
                lambda: (lambda result: ({"holders": result[0]}, result[1]))(
                    holders_from_primary(inputs, source, material)
                ),
            )
        else:
            failures["holders"] = "MATERIAL_APPLICABILITY_NOT_YET_CLOSED"
    if not diagnostics:
        expected = {name for terms in policy(inputs["basis"])["terms"].values() for name, _ in terms}
        require(
            not failures and set(nodes) == expected and material is not None and "admission" in audit,
            "COMPLETE_NINETEEN_RAW_NODES",
        )
    return {
        "version": VERSION,
        "observations": nodes,
        "anchors": material["anchors"] if material else None,
        "material_event_pending": material["material_event_pending"] if material else None,
        "audit": audit,
        "failures": failures,
        "used_source_refs": sorted(source.used),
        "complete": not failures and len(nodes) == 19,
    }


def build_packet(inputs, compiled):
    """仅由实际编译结果生成标准完整请求；不接收候选八数字或当前旧评分。"""
    require(compiled["complete"] and compiled["material_event_pending"] is False, "COMPLETE_PACKET_WITHHELD")
    nodes = deepcopy(compiled["observations"])
    for node in nodes.values():
        if not node.get("source_refs"):
            node["source_refs"] = compiled["used_source_refs"]
    refs = deepcopy(inputs["sources"])
    registry = {sha: {**ref, "path": ref["file"]} for sha, ref in refs.items()}
    raw_archive = [{**ref, "path": ref["file"], "kind": ref["archive_kind"]} for ref in refs.values()]
    labels = {
        "return": "收益",
        "risk": "风险",
        "efficiency": "投资执行",
        "stability": "持续性",
        "management": "管理",
        "holdings": "持仓",
        "liquidity": "流动性",
        "cost": "费用",
    }
    explanations = {
        "return": "比较本C份额一年收益及三年年化费后收益，与同主题、风险暴露相近的基金参照。",
        "risk": "结合历史相对回撤和下行波动观察风险；较高收益不能抵消已发生的回撤。",
        "efficiency": "结合已披露期末仓位、体育文化主题投入及投资范围，评价投资执行。",
        "stability": "观察完整月度及滚动年度收益相对参照的持续性；滚动年度窗口存在重叠。",
        "management": "结合经理职责、责任接续及近三年已公开治理事项，评价管理与策略。",
        "holdings": "结合股票发行人及行业集中程度、已披露资产质量事项；持仓为最近完整定期披露快照。",
        "liquidity": "结合市场承接、受限资产、本金与负债用途匹配及持有人申赎履职；报告日覆盖不保证未来赎回安全。",
        "cost": "按一万元持有本C份额365天的公开费用评价；实际投资成本还会受交易方式影响。",
    }
    packet = {
        "fund_code": inputs["fund_code"],
        "basis": inputs["basis"],
        "family": inputs["family"],
        "as_of_date": inputs["as_of_date"],
        "verified_at": inputs["verified_at"],
        "source_scope": "PUBLIC_DISCLOSURE_LOCAL_ANALYSIS",
        "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
        "quality_metrics": {
            "target_product_aliases": ["CN_PUBLIC_FUND:018590", "CN_PUBLIC_FUND:003397"],
            "material_event_pending": compiled["material_event_pending"],
            "observations": nodes,
        },
        "classification": {
            "product_id": "CN_PUBLIC_FUND:003397",
            "relationship_verified": True,
            "source_refs": [inputs["primary_ref"]],
            "effective_from": inputs["as_of_date"],
            "identity_bindings": [
                {
                    "source_ref": inputs["primary_ref"],
                    "required_text": ["018590", "003397", inputs["legal_name"], inputs["manager_name"]],
                }
            ],
        },
        "r4_evidence_registry": registry,
        "raw_archive": raw_archive,
        "thematic_complete_compilation": deepcopy(inputs),
        "quality_public": {
            "comparison": {
                "category": "同主题主动权益",
                "currency": "CNY",
                "productCount": 7,
                "scope": "收益7个独立产品；风险、持续性、结构和容量5个严格适配产品。",
                "period": "2023-09-28至2026-09-30",
            },
            "dimensions": [
                {"key": key, "name": labels[key], "score": None, "explanation": explanations[key]} for key in DIMENSIONS
            ],
            "strengths": [],
            "weaknesses": [],
            "dataDates": {
                "nav": inputs["as_of_date"],
                "holdings": inputs["holding_date"],
                "manager": inputs["as_of_date"],
                "fees": "2026-02-06",
            },
        },
    }
    return packet, deepcopy(compiled["anchors"])


def check_binding(packet, anchors, *, root):
    """标准calculate入口必须先调用；原始输入、观察与事项anchor一起重放。"""
    inputs = packet["thematic_complete_compilation"]
    require(
        all(inputs[key] == packet[key] for key in ["fund_code", "as_of_date", "verified_at", "basis", "family"]),
        "PACKET_SCOPE_CONFLICT",
    )
    compiled = compile_observations(inputs, root=root)
    return check_compiled_binding(packet, anchors, compiled)


def check_compiled_binding(packet, anchors, compiled):
    """纯比较阶段供完整入口及反例测试共用；正式入口先执行原件重编译。"""
    inputs = packet["thematic_complete_compilation"]
    expected, expected_anchors = build_packet(inputs, compiled)
    require(packet.get("unit_contract") == expected["unit_contract"], "ORIGINAL_OBSERVATION_UNIT_CONFLICT")
    require(
        packet["quality_public"]["dataDates"] == expected["quality_public"]["dataDates"],
        "ORIGINAL_OBSERVATION_DATE_CONFLICT",
    )
    require(packet["classification"] == expected["classification"], "ORIGINAL_PRODUCT_CLASSIFICATION_CONFLICT")
    require(expected["quality_metrics"] == packet["quality_metrics"], "ORIGINAL_OBSERVATIONS_CONFLICT")
    require(expected_anchors == anchors, "MATERIAL_ANCHORS_CONFLICT")
    return compiled


def prepare(session, request, *, root):
    """先规范归档全部原事实，再走既有prepare_complete；由调用方事务决定写入。

    外部请求仅给原件输入，观察及事项anchors在这里重新产生。标准路径重写
    覆盖主源图和管理组件的嵌套源图，避免正式根意外读取研究目录或旧副本。
    """
    from app.services.fund_rating_pipeline_prepare import canonical_source, prepare_complete

    require(request.get("fund_code") == "018590", "PREPARE_REQUEST_PRODUCT")
    inputs = deepcopy(request["inputs"])
    require(
        inputs["fund_code"] == request["fund_code"] and inputs["as_of_date"] == request["as_of_date"],
        "PREPARE_REQUEST_SCOPE",
    )
    canonical = {sha: canonical_source(root, ref) for sha, ref in inputs["sources"].items()}
    inputs["sources"] = canonical
    inputs["management"]["sources"] = {sha: canonical[sha] for sha in inputs["management"]["sources"]}
    result = compile_observations(inputs, root=root)
    packet, anchors = build_packet(inputs, result)
    return prepare_complete(session, packet, anchors, root=root)
