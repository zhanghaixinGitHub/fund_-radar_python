"""013180最后流动性维度的原件约束区间候选；只读、无完整分、无发布。

只复用本基金既已冻结的同指数联接原件图和原R5公式，不借其他基金账户事实。
已知容量和持有人由原件重新编译；全资产限制及可用银行本金仍未知，保留域。
"""

import re
from copy import deepcopy
from datetime import UTC, date, datetime
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path

from app.services.fund_quality_v2_accounting import accounting_observations
from app.services.fund_quality_v2_restrictions import full_asset_restriction_scope, restricted_securities
from app.services.fund_quality_v2_stock_book import _stock_parser_view, periodic_financial_view
from app.services.fund_rating_direct_index import Originals, _flows, _liabilities
from app.services.fund_rating_direct_index import compile_observations as compile_frozen
from app.services.fund_rating_inputs import digest
from app.services.fund_rating_market_review import normalized_text
from app.services.fund_rating_passive_portfolio import portfolio
from app.services.fund_rating_r5 import PROTOCOL, exposure_factor, peer_gap_score
from app.services.fund_rating_r5_rules import holders_score
from app.services.fund_rating_reviewed_quality import PASSIVE_BASIS, policy
from app.services.material_content import document_pdfium

VERSION = "R5_GF_BATTERY_LIQUIDITY_SOURCE_BOUNDED_INTERVAL_20261030_1"
FUND, ASOF, HOLDING = "013180", "2026-09-30", "2026-06-30"
LEGAL = "广发国证新能源车电池交易型开放式指数证券投资基金发起式联接基金"
MANAGER = "广发基金管理有限公司"
PRIMARY = "f7ad041c6c36a6beaec0fa9cc52189927277e43d6be50e456985010fdb0b8643"
OWN_PDF = "04f446cd219e126c3780e13fcceb67e8bd98d009d31bdab8a190a86e114f93e2"
ETF_PDF = "e1ac7702503a7d78369d9a696b51a4701a3288c3be1c4443428b0ff028ee43ff"
LEGAL_PDF = "6c9ae39f5c4e5ebd3e655a19aa149bae01b29d02b2aa9535d9b33ef7de4ed726"
WEIGHTS = {"market_capacity": 40, "restricted": 20, "known_payment": 25, "holders": 15}
PACKAGE = "gf_battery_liquidity_interval_calculation"


def require(ok, code):
    if not ok:
        raise ValueError("R5_GF_INTERVAL_" + code)


def section(text, start, stop):
    """只选正文最后一个完整节，不把目录或后继章节当成原事实。"""
    begin = text.rfind(start)
    end = text.find(stop, begin + len(start))
    require(begin >= 0 and end > begin, "SECTION:" + start)
    return text[begin:end]


def category_table(text, flows):
    """核A/C完整持有人类别表；户数/均值不作为集中程度的替代值。"""
    body = section(text, "8.1期末基金份额持有人户数及持有人结构", "8.2期末基金管理人的从业人员持有本基金的情况")
    integer = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
    money = integer + r"\.\d{2}"
    pattern = rf"发起联接([AC])({integer})({money})({money})(\d+\.\d{{2}})%({money})(\d+\.\d{{2}})%"
    matches = re.findall(pattern, body)
    require(len(matches) == 2 and {r[0] for r in matches} == {"A", "C"}, "COMPLETE_CATEGORY_COLUMNS")
    closing = {r["share"]: D(r["closing"]) for r in flows}
    rows = []
    for share, count, average, institution, institution_pct, individual, individual_pct in matches:
        values = [D(x.replace(",", "")) for x in [institution, individual]]
        require(sum(values) == closing[share], "CATEGORY_FLOW_CONSERVATION:" + share)
        require(D(institution_pct) + D(individual_pct) == 100, "CATEGORY_PERCENT_COLUMNS:" + share)
        rows.append(
            {
                "share": share,
                "holders": count,
                "average_shares": average,
                "institution_shares": str(values[0]),
                "individual_shares": str(values[1]),
                "institution_pct": institution_pct,
                "individual_pct": individual_pct,
            }
        )
    return rows


def pdf_text(source, sha):
    """从本轮已校验的字节提取原PDF文本，不再按路径另读一份内容。"""
    from contextlib import closing

    from app.core.pdf import PDF_LOCK

    with PDF_LOCK, document_pdfium(source.raw[sha]) as document:
        pages = []
        for index in range(len(document)):
            with closing(document[index]) as page, closing(page.get_textpage()) as text:
                pages.append(text.get_text_range())
    return pages


def original_scope(inputs, root):
    """每次均核源字节、份额/报告/研究日期和已冻结同类图；缓存不可免核。"""
    require(
        inputs.get("version") == VERSION
        and inputs.get("fund_code") == FUND
        and inputs.get("share_class") == "C"
        and inputs.get("as_of_date") == ASOF
        and inputs.get("holding_date") == HOLDING
        and inputs.get("legal_name") == LEGAL
        and inputs.get("manager_name") == MANAGER
        and inputs.get("primary_ref") == PRIMARY,
        "IDENTITY_OR_PERIOD",
    )
    require(
        not set(inputs) & {"score", "lower", "upper", "candidate", "dimensions", "observations", "bank_principal"},
        "RAW_INPUT_REQUIRED",
    )
    research = date.fromisoformat(inputs["research_on"])
    require(date.fromisoformat(ASOF) <= research <= datetime.now(UTC).astimezone().date(), "RESEARCH_DATE")
    base = inputs["base_compilation"]
    require(
        digest(base) == inputs["base_compilation_sha256"]
        and base.get("component_scope") == "NON_LIQUIDITY_SEVEN_DIMENSIONS"
        and base["fund_code"] == FUND
        and base["share_class"] == "C"
        and base["as_of_date"] == ASOF
        and base["holding_date"] == HOLDING
        and base["target_refs"]["half"] == PRIMARY,
        "FROZEN_SEVEN_INPUT_BINDING",
    )
    require(set(inputs["sources"]) == set(base["sources"]) | {OWN_PDF, ETF_PDF, LEGAL_PDF}, "SOURCE_GRAPH")
    require(all(inputs["sources"][h] == r for h, r in base["sources"].items()), "FROZEN_SOURCE_METADATA")
    require(
        dict(policy(PASSIVE_BASIS)["terms"]["liquidity"]) == WEIGHTS,
        "ORIGINAL_WEIGHTS_CHANGED",
    )
    for h in [OWN_PDF, ETF_PDF, LEGAL_PDF]:
        ref = inputs["sources"][h]
        require(ref.get("published_on") and ref["published_on"] <= ASOF, "OFFICIAL_PUBLICATION_DATE")
        require(datetime.fromisoformat(ref["acquired_at"]).date() <= research, "SOURCE_AFTER_RESEARCH")
    return Originals(inputs, Path(root))


def compile_observations(inputs, *, root):
    """保留两项未知域；证券限制下界不冒充全资产点值，银行本金不冒充可用额。"""
    source = original_scope(inputs, root)
    base = inputs["base_compilation"]
    # 日期和逐股完整性仍由原编译器核；另核每个原HTTP响应成功且分页闭合，
    # 避免在来源容器未完整取得时把可解析的行误当成完整60日成交总体。
    market_audit = []
    for day, sha in base["market"]["receipts"].items():
        response, ref = source.json(sha), source.sources[sha]
        data = response.get("data", {})
        fields, rows = data.get("fields"), data.get("items")
        require(
            response.get("code") == 0
            and data.get("has_more") is False
            and isinstance(fields, list)
            and len(fields) == len(set(fields))
            and isinstance(rows, list)
            and all(len(r) == len(fields) for r in rows),
            "MARKET_RESPONSE_OR_PAGINATION:" + day,
        )
        status = ref.get("status", ref.get("original_receipt", {}).get("http_status"))
        # 旧原回执有一段只保留业务响应及取得时刻，没有HTTP状态字段；
        # 不把缺字段伪写200，也不以本次新增日志要求否定已取得的完整成功响应。
        require(status is None or status == 200, "MARKET_HTTP_STATUS:" + day)
        market_audit.append(
            {
                "day": day,
                "source_hash": sha,
                "http_status": status,
                "transport_status_preserved_or_unknown": True,
                "response_business_code": response["code"],
                "has_more": data["has_more"],
                "row_count": len(rows),
                "provider_count_original": data.get("count"),
                "provider_count_semantics_not_assumed_row_cardinality": True,
            }
        )
    # 原标准编译器仍实际核冻结同类、全窗合同、60日成交及穿透股票风险预算。
    # NON_LIQUIDITY作用域只移除容量节点，audit.performance保留同次原件计算的K。
    replay = compile_frozen(base, root=Path(root))
    values = replay["audit"]["performance"]
    peers = base["cohort"]["peers"]
    require(peers == ["015874"] and set(values) == {FUND, *peers}, "FROZEN_PEER_SET")
    require(len(base["market"]["dates"]) == 60 and base["market"]["dates"][-1] == ASOF, "CAPACITY_WINDOW")
    capacity_node = {
        "kind": "rank",
        "direction": "lower",
        "target": values[FUND]["market_capacity"],
        "expected_products": ["CN_PUBLIC_FUND:" + code for code in peers],
        "peers": [{"product_id": "CN_PUBLIC_FUND:" + code, "value": values[code]["market_capacity"]} for code in peers],
        "source_refs": sorted(base["sources"]),
    }
    own_pages = source.pages(PRIMARY)
    own = portfolio(own_pages, holding_date=HOLDING)
    etf_ref = base["etf_refs"][FUND]
    require(etf_ref["fund_code"] == "159755", "ACTUAL_TARGET_ETF")
    target_pages = source.pages(etf_ref["source_hash"])
    target = portfolio(target_pages, holding_date=HOLDING)
    view, _ = periodic_financial_view(target_pages)
    numeric_view, layout = _stock_parser_view(view, target["positions"])
    securities = restricted_securities(numeric_view, target["positions"], target["balance"]["bond_amount"])
    official_own = normalized_text("\n".join(pdf_text(source, OWN_PDF)))
    official_etf = normalized_text("\n".join(pdf_text(source, ETF_PDF)))
    legal = normalized_text("\n".join(pdf_text(source, LEGAL_PDF)))
    own_text = normalized_text("\n".join(own_pages))
    require(LEGAL in official_own[:18000] and FUND in official_own[:18000], "OFFICIAL_C_SHARE_IDENTITY")
    require("159755" in official_etf[:18000] and etf_ref["legal_name"] in official_etf[:18000], "OFFICIAL_ETF_IDENTITY")
    require("2026年6月30日" in official_own[:20000] and "2026年6月30日" in official_etf[:20000], "OFFICIAL_PERIOD")
    weak_scope = "本基金所持大部分投资标的为具有良好流动性的金融工具"
    require(weak_scope in official_own and weak_scope in official_etf, "ACTUAL_SCOPE_WORDING_CHANGED")
    require(
        not full_asset_restriction_scope(official_own) and not full_asset_restriction_scope(official_etf),
        "UNKNOWN_SCOPE_MUST_NOT_BE_CLEARED",
    )
    official_restrictions = section(
        official_etf,
        "6.4.12.1因认购新发/增发证券而于期末持有的流通受限证券",
        "6.4.12.2期末持有的暂时停牌等流通受限股票",
    )
    require(
        all(
            r["code"] in official_restrictions and format(D(r["amount"]), ",.2f") in official_restrictions
            for r in securities["rows"]
        ),
        "OFFICIAL_RESTRICTED_ROW_CROSSCHECK",
    )
    require(LEGAL in legal[:12000] and "分为A类和C类不同的类别" in legal, "LEGAL_SAME_PRODUCT")
    account_quote = "本基金的一切货币收支活动,均需通过基金托管人的资产托管专户进行"
    require(account_quote in legal, "BANK_ACCOUNT_PURPOSE")
    cash_view, _ = periodic_financial_view(own_pages)
    cash = accounting_observations(cash_view, date.fromisoformat(HOLDING))["observations"]
    bank = D(cash["demand_deposits_principal"]["value"])
    interest = D(cash["demand_deposits_total"]["value"]) - bank
    require(f"等于:本金{bank:,.2f}加:应计利息{interest:,.2f}" in official_own, "OFFICIAL_PRINCIPAL_NOT_INTEREST")
    liabilities = _liabilities(cash_view, own["balance"]["liabilities"])
    payable = D(liabilities["explicit_total"])
    require(D(liabilities["point_values"]["应付清算款"]) == 0, "SETTLEMENT_NOT_PROVEN_ZERO")
    require(
        format(payable, ",.2f") in official_own and format(D(own["balance"]["net_assets"]), ",.2f") in official_own,
        "OFFICIAL_BALANCE_NOT_RECONCILED",
    )
    flows, flow_proof = _flows(own_pages)
    categories = category_table(official_own, flows)
    structure_quote = "本基金本报告期无需要披露的单一投资者持有基金份额比例达到或超过20%的情况。"
    require(structure_quote in own_text and structure_quote in official_own, "ACTUAL_SINGLE_INVESTOR_DISCLOSURE")
    mechanism = section(legal, "巨额赎回的处理方式", "十二、暂停申购或赎回的公告")
    require(
        all(
            x in mechanism
            for x in [
                "全额赎回",
                "部分延期赎回",
                "单个基金份额持有人的赎回申请",
                "超过上一开放日基金总份额20%",
                "无优先权",
            ]
        ),
        "REDEMPTION_MECHANISM",
    )
    operation = section(official_own, "4.4.1报告期内基金投资策略和运作分析", "4.4.2报告期内基金的业绩表现")
    trustee = section(official_own, "5.1报告期内本基金托管人遵规守信情况声明", "6半年度财务会计报告")
    require(
        all(x in operation for x in ["认真对待投资者日常申购", "赎回及成份股调整", "未发生任何风险事件"]),
        "ACTUAL_REDEMPTION_EXECUTION",
    )
    require(
        all(
            x in trustee
            for x in [
                "基金份额申购赎回价格计算",
                "托管人未发现损害基金份额持有人利益的行为",
                "进行了核查",
                "以上内容真实、准确和完整",
            ]
        ),
        "ACTUAL_TRUSTEE_REVIEW",
    )
    # 仅由当期明确实际运作陈述形成此H1持有人事项范围，不声称未来压力验收。
    holder = {
        "kind": "holder_execution_r5",
        "structure_type": "DISPERSED",
        "structure_closed": True,
        "share_flow_closed": True,
        "actual_execution_verified": True,
        "redemption_mechanism_verified": True,
        "actual_redemption_shares": str(sum(D(r["redeemed"]) for r in flows)),
        "confirmed_pressure_effects": [],
        "source_refs": [PRIMARY, OWN_PDF, LEGAL_PDF],
    }
    with localcontext() as context:
        context.prec = 70
        scale = D(own["balance"]["fund_amount"]) / D(target["balance"]["net_assets"])
        require(scale == D(replay["audit"]["feeder_budgets"][FUND]["etf_scale"]), "LOOKTHROUGH_SCALE")
        restriction_lower = scale * D(securities["restricted_amount"])
    return {
        "observations": {"market_capacity": capacity_node, "holders": holder},
        "domains": {
            "available_bank_principal": {"lower": "0", "upper": str(bank), "candidate": None},
            "restricted_amount": {"lower": str(restriction_lower), "upper": None, "candidate": None},
        },
        "audit": {
            "bank_book_principal": str(bank),
            "bank_interest_excluded": str(interest),
            "all_book_liabilities": str(payable),
            "liability_conservation": liabilities,
            "bank_restriction_upper_needed_for_75": str(bank - payable),
            "available_bank_principal": None,
            "all_asset_restricted_amount": None,
            "bank_ordinary_payment_purpose_quote": account_quote,
            "weak_full_asset_scope_quote": weak_scope,
            "full_asset_scope_closed": False,
            "own_balance": own["balance"],
            "etf_balance": target["balance"],
            "etf_securities_only_restrictions": securities,
            "restriction_layout": layout,
            "lookthrough_scale": str(scale),
            "restriction_risk_budget_lower": str(restriction_lower),
            "holder_flows": flows,
            "flow_conservation": flow_proof,
            "complete_A_C_category_table": categories,
            "market_response_audit": market_audit,
            "structure_original": structure_quote,
            "mechanism_original": mechanism,
            "current_operation_original": operation,
            "current_trustee_original": trustee,
            "frozen_source_count": len(base["sources"]),
            "capacity_raw": capacity_node,
            "no_fresh_peer_selection": True,
            "other_seven_not_changed": True,
        },
    }


def calculate(inputs, *, root):
    """按原40/20/25/15权重传播保守界，不取中点、不虚构任一未知金额。"""
    compiled = compile_observations(inputs, root=root)
    nodes, audit = compiled["observations"], compiled["audit"]
    require(PROTOCOL.get("absolute_exposure_adjustment") is True, "CAPACITY_RULE_CHANGED")
    with localcontext() as context:
        context.prec = 60
        cap = nodes["market_capacity"]
        relative = peer_gap_score(
            cap["target"],
            [r["value"] for r in cap["peers"]],
            neutral=PROTOCOL["parameters"]["relative_neutral"],
            amplitude=PROTOCOL["parameters"]["relative_amplitude"],
        )
        capacity = relative * exposure_factor("market_capacity", cap["target"])
        holder, holder_proof = holders_score(nodes["holders"])
        restricted_upper = D(100) * max(
            D(0), D(1) - D(audit["restriction_risk_budget_lower"]) / D(audit["own_balance"]["net_assets"])
        )
        lower = capacity * D(".40") + holder * D(".15")
        upper = lower + restricted_upper * D(".20") + D(75) * D(".25")
    return {
        "version": VERSION,
        "lower": str(lower),
        "upper": str(upper),
        "candidate": None,
        "weights": deepcopy(WEIGHTS),
        "observations": nodes,
        "subscores": {"market_capacity": str(capacity), "holders": str(holder)},
        "subscore_domains": {
            "restricted": {"lower": "0", "upper": str(restricted_upper), "candidate": None},
            "known_payment": {"lower": "0", "upper": "75", "candidate": None},
        },
        "audit": {
            **audit,
            "holder_proof": holder_proof,
            "bounds_are_conservative_outer_bounds_not_joint_sharp_extrema": True,
            "payment_domain_proof": "S=0 uniquely by complete nonnegative liabilities; "
            "T cannot cover other P. "
            "0<=availableB<=bookB and P>0; score=75*min(availableB,P)/P in[0,75]. No actualB assigned.",
            "restriction_domain_proof": "All restrictions>=lookthrough known ETF securities restrictions. "
            "100*max(0,1-R/NAV) is nonincreasing, so its upper bound follows from known R lower; "
            "unknown remainder never set0. Lower0 is conservative, not an observed restriction total.",
        },
        "complete_score": None,
        "grade": None,
    }


def check_binding(bundle, inputs):
    """绑定fresh current的本C、同报告/管理人、原七维冻结输入，拒绝借份额或换同类。"""
    require(
        bundle.get("family") == "INDEX"
        and bundle.get("fund_code") == FUND
        and bundle.get("share_class") == "C"
        and str(bundle.get("as_of_date")) == ASOF
        and str(bundle.get("holding_date")) == HOLDING
        and bundle["identity"] == {"legal_name": LEGAL, "manager_name": MANAGER}
        and bundle["original_report_archive"]["sha256"] == PRIMARY,
        "CURRENT_BINDING",
    )
    base = bundle["passive_components_calculation"]["packet"]["passive_index_compilation"]
    require(
        base == inputs["base_compilation"] and digest(base) == inputs["base_compilation_sha256"],
        "CURRENT_FROZEN_COMPILATION",
    )
    require(not bundle.get("r5_calculation"), "COMPLETE_POINT_CONFLICT")


def replay_snapshot_interval(bundle, dimensions, *, root):
    """prepare/validate/calculate三处都须调用；两端和空candidate均由原件校验。"""
    inputs = bundle[PACKAGE]["inputs"]
    check_binding(bundle, inputs)
    result = calculate(inputs, root=root)
    value = dimensions["liquidity"]
    require(
        value.get("candidate") is None
        and all(abs(D(str(value[k])) - D(result[k])) <= D("1e-55") for k in ["lower", "upper"]),
        "SNAPSHOT_INTERVAL_MISMATCH",
    )
    return result


def prepare(
    session, request, *, root, current_reader=None, archive_writer=None, source_validator=None, snapshot_calculator=None
):
    """准备可审阅清单，不写库；只替换流动性范围，七维和验证时刻/有效期不动。

    shared snapshot尚未接线时可注入其仅新增本分支的隔离副本；默认仍用真实app。
    未经接线的默认snapshot不会把新流动性包视为已核准，调用方必须先实质接入。
    """
    from app.services.fund_rating_pipeline import current_inputs
    from app.services.fund_rating_pipeline_prepare import archive_json, canonical_source
    from app.services.fund_rating_reviewed_snapshot import BASIS, validate_sources
    from app.services.fund_rating_reviewed_snapshot import calculate as snapshot_calculate

    root = Path(root)
    current = (current_reader or current_inputs)(session, [FUND]).get(FUND)
    require(current and str(current["as_of_date"]) == ASOF, "LATEST_CURRENT")
    bundle = deepcopy(next(m["bundle"] for m in current["input_snapshot"]["members"] if m["fund_code"] == FUND))
    raw = deepcopy(request["inputs"])
    check_binding(bundle, raw)
    require(
        request.get("fund_code") == FUND and request.get("as_of_date") == ASOF and bundle.get("basis") == BASIS,
        "REQUEST_SCOPE",
    )
    before = deepcopy(bundle["reviewed_document"]["dimensions"])
    old_verified = bundle["verified_at"]
    # 原95源来自现存正式包，保持其元数据逐字不动；新官方PDF独立规范归档。
    for h in [OWN_PDF, ETF_PDF, LEGAL_PDF]:
        raw["sources"][h] = canonical_source(root, raw["sources"][h])
    value = calculate(raw, root=root)
    explanation = request["explanation"].strip()
    require(len(explanation) >= 40, "BUSINESS_EXPLANATION")
    bundle["reviewed_document"]["dimensions"]["liquidity"] = {
        "lower": value["lower"],
        "upper": value["upper"],
        "candidate": None,
        "reason": explanation,
    }
    require(
        all(bundle["reviewed_document"]["dimensions"][k] == v for k, v in before.items() if k != "liquidity")
        and bundle["verified_at"] == old_verified,
        "OTHER_SEVEN_OR_VALIDITY_CHANGED",
    )
    writer = archive_writer or archive_json
    bundle.update(research_format="single", research_archive=writer(root, bundle["reviewed_document"]))
    bundle[PACKAGE] = {"inputs": raw, "input_archive": writer(root, raw)}
    bundle.setdefault("public_explanations", {})["liquidity"] = explanation
    bundle.get("public_metrics", {}).pop("liquidity", None)
    known = {r["sha256"] for r in bundle.setdefault("extra_archives", [])}
    bundle["extra_archives"].extend(r for h, r in raw["sources"].items() if h not in known)
    (source_validator or validate_sources)(bundle, datetime.now(UTC), root=root)
    checked = (snapshot_calculator or snapshot_calculate)(bundle, root=root)
    replay_snapshot_interval(bundle, checked["dimensions"], root=root)
    require(
        checked["score"] is None and checked["grade"] is None and checked["status"] == "PARTIAL", "MUST_REMAIN_PARTIAL"
    )
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}]}
