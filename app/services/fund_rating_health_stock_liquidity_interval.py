"""010710流动性原件区间；容量未知不填条件分或中点，其他七维/有效期不变。"""

import json
import re
import unicodedata
from copy import deepcopy
from datetime import UTC, date, datetime
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path

from bs4 import BeautifulSoup

from app.services.fund_quality_v2_balance_proofs import strict_balance_values
from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_quality_v2_restrictions import full_asset_restriction_scope, restricted_securities
from app.services.fund_quality_v2_stock_book import periodic_financial_view
from app.services.fund_rating_direct_index import Originals, _flows, _liabilities
from app.services.fund_rating_public_stock_tables import parse as stock_parse
from app.services.fund_rating_r5 import PROTOCOL
from app.services.fund_rating_r5_rules import holders_score, payment_score
from app.services.fund_rating_reviewed_quality import _node, policy

MONEY = "(?:\\d{1,3}(?:,\\d{3})+|\\d+)\\.\\d{2}"


def require(ok, code):
    if not ok:
        raise ValueError("HEALTH_" + code)


def iso(x):
    require(isinstance(x, str) and re.fullmatch("\\d{8}|\\d{4}-\\d{2}-\\d{2}", x), "DATE_SCHEMA")
    return date.fromisoformat(x).isoformat()


def plain(s):
    return re.sub("\\s+", "", unicodedata.normalize("NFKC", s))


def amount(s):
    value = D(str(s).replace(",", ""))
    require(value.is_finite(), "NONFINITE_AMOUNT")
    return value


def text(source, digest):
    cache = getattr(source, "health_text", None)
    if cache is None:
        cache = source.health_text = {}
    if digest in cache:
        return cache[digest]
    raw = source.raw[digest]
    if raw.startswith(b"%PDF"):
        from app.services.fund_rating_pdf import text_pages

        result = "\n".join(text_pages(source.root / source.sources[digest]["file"]))
    elif raw.lstrip().startswith(b"{"):
        obj = json.loads(raw)
        require(
            obj.get("success") == 1 and isinstance(obj.get("data"), dict) and obj["data"].get("notice_content"),
            "DOCUMENT_BODY_REQUIRED",
        )
        if source.sources[digest].get("published_on"):
            require(
                obj["data"].get("notice_date", "")[:10] == source.sources[digest]["published_on"],
                "ORIGINAL_NOTICE_PUBLICATION_DATE",
            )
        result = obj["data"]["notice_content"]
    else:
        try:
            result = raw.decode("utf-8")
        except UnicodeDecodeError:
            result = raw.decode("gb18030")
        if "<html" in result.lower() or "<!doctype" in result.lower():
            soup = BeautifulSoup(result, "html.parser")
            body = soup.find(id="content")
            require(body is not None, "ACTUAL_ARTICLE_CONTAINER_REQUIRED")
            result = body.get_text("\n", strip=True)
    cache[digest] = result
    return result


def report(source, digest, code, end):
    s = text(source, digest)
    u = plain(s)
    _, actual = report_period(s)
    require(str(actual) == end, "ACTUAL_REPORT_PERIOD:" + code)
    m = re.search("基金主代码(?:基金代码)?(\\d{6})", u[:30000])
    require(m is not None and m[1] == MASTERS[code] and (code in u[:35000]), "REPORT_SHARE_OR_LEGAL_PRODUCT:" + code)
    require(
        source.sources[digest].get("published_on") and iso(source.sources[digest]["published_on"]) <= ASOF,
        "REPORT_PUBLICATION_REQUIRED",
    )
    return s


WRAP = re.compile(
    "(?m)^(?P<prefix>[ \\t]*\\d{1,4}[ \\t]+\\d{5,6}[ \\t]+[\\u4e00-\\u9fffA-Za-z*\\-－—]+[ \\t]+)(?P<quantity>\\d{1,3}(?:,\\d{3})*,\\d{1,2})(?P<rest>[ \\t]+(?:\\d{1,3}(?:,\\d{3})+|\\d+)\\.\\d{2}[ \\t]+\\d+(?:\\.\\d+)?[ \\t]*)\\r?\\n(?:[ \\t]*\\r?\\n){0,3}[ \\t]*(?P<tail>\\d{1,2})[ \\t]*$"  # noqa: E501
)


def stock_book(source, digest, code):
    s = report(source, digest, code, HOLDING)
    view, trace = periodic_financial_view([s])
    edits = []

    def join(m):
        q = m["quantity"] + m["tail"]
        require(re.fullmatch("\\d{1,3}(?:,\\d{3})+", q), "VISIBLE_QUANTITY_NOT_UNIQUE")
        after = m["prefix"] + q + m["rest"]
        edits.append({"before": m[0], "after": after})
        return after

    view = [WRAP.sub(join, v) for v in view]
    b = strict_balance_values(view, observed_on=HOLDING)["point_book"]
    book = stock_parse(view, b["stock_amount"], b["net_assets"], require_industry=False)
    return (book, b, view, {"financial_layout_edits": trace, "visible_quantity_edits": edits})


VERSION = "R5_010710_LIQUIDITY_INTERVAL_ORIGINALS_20261029_1"
FUND = "010710"
ASOF = "2026-09-30"
HOLDING = "2026-06-30"
MASTERS = {FUND: "010709"}
LEGAL_NAME = "安信医药健康主题股票型发起式证券投资基金"
MANAGER_NAME = "安信基金管理有限责任公司"
PRIMARY_HASH = "dd628abbb8f99fe25e27b89837e4bdcd5b3f820b6e67d1d517532c142f29b228"
WEIGHTS = {"market_capacity": 40, "restricted": 20, "known_payment": 25, "holders": 15}


def compile_observations(inputs, *, root):
    """从本期完整原表编译三项；不构造容量点观测。"""
    source = originals(inputs, root)
    book, balance, view, layout = stock_book(source, inputs["primary_ref"], FUND)
    books = {FUND: book}
    balances = {FUND: balance}
    views = {FUND: view}
    b = balances[FUND]
    s = text(source, inputs["primary_ref"])
    u = plain(s)
    restriction = restricted_securities(views[FUND], books[FUND]["holdings"], b["bond_amount"])
    scope = full_asset_restriction_scope(s) or re.search(
        '除在本报告[“\\"]期末本基金持有的流通受限证券[”\\"]章节中列示的部分基金资产流通暂时受限制外[（(]如有[）)][，,]其余均能及时变现',
        u,
    )
    require(scope, "FULL_ASSET_RESTRICTION_SCOPE_REQUIRED")
    legal = plain(text(source, inputs["payment_legal"]))
    require(
        "用于基金财产证券交易结算资金的存管" in legal and "证券交易资金账户为主资金账户" in legal, "BROKER_CASH_PURPOSE"
    )
    a = s.rfind("6.4.7.1 货币资金")
    z = s.find("6.4.7.2 交易性金融资产", a)
    require(a >= 0 and z > a, "ACTUAL_BANK_PRINCIPAL_NOTE")
    note = s[a:z]
    matches = re.findall("等于[：:]本金\\s+(" + MONEY + "|-)", note)
    require(len(matches) == 3 and matches[1] == "-", "BANK_DEPOSIT_PRINCIPAL_SCOPE")
    bank, broker = (amount(matches[0]), amount(matches[2]))
    require(
        "其他存款为存放在证券经纪商基金专用证券资金账户的证券交易结算资金" in plain(note),
        "BROKER_ACCOUNT_ORIGINAL_PURPOSE",
    )
    liabilities = _liabilities([s], b["liabilities"])
    settle = liabilities["point_values"]["应付清算款"]
    rows, flow = _flows([s])
    require(len(rows) == 2, "A_C_ACTUAL_FLOWS_REQUIRED")
    start = s.rfind("8.1 期末基金份额持有人户数及持有人结构")
    end = s.find("8.2", start)
    require(start >= 0 and end > start, "HOLDER_STRUCTURE_REQUIRED")
    structure = s[start:end]
    category_pattern = f"安信医药健\\s+(\\d{{1,3}}(?:,\\d{{3}})*)\\s+({MONEY})\\s+({MONEY})\\s+({MONEY})\\s+({MONEY})\\s+({MONEY})\\s+康股票\\s+([AC])"  # noqa: E501
    category_matches = re.findall(category_pattern, structure)
    require(
        len(category_matches) == 2 and [r[-1] for r in category_matches] == ["A", "C"],
        "COMPLETE_A_C_HOLDER_CATEGORY_ROWS",
    )
    categories = []
    for cells, share_flow in zip(category_matches, rows, strict=True):
        households, average, institution, institution_pct, individual, individual_pct, share = cells
        require(share == share_flow["share"] and int(households.replace(",", "")) > 0, "CATEGORY_SHARE_IDENTITY")
        require(
            amount(institution) + amount(individual) == D(share_flow["closing"]), "CATEGORY_SHARE_FLOW_CONSERVATION"
        )
        require(amount(institution_pct) + amount(individual_pct) == 100, "CATEGORY_REPORTED_PERCENTAGE_CONSERVATION")
        categories.append(
            {
                "share": share,
                "households": int(households.replace(",", "")),
                "institution_shares": str(amount(institution)),
                "individual_shares": str(amount(individual)),
                "closing_shares": share_flow["closing"],
            }
        )
    aggregate = re.search(
        f"合计\\s+[\\d,]+\\s+({MONEY})\\s+({MONEY})\\s+({MONEY})\\s+({MONEY})\\s+({MONEY})", structure
    )
    require(
        aggregate is not None
        and amount(aggregate[2]) == sum(amount(r["institution_shares"]) for r in categories)
        and (amount(aggregate[4]) == sum(amount(r["individual_shares"]) for r in categories)),
        "WHOLE_PRODUCT_CATEGORY_CONSERVATION",
    )
    single = s[s.rfind("11.1 报告期内单一投资者") : s.rfind("11.2")]
    require(re.search("情况\\s*无[。.]", single), "SINGLE_HOLDER_DISCLOSURE_UNKNOWN")
    operation, trust, mechanism = public_redemption(s, legal)
    from app.services.fund_rating_cmf_liquidity import COMPATIBLE_TYPES, FOUNDATION_GATES
    from app.services.fund_rating_cmf_liquidity import PROTOCOL as HOLDER_DOMAIN_PROTOCOL

    refs = [inputs["primary_ref"], inputs["payment_legal"]]
    nodes = {
        "restricted": {
            "kind": "restricted",
            "restricted_amount": restriction["restricted_amount"],
            "net_assets": b["net_assets"],
            "scope_closed": True,
            "source_refs": refs,
        },
        "known_payment": {
            "kind": "purpose_covered_payment",
            "as_of": HOLDING,
            "currency": "CNY",
            "bank_principal": str(bank),
            "broker_settlement_principal": str(broker),
            "settlement_liabilities": settle,
            "all_book_liabilities": b["liabilities"],
            "scope_closed": True,
            "restriction_amount_pending": False,
            "disclosed_restrictions": [],
            "source_refs": refs,
        },
        "holders": {
            "kind": "holder_execution_r5",
            "structure_type": None,
            "structure_condition_protocol": HOLDER_DOMAIN_PROTOCOL,
            "compatible_structure_types": list(COMPATIBLE_TYPES),
            "structure_basis": "OPEN_END_CASH_LEGAL_PRODUCT_AND_COMPLETE_REPORTED_CATEGORY_TABLE",
            "foundation_gate_checks": {name: True for name in FOUNDATION_GATES},
            "structure_closed": True,
            "share_flow_closed": True,
            "actual_execution_verified": True,
            "redemption_mechanism_verified": True,
            "actual_redemption_shares": str(sum(D(r["redeemed"]) for r in rows)),
            "confirmed_pressure_effects": [],
            "source_refs": refs,
        },
    }
    return {
        "observations": nodes,
        "audit": {
            "stock_count": len(book["holdings"]),
            "stock_layout": layout,
            "balance": b,
            "full_restriction_scope": scope[0],
            "restricted_security_details": restriction,
            "cash_note_original": note,
            "bank_principal": str(bank),
            "broker_principal": str(broker),
            "liability_components_and_conservation": liabilities,
            "holder_flows": rows,
            "flow_reconciliation": flow,
            "complete_A_C_category_rows": categories,
            "single_investor_20pct_disclosure": single,
            "actual_individual_concentration_type": None,
            "ordinary_and_single_large_mechanism": mechanism,
            "current_liquidity_execution_original": operation,
            "current_trustee_execution_original": trust,
            "not_future_redemption_safety": True,
            "does_not_clear_company_governance_or_other_dimensions": True,
        },
    }


def public_redemption(s: str, legal: str) -> tuple[str, str, str]:
    """只核公开申赎机制、报告期实际履职和托管复核。"""
    u = plain(s)
    a = s.rfind("6.4.13.3.1 报告期内本基金组合资产的流动性风险分析")
    z = s.find("6.4.13.4", a)
    operation = plain(s[a:z])
    require(
        a >= 0
        and z > a
        and ("开放日对本基金的申购赎回情况进行监控" in operation)
        and ("本报告期内,本基金未发生重大流动性风险事件" in operation),
        "ACTUAL_LIQUIDITY_EXECUTION",
    )
    a = s.rfind("5.1 报告期内本基金托管人")
    z = s.find("§6", a)
    trust = plain(s[a:z])
    require(
        a >= 0
        and "对托管产品的投资行为进行监督" in trust
        and ("独立地设置、登录和保管本产品的全套账册" in trust)
        and ("进行会计核算和资产估值并与管理人建立对账机制" in trust),
        "TRUSTEE_ACTUAL_OPERATION",
    )
    require(
        "2026年8月27日复核" in u and all(t in u for t in ["本报告", "财务指标", "投资组合报告"]),
        "TRUSTEE_SPECIFIC_REPORT_REVIEW",
    )
    a = legal.rfind("巨额赎回的处理方式")
    mechanism = legal[a : a + 4300]
    require(
        a >= 0
        and all(
                term in mechanism
                for term in [
                    "全额赎回",
                    "当日接受赎回比例不低于上一开放日基金总份额的10%",
                    "单个基金份额持有人超过上一开放日基金总份额30%",
                    "无优先权",
                ]
        ),
        "ORDINARY_AND_SINGLE_LARGE_HOLDER_MECHANISMS",
    )
    require("契约型开放式" in u and "没有损害基金份额持有人利益的行为" in u, "CURRENT_MANAGER_HOLDER_EXECUTION")
    return operation, trust, mechanism


def originals(inputs, root):
    """逐次核字节和公开日；只读同基金财报及实际账户/赎回法律文件。"""
    require(
        inputs.get("version") == VERSION
        and inputs.get("fund_code") == FUND
        and inputs.get("share_class") == "C"
        and inputs.get("currency") == "CNY"
        and inputs.get("as_of_date") == ASOF
        and inputs.get("holding_date") == HOLDING
        and inputs.get("legal_name") == LEGAL_NAME
        and inputs.get("manager_name") == MANAGER_NAME
        and inputs.get("primary_ref") == PRIMARY_HASH,
        "INTERVAL_SCOPE",
    )
    require(
        not any(
            k in inputs
            for k in [
                "score",
                "candidate",
                "lower",
                "upper",
                "observations",
                "dimensions",
                "market_capacity_score",
                "market_capacity_range",
                "bank_principal",
                "broker_settlement_principal",
                "all_book_liabilities",
                "restricted_amount",
            ]
        ),
        "INTERVAL_PRECOMPUTED_VALUES_FORBIDDEN",
    )
    require(
        inputs.get("weights") == WEIGHTS
        and dict(policy("PUBLIC_REVIEWED_ACTIVE_QUALITY_V2")["terms"]["liquidity"]) == WEIGHTS,
        "INTERVAL_ORIGINAL_WEIGHTS",
    )
    require(set(inputs["sources"]) == {inputs["primary_ref"], inputs["payment_legal"]}, "INTERVAL_TWO_ORIGINALS_ONLY")
    for ref in inputs["sources"].values():
        require(
            not Path(ref["file"]).is_absolute() and ref.get("published_on") and ref["published_on"] <= ASOF,
            "INTERVAL_ORIGINAL_PUBLIC_DATE",
        )
    source = Originals(inputs, Path(root))
    u = plain(text(source, inputs["primary_ref"]))
    require(
        LEGAL_NAME in u
        and MANAGER_NAME in u
        and "010710" in u[:35000]
        and "010709" in u[:35000]
        and "单位:人民币元" in u,
        "INTERVAL_PRIMARY_IDENTITY",
    )
    legal = plain(text(source, inputs["payment_legal"]))
    require(LEGAL_NAME in legal and "招募说明书" in legal, "INTERVAL_LEGAL_PRODUCT")
    return source


def calculate(inputs, *, root):
    """原换分算三项，容量取0..100保守闭域；单一候选分始终为空。"""
    compiled = compile_observations(inputs, root=root)
    nodes = compiled["observations"]
    require(
        D(PROTOCOL["parameters"]["relative_neutral"]) == 75
        and D(PROTOCOL["parameters"]["relative_amplitude"]) == 25
        and PROTOCOL.get("absolute_exposure_adjustment") is True,
        "INTERVAL_CAPACITY_DOMAIN_PROTOCOL_CHANGED",
    )
    with localcontext() as context:
        context.prec = 60
        fraction = _node(nodes["restricted"], aliases={"010709", "010710"})
        restricted = D(fraction.numerator) / D(fraction.denominator)
        payment, payment_proof = payment_score(nodes["known_payment"])
        holder, holder_proof = holders_score(nodes["holders"])
        known = restricted * D(20) / 100 + payment * D(25) / 100 + holder * D(15) / 100
        lower, upper = known, known + D(40)
    return {
        "version": VERSION,
        "lower": str(lower),
        "upper": str(upper),
        "candidate": None,
        "subscores": {"restricted": str(restricted), "known_payment": str(payment), "holders": str(holder)},
        "weights": deepcopy(WEIGHTS),
        "capacity_score_domain": {"lower": "0", "upper": "100", "candidate": None},
        "observations": nodes,
        "audit": {
            **compiled["audit"],
            "payment_proof": payment_proof,
            "holder_proof": holder_proof,
            "capacity_domain_proof": "K>=0,m>=0 gives -1<=(m-K)/(m+K)<=1; both0 uses75. C is50..100; "
            "0<1/(1+K)<=1. Thus0<=C/(1+K)<=100. Conservative closed domain only; "
            "no targetK, peer median, midpoint or conditional66.85 is substituted.",
            "unknown_capacity_not_defaulted_to_zero": True,
            "capacity_domain_not_a_confidence_interval": True,
        },
        "complete_score": None,
        "grade": None,
    }


def check_binding(bundle, inputs):
    """同份额、法律产品、报告日、管理人、主原件和截止日绑定。"""
    require(
        bundle.get("family") == "ACTIVE_EQUITY"
        and bundle.get("fund_code") == FUND == inputs.get("fund_code")
        and bundle.get("share_class") == "C" == inputs.get("share_class")
        and str(bundle.get("as_of_date")) == ASOF == inputs.get("as_of_date")
        and str(bundle.get("holding_date")) == HOLDING == inputs.get("holding_date")
        and bundle["identity"]["legal_name"] == inputs["legal_name"] == LEGAL_NAME
        and bundle["identity"]["manager_name"] == inputs["manager_name"] == MANAGER_NAME
        and bundle["original_report_archive"]["sha256"] == inputs["primary_ref"] == PRIMARY_HASH,
        "INTERVAL_CURRENT_BINDING",
    )
    require(
        not bundle.get("r5_calculation") and not bundle.get("health_stock_components_calculation"),
        "INTERVAL_CONFLICTING_POINT_CALCULATION",
    )


def replay_snapshot_interval(bundle, dimensions, *, root):
    """正式snapshot接线：从原件重算两端，拒绝改成中点或条件分。"""
    raw = bundle["health_stock_liquidity_interval_calculation"]["inputs"]
    check_binding(bundle, raw)
    result = calculate(raw, root=root)
    value = dimensions["liquidity"]
    require(
        value.get("candidate") is None
        and all(abs(D(str(value[k])) - D(result[k])) <= D("1e-55") for k in ["lower", "upper"]),
        "INTERVAL_SNAPSHOT_RECALCULATION",
    )
    return result


def prepare(session, request, *, root):
    """只更改流动性区间，保留其他七维及有效期；不发布数据库。"""
    from app.services.fund_rating_pipeline import current_inputs
    from app.services.fund_rating_pipeline_prepare import archive_json, canonical_source
    from app.services.fund_rating_reviewed_snapshot import BASIS, validate_sources
    from app.services.fund_rating_reviewed_snapshot import calculate as snapshot_calculate

    require(request.get("fund_code") == FUND and request.get("as_of_date") == ASOF, "INTERVAL_REQUEST_SCOPE")
    current = current_inputs(session, [FUND]).get(FUND)
    require(current and str(current["as_of_date"]) == ASOF, "INTERVAL_LATEST_CURRENT")
    bundle = deepcopy(next(x["bundle"] for x in current["input_snapshot"]["members"] if x["fund_code"] == FUND))
    raw = deepcopy(request["inputs"])
    check_binding(bundle, raw)
    require(bundle.get("basis") == BASIS, "INTERVAL_CURRENT_BASIS")
    before = deepcopy(bundle["reviewed_document"]["dimensions"])
    for h, ref in raw["sources"].items():
        raw["sources"][h] = canonical_source(Path(root), ref)
    result = calculate(raw, root=root)
    reason = request["explanation"].strip()
    require(len(reason) >= 40, "INTERVAL_BUSINESS_EXPLANATION")
    bundle["reviewed_document"]["dimensions"]["liquidity"] = {
        "lower": result["lower"],
        "upper": result["upper"],
        "candidate": None,
        "reason": reason,
    }
    require(
        all(bundle["reviewed_document"]["dimensions"][k] == v for k, v in before.items() if k != "liquidity"),
        "INTERVAL_OTHER_SEVEN_CHANGED",
    )
    bundle.setdefault("public_explanations", {})["liquidity"] = reason
    bundle.get("public_metrics", {}).pop("liquidity", None)
    bundle.update(
        research_format="single",
        research_archive=archive_json(Path(root), bundle["reviewed_document"]),
        health_stock_liquidity_interval_calculation={"inputs": raw, "input_archive": archive_json(Path(root), raw)},
    )
    known = {x["sha256"] for x in bundle.setdefault("extra_archives", [])}
    bundle["extra_archives"].extend(ref for h, ref in raw["sources"].items() if h not in known)
    validate_sources(bundle, datetime.now(UTC), root=Path(root))
    checked = snapshot_calculate(bundle, root=Path(root))
    replay_snapshot_interval(bundle, checked["dimensions"], root=Path(root))
    require(
        checked["score"] is None
        and checked["grade"] is None
        and checked["status"] == "PARTIAL"
        and checked["dimensions"]["liquidity"]["candidate"] is None,
        "INTERVAL_MUST_REMAIN_PARTIAL",
    )
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}]}
