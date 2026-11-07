"""018125流动性已知三子项的原件编译候选；成交承接仍缺，完整分恒空。

只依赖正式app模块。原件按inputs.sources和Originals(root)限目录核字节，
不读取研究目录或旧绝对路径，不把未知重大事项默认成False。
"""

import json
import re
from contextlib import closing
from copy import deepcopy
from datetime import UTC, date, datetime
from decimal import Decimal as D
from functools import lru_cache
from pathlib import Path

from bs4 import BeautifulSoup

from app.core.pdf import PDF_LOCK
from app.services.fund_quality_v2_accounting import accounting_observations
from app.services.fund_quality_v2_disclosures import normalize_number_lines
from app.services.fund_quality_v2_restrictions import full_asset_restriction_scope, restricted_securities
from app.services.fund_rating_direct_index import Originals, _flows
from app.services.fund_rating_inputs import canonical
from app.services.fund_rating_market_review import normalized_text
from app.services.fund_rating_r5_rules import holders_score
from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS, _node, policy
from app.services.material_content import document_pdfium

VERSION = "R5_018125_LIQUIDITY_ORIGINALS_20261029_1"
PRIMARY = "4ec67324845685442b529bf43e29736143289fe80de7349afa7df581a662c26e"
LEGAL = "永赢先进制造智选混合型发起式证券投资基金"
MANAGER = "永赢基金管理有限公司"


def require(condition, code):
    if not condition:
        raise ValueError(code)


def section(body, start, stop):
    """取正文中唯一的完整编号节；目录标题不参与正文匹配。"""
    pattern = rf"(?m)^\s*{start}[^\r\n]*\r?$"
    hits = [m for m in re.finditer(pattern, body) if not re.search(r"\.{2,}|…", normalized_text(m[0]))]
    require(len(hits) == 1, "YY_COMPLETE_SECTION:" + start)
    tail = body[hits[0].start() :]
    end = re.search(rf"(?m)^\s*{stop}[^\r\n]*\r?$", tail)
    require(end is not None, "YY_COMPLETE_SECTION_STOP:" + stop)
    return tail[: end.start()]


def original_pages(raw):
    """从已经核字节的内存原件读页，沿用项目PDFium与串行原生库锁。"""

    out = []
    with PDF_LOCK, document_pdfium(raw) as doc:
        for i in range(len(doc)):
            with closing(doc[i]) as page, closing(page.get_textpage()) as text:
                out.append(text.get_text_range())
    return out


def reconcile_structure(section_text, closing_shares):
    """保留占比公开精度，恢复原有小数尾数字折行，并核A/C及总计守恒。"""
    section_text = normalize_number_lines(section_text).replace("％", "%")
    money = r"\d[\d,]*\.\d{2}"
    percent = r"\d+(?:\.\d{1,6})?"
    pattern = re.compile(
        rf"(?<![\d,.])(?P<average>{money})\s+(?P<institution>{money})\s+"
        rf"(?P<institution_percent>{percent})\s*%\s+(?P<individual>{money})\s+(?P<individual_percent>{percent})\s*%"
    )
    rows = [m.groupdict() for m in pattern.finditer(section_text)]
    require(len(rows) == len(closing_shares) + 1, "YY_HOLDER_CATEGORY_ROWS")
    totals = [D(x) for x in closing_shares]
    totals.append(sum(totals))
    out = []
    for row, total in zip(rows, totals, strict=True):
        inst, individual = (D(row[k].replace(",", "")) for k in ["institution", "individual"])
        require(total > 0 and inst + individual == total, "YY_HOLDER_AMOUNT_RECONCILIATION")
        for field, value in [("institution_percent", inst), ("individual_percent", individual)]:
            display = D(row[field])
            quantum = D(1).scaleb(display.as_tuple().exponent)
            require(
                0 <= display <= 100 and abs(value / total * 100 - display) <= quantum / 2, "YY_HOLDER_PERCENT_PRECISION"
            )
        out.append({**row, "institution": str(inst), "individual": str(individual), "closing": str(total)})
    for k in ["institution", "individual", "closing"]:
        require(sum(D(x[k]) for x in out[:-1]) == D(out[-1][k]), "YY_HOLDER_LEGAL_TOTAL")
    return out


def _source(inputs, root):
    require(
        inputs.get("version") == VERSION
        and inputs.get("fund_code") == "018125"
        and inputs.get("share_class") == "C"
        and inputs.get("basis") == ACTIVE_BASIS
        and inputs.get("legal_name") == LEGAL
        and inputs.get("manager_name") == MANAGER
        and inputs.get("as_of_date") == "2026-09-30"
        and inputs.get("holding_date") == "2026-06-30"
        and inputs.get("primary_ref") == PRIMARY,
        "YY_COMPONENT_SCOPE",
    )
    require(
        not any(k in inputs for k in ["score", "dimensions", "observations", "quality_metrics", "packet"]),
        "YY_RAW_ORIGINALS_REQUIRED",
    )
    require(inputs.get("material_event_pending") is None, "YY_NO_DEFAULT_GLOBAL_MATERIAL_FALSE")
    require(inputs.get("performance_release_requested") is False, "YY_LIQUIDITY_ONLY_SCOPE")
    sources = inputs.get("sources")
    require(isinstance(sources, dict) and PRIMARY in sources, "YY_PRIMARY_SOURCE_REQUIRED")
    require(
        isinstance(inputs.get("legal_refs"), list)
        and len(inputs["legal_refs"]) == 2
        and set(sources) == {PRIMARY, *inputs["legal_refs"]},
        "YY_UNKNOWN_OR_MISSING_ORIGINAL",
    )
    review = inputs.get("holder_condition_review")
    require(
        isinstance(review, dict) and review.get("complete") is True and review.get("confirmed_pressure_effects") == [],
        "YY_HOLDER_PRESSURE_EVENT_REVIEW_UNKNOWN",
    )
    for _h, ref in sources.items():
        stamp = ref.get("published_on")
        require(
            isinstance(stamp, str) and date.fromisoformat(stamp) <= date(2026, 9, 30),
            "YY_SOURCE_PUBLICATION_DATE_REQUIRED",
        )
        require(ref.get("archive_kind") == "ORIGINAL", "YY_ACTUAL_ORIGINAL_REQUIRED")
    return Originals(inputs, Path(root).resolve())


def _legal_text(source, sha):
    ref = source.sources[sha]
    require(ref.get("encoding") == "gb18030" and ref.get("kind") == "html", "YY_LEGAL_ORIGINAL_FORMAT")
    body = BeautifulSoup(source.raw[sha].decode("gb18030"), "html.parser").get_text(" ", strip=True)
    compact = normalized_text(body)
    # 联合法律文件通常列A/C类别而不印交易代码；代码关系由本期主报告明确绑定。
    # 不能因合同不印C代码而误判原件缺失，仍须同法律全称、管理人、C类及完整机制。
    require(
        LEGAL in compact
        and MANAGER in compact
        and "C类基金份额" in compact
        and "T+7" in compact
        and "支付赎回款项" in compact,
        "YY_LEGAL_IDENTITY_OR_MECHANISM",
    )
    return compact


def _compile(inputs, source):
    # 同一原件只解析一次；外层每次调用仍重新核验实际字节和所有引用。
    pages = original_pages(source.raw[PRIMARY])
    raw = "\n".join(pages)
    plain = normalized_text(raw)
    require(
        len(pages) == 55 and all(x in plain for x in [LEGAL, MANAGER, "018124", "018125", "2026年中期报告"]),
        "YY_PRIMARY_IDENTITY_OR_PERIOD",
    )
    require("2026年6月30日" in plain or "2026年06月30日" in plain, "YY_PRIMARY_HOLDING_DATE")
    accounting = accounting_observations(pages, date(2026, 6, 30))
    obs = accounting["observations"]
    required = ["net_assets", "total_assets", "total_liabilities", "demand_deposits_principal"]
    require(all(k in obs and obs[k].get("value") is not None for k in required), "YY_ACCOUNTING_REQUIRED")
    nav, assets, liabilities, bank = (D(obs[k]["value"]) for k in required)
    require(
        nav > 0 and assets == nav + liabilities and bank >= liabilities >= 0, "YY_BANK_PAYMENT_UNIQUE_BOUND_NOT_PROVEN"
    )
    scope = full_asset_restriction_scope(raw)
    require(scope is not None, "YY_FULL_ASSET_RESTRICTION_SCOPE")
    restriction = restricted_securities(pages, [], scope="ALL_ASSETS")
    require(D(restriction["restricted_amount"]) == 0 and restriction["rows"] == [], "YY_RESTRICTION_PROFILE_CHANGED")
    flows, flow_proof = _flows(pages)
    require([r["share"] for r in flows] == ["A", "C"], "YY_A_C_FLOW_COLUMNS")
    structure = section(raw, r"8\.1\s*期末基金份额持有人户数及持有人结构", r"8\.2")
    rows = reconcile_structure(structure, [r["closing"] for r in flows])
    concentration = section(raw, r"11\.1\s*报告期内单一投资者", r"11\.2")
    require(
        "本基金本报告期内不存在单一投资者持有基金份额比例达到或超过20%的情况" in normalized_text(concentration),
        "YY_FULL_PERIOD_HOLDER_CONCENTRATION_UNKNOWN",
    )
    execution = section(raw, r"5\.2\s*托管人对报告期内本基金投资运作", r"5\.3")
    require(
        all(
            x in normalized_text(execution)
            for x in ["报告期", "复核", "监督", "未发现基金管理人有损害基金份额持有人利益的行为"]
        ),
        "YY_TRUSTEE_CURRENT_OPERATIONS_UNKNOWN",
    )
    compliance = section(raw, r"4\.2\s*管理人对报告期内本基金运作遵规守信情况", r"4\.3")
    require("无损害基金份额持有人利益的行为" in normalized_text(compliance), "YY_MANAGER_CURRENT_OPERATIONS_UNKNOWN")
    require(
        "确保每日确认的净赎回申请不超过7个工作日可变现资产的可变现价值" in plain,
        "YY_ACTUAL_REDEMPTION_LIQUIDITY_OPERATIONS_UNKNOWN",
    )
    legal = inputs.get("legal_refs")
    require(isinstance(legal, list) and len(legal) == 2, "YY_LEGAL_REFS_REQUIRED")
    for sha in legal:
        _legal_text(source, sha)
    review = inputs.get("holder_condition_review")
    require(
        isinstance(review, dict) and review.get("complete") is True and review.get("confirmed_pressure_effects") == [],
        "YY_HOLDER_PRESSURE_EVENT_REVIEW_UNKNOWN",
    )
    expected = {
        "structure": structure,
        "full_period_concentration": concentration,
        "manager_execution": compliance,
        "trustee_execution": execution,
    }
    require(set(review.get("claims", {})) == set(expected), "YY_HOLDER_REVIEW_CLAIM_COVERAGE")
    for key, text in expected.items():
        item = review["claims"][key]
        require(
            item.get("source_hash") == PRIMARY and normalized_text(item.get("quote", "")) == normalized_text(text),
            "YY_HOLDER_ORIGINAL_REVIEW_MISMATCH:" + key,
        )
    node = {
        "kind": "holder_execution_r5",
        "as_of": inputs["holding_date"],
        "structure_closed": True,
        "share_flow_closed": True,
        "redemption_mechanism_verified": True,
        "actual_execution_verified": True,
        "structure_type": "DISPERSED",
        "confirmed_pressure_effects": review["confirmed_pressure_effects"],
        "actual_redemption_shares": str(sum(D(r["redeemed"]) for r in flows)),
        "source_refs": [PRIMARY, *legal],
    }
    holder, holder_audit = holders_score(node)
    restriction_node = {
        "kind": "restricted",
        "restricted_amount": restriction["restricted_amount"],
        "net_assets": str(nav),
        "scope_closed": True,
    }
    restricted = D(str(_node(restriction_node, aliases={"CN_PUBLIC_FUND:018124", "CN_PUBLIC_FUND:018125"})))
    # 对原支付公式的所有相容未知T>=0、0<=S<=P，先抵交收再用普通银行本金：
    # min(T,S)+min(B,P-min(T,S))=P (B>=P)。因此唯一75，不虚填T/S为零。
    payment = D(75)
    scores = {"restricted": restricted, "known_payment": payment, "holders": holder}
    terms = policy(ACTIVE_BASIS)["terms"]["liquidity"]
    known = sum(scores[k] * D(w) / 100 for k, w in terms if k in scores)
    unknown_weight = sum(D(w) for k, w in terms if k not in scores)
    dimension = {
        "lower": str(known),
        "upper": str(known + unknown_weight),
        "candidate": None,
        "missing": ["market_capacity"],
    }
    return {
        "dimension": dimension,
        "subscores": {k: str(v) for k, v in scores.items()},
        "observations": {
            "holders": node,
            "restricted": restriction_node,
            "known_payment": {
                "as_of": inputs["holding_date"],
                "currency": "CNY",
                "bank_principal": str(bank),
                "all_book_liabilities": str(liabilities),
                "broker_settlement_principal": None,
                "settlement_liabilities": None,
                "unique_score_under_all_compatible_purpose_allocations": str(payment),
            },
        },
        "audit": {
            "primary_sha256": PRIMARY,
            "actual_pages": len(pages),
            "holder_category_rows": rows,
            "flows": flows,
            "flow_proof": flow_proof,
            "accounting": accounting,
            "restriction": restriction,
            "restriction_scope_original": scope[0],
            "holder_execution": holder_audit,
            "material_event_pending": None,
            "performance_release_not_reviewed_by_this_component": True,
            "payment_proof": "B>=P；对于全部T>=0及0<=S<=P，min(T,S)+min(B,P-min(T,S))=P。",
            "future_redemption_guarantee": False,
        },
    }


@lru_cache(maxsize=2)
def _cached(serialized, root_name):
    inputs = json.loads(serialized)
    return _compile(inputs, _source(inputs, root_name))


def calculate(inputs, *, root):
    """每次重验原件后重放已知事实；缓存仅复用相同原字节的PDF解析。"""
    _source(inputs, root)
    compiled = deepcopy(_cached(canonical(inputs), str(Path(root).resolve())))
    return {
        "dimensions": {"liquidity": compiled["dimension"]},
        "complete_score": None,
        "grade": None,
        **{k: v for k, v in compiled.items() if k != "dimension"},
    }


def check_binding(bundle, inputs):
    """绑定当前本C份额、报告期、管理人和真实主报告，不能借同基金A。"""
    require(
        bundle.get("family") == "ACTIVE_EQUITY"
        and bundle.get("fund_code") == inputs.get("fund_code")
        and bundle.get("share_class") == inputs.get("share_class")
        and str(bundle.get("as_of_date")) == inputs.get("as_of_date")
        and str(bundle.get("holding_date")) == inputs.get("holding_date")
        and bundle["identity"]["legal_name"] == inputs.get("legal_name")
        and bundle["identity"]["manager_name"] == inputs.get("manager_name")
        and bundle["original_report_archive"]["sha256"] == inputs.get("primary_ref"),
        "YY_CURRENT_BINDING",
    )


def prepare(session, request, *, root):
    """只读最新current，原件重编译后仅更新流动性区间，其余维度完整保留。"""
    from app.services.fund_rating_pipeline import current_inputs
    from app.services.fund_rating_pipeline_prepare import archive_json, canonical_source
    from app.services.fund_rating_reviewed_snapshot import BASIS, validate_sources
    from app.services.fund_rating_reviewed_snapshot import calculate as snapshot_calculate

    code = request["fund_code"]
    raw = deepcopy(request["inputs"])
    require(code == "018125" and request["as_of_date"] == raw["as_of_date"], "YY_REQUEST_SCOPE")
    previous = current_inputs(session, [code]).get(code)
    require(previous and str(previous["as_of_date"]) == request["as_of_date"], "YY_LATEST_CURRENT_REQUIRED")
    require(
        not request.get("expected_current_ref") or request["expected_current_ref"] == previous["batch_id"],
        "YY_CURRENT_CHANGED_REVIEW_REQUIRED",
    )
    bundle = deepcopy(next(m["bundle"] for m in previous["input_snapshot"]["members"] if m["fund_code"] == code))
    check_binding(bundle, raw)
    require(bundle.get("basis") == BASIS, "YY_CURRENT_BASIS")
    require(not bundle.get("r5_calculation"), "YY_PARTIAL_SNAPSHOT_REQUIRED")
    for h, ref in raw["sources"].items():
        raw["sources"][h] = canonical_source(root, ref)
    value = calculate(raw, root=root)
    reason = request["explanations"]["liquidity"].strip()
    require(len(reason) >= 40, "YY_BUSINESS_EXPLANATION_REQUIRED")
    doc = bundle["reviewed_document"]
    before = deepcopy(doc["dimensions"])
    doc["dimensions"]["liquidity"] = {**value["dimensions"]["liquidity"], "reason": reason}
    bundle.setdefault("public_explanations", {})["liquidity"] = reason
    bundle.get("public_metrics", {}).pop("liquidity", None)
    bundle.update(
        research_format="single",
        research_archive=archive_json(root, doc),
        yy_manufacturing_liquidity_calculation={"inputs": raw, "input_archive": archive_json(root, raw)},
    )
    known = {r["sha256"] for r in bundle.setdefault("extra_archives", [])}
    bundle["extra_archives"].extend(r for h, r in raw["sources"].items() if h not in known)
    require(
        all(doc["dimensions"][k] == v for k, v in before.items() if k != "liquidity"), "YY_OTHER_DIMENSIONS_CHANGED"
    )
    validate_sources(bundle, datetime.now(UTC), root=root)
    checked = snapshot_calculate(bundle, root=root)
    require(checked["status"] == "PARTIAL" and checked["score"] is None, "YY_NO_COMPLETE_SCORE")
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}]}
