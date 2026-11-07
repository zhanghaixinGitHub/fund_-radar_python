"""黄金联接A的收益和稳定性原件编译候选，供总指挥集中集成。

000216与五个独立法律产品均须证明Au99.99、95%黄金与5%活期现金的
同一路线。共同参照在收益差中严格消去，未观测的逐日合同基准保持未知；
不将本模块外推为风险、执行或持仓评价。文件、份额、窗口、固定群和历史
合同逐次核验；只输出两个分项的原始观察及标准R5重放，无完整总分或等级。
"""

from __future__ import annotations

import json
import re
import unicodedata
from copy import deepcopy
from datetime import UTC, date, datetime
from decimal import Decimal, localcontext
from functools import lru_cache
from pathlib import Path

from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_rating_direct_index import Originals
from app.services.fund_rating_inputs import canonical
from app.services.fund_rating_r5 import PROTOCOL, _calculate_frozen
from app.services.fund_rating_reviewed_quality import PASSIVE_BASIS

VERSION = "R5_GOLD_RETURN_STABILITY_ORIGINALS_20261029_1"
AS_OF = "2026-09-30"
PERIODS = {"2023-12-31", "2024-12-31", "2025-12-31", "2026-06-30"}
TARGET_ALIASES = {"000216", "000217", "022653"}
INVENTORY_SHA256 = "2f850c45843d05d5f04380ef0a63c633051f80644da83674399366cebccf1d6d"
COHORT_SHA256 = "6cc16498df65301e118c1eeb69e148d5864c7a6972574da7a319b7557d7c666b"
# 本轮24份报告首页逐件核过的法律全名；不以目录短名相似推断同一产品。
LEGAL_NAMES = {
    "000216": "华安易富黄金交易型开放式证券投资基金联接基金",
    "000218": "国泰黄金交易型开放式证券投资基金联接基金",
    "000307": "易方达黄金交易型开放式证券投资基金联接基金",
    "002610": "博时黄金交易型开放式证券投资基金联接基金",
    "008142": "工银瑞信黄金交易型开放式证券投资基金联接基金",
    "008701": "华夏黄金交易型开放式证券投资基金发起式联接基金",
}
CATALOG_COLUMNS = (
    "ts_code",
    "name",
    "management",
    "custodian",
    "fund_type",
    "found_date",
    "status",
    "market",
    "benchmark",
    "invest_type",
)


def plain(text: str) -> str:
    """仅正规化排版，不改变数值、分红时点或指数定义。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).upper().replace("AU9999", "AU99.99")


def economic_date(value, label: str) -> date:
    """API日期统一为真实自然日；仅接收明确ISO或八位日期，不比较原字符串。"""
    if not isinstance(value, str) or not re.fullmatch(r"(?:\d{8}|\d{4}-\d{2}-\d{2})", value):
        raise ValueError("GOLD_DATE_FORMAT:" + label)
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("GOLD_DATE_INVALID:" + label) from error


def dividend_query(source: Originals, digest: str) -> dict:
    """空事件响应的身份取真实捕获请求；空items本身不能证明请求属于本份额。"""
    ref = source.sources[digest]
    captured = ref
    if ref.get("query_metadata_ref"):
        captured = source.json(ref["query_metadata_ref"])
        if captured.get("source", {}).get("sha256") != digest:
            raise ValueError("GOLD_DIVIDEND_QUERY_PAYLOAD_BINDING")
    params = captured.get("params")
    if (
        ref.get("url") != "https://api.tushare.pro"
        or captured.get("api_name") != "fund_div"
        or not isinstance(params, dict)
        or set(params) - {"ts_code", "limit", "offset"}
        or params.get("offset", 0) != 0
    ):
        raise ValueError("GOLD_DIVIDEND_UNFILTERED_QUERY_REQUIRED")
    return params


def table(source: Originals, digest: str) -> list[dict]:
    """原接口字段数组严格还原行；失败、重名字段、残缺行不降为缺省零。"""
    raw = json.loads(source.raw[digest], parse_float=Decimal)
    data = raw.get("data")
    if raw.get("code") != 0 or not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise ValueError("GOLD_PROVIDER_RESPONSE")
    fields = data.get("fields", [])
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("GOLD_PROVIDER_FIELDS")
    return [dict(zip(fields, row, strict=True)) for row in data["items"]]


def cohort(source: Originals, inputs: dict) -> tuple[list[str], dict]:
    """复读原市场目录及全部分页，不以手写的五个ID代替固定群的穷尽性。"""
    inventory = source.json(inputs["inventory_ref"])
    frozen = source.json(inputs["cohort_ref"])
    if (
        inputs["inventory_ref"] != INVENTORY_SHA256
        or inputs["cohort_ref"] != COHORT_SHA256
        or inventory.get("catalog_complete") is not True
        or inventory.get("duplicate_pagination") is not False
        or inventory.get("status_filter") != "L"
        or len(inventory.get("entries", [])) != 27487
        or frozen.get("source_sha256") != inputs["inventory_ref"]
        or frozen.get("directory_entry_count") != 27487
        or frozen.get("fund_code") != "000216"
        or frozen.get("as_of") != AS_OF
        or frozen.get("scores_observed_before_freeze") is not False
    ):
        raise ValueError("GOLD_MARKET_INVENTORY_OR_FREEZE")
    pages = inputs["catalog_refs"]
    if len(pages) != 28 or len(set(pages)) != 28 or set(pages) != {r["sha256"] for r in inventory["sources"]}:
        raise ValueError("GOLD_CATALOG_PAGE_SET")
    raw_rows = [row for digest in pages for row in table(source, digest)]
    if len(raw_rows) != 27487 or len({r["ts_code"] for r in raw_rows}) != 27487:
        raise ValueError("GOLD_CATALOG_ROWS_DUPLICATE_OR_GAP")
    by_code = {r["ts_code"]: r for r in raw_rows}
    for row in inventory["entries"]:
        original = by_code.get(row["ts_code"])
        if original is None or any(original.get(k) != row.get(k) for k in CATALOG_COLUMNS):
            raise ValueError("GOLD_INVENTORY_NOT_RECONCILED_TO_ORIGINAL_PAGES")
    groups, excluded = {}, []
    for row in raw_rows:
        name, benchmark = row.get("name") or "", row.get("benchmark") or ""
        if "黄金" not in name or "联接" not in name:
            continue
        code = row["ts_code"][:6]
        reason = None
        if code in TARGET_ALIASES:
            reason = "TARGET_ALL_SHARE_CLASSES_EXCLUDED"
        elif row.get("status") != "L" or row.get("market") != "O":
            reason = "OTHER_MARKET_OR_STATUS"
        elif not row.get("found_date") or row["found_date"] > "20230928":
            reason = "OWN_SHARE_UNDER_FIXED_THREE_YEAR_WINDOW"
        elif (
            not re.search(r"[Aa][Uu](?:99\.99|9999)", benchmark)
            or not re.search(r"(?<!\d)95%", benchmark)
            or not re.search(r"(?<!\d)5%", benchmark)
            or "活期" not in benchmark
        ):
            reason = "DIFFERENT_UNDERLYING_OR_GOLD_CASH_BUDGET"
        if reason:
            excluded.append({"fund_code": code, "reason": reason, "benchmark": benchmark})
        else:
            key = re.sub(r"-[A-Za-z]+$", "", name)
            groups.setdefault(key, []).append(row)
    rebuilt = []
    for name, rows in sorted(groups.items()):
        rows.sort(key=lambda r: (r["found_date"], 0 if r["name"].endswith("-A") else 1, r["ts_code"]))
        rebuilt.append(
            {
                "candidate_product_name": name,
                "representative": rows[0]["ts_code"][:6],
                "all_directory_shares": [r["ts_code"][:6] for r in rows],
            }
        )
    frozen_rows = [
        {
            "candidate_product_name": r["candidate_product_name"],
            "representative": r["representative"]["ts_code"][:6],
            "all_directory_shares": r["all_directory_shares"],
        }
        for r in frozen["representatives"]
    ]
    expected = sorted(r["representative"] for r in rebuilt)
    if (
        rebuilt != frozen_rows
        or len(expected) != 5
        or sorted(inputs["peers"]) != expected
        or TARGET_ALIASES.intersection(expected)
    ):
        raise ValueError("GOLD_FIXED_COHORT_NOT_CLOSED")
    return expected, {
        "catalog_rows": len(raw_rows),
        "rebuilt_representatives": rebuilt,
        "excluded": excluded,
        "selection_uses_returns": False,
    }


def contract(source: Originals, digest: str, code: str, period: str) -> dict:
    """实际经济报告期和同法律产品由PDF正文核准，不信调用者填写的期末。"""
    if not source.raw[digest].startswith(b"%PDF"):
        raise ValueError("GOLD_CONTRACT_ACTUAL_PDF_REQUIRED")
    pages = source.pages(digest)
    start, end = report_period("\n".join(pages))
    if str(end) != period or str(start) != period[:4] + "-01-01":
        raise ValueError("GOLD_CONTRACT_ECONOMIC_PERIOD")
    body = plain("\n".join(pages))
    names = {name for name in LEGAL_NAMES.values() if name in plain(pages[0])}
    if names != {LEGAL_NAMES[code]} or code not in body[:25000]:
        raise ValueError("GOLD_CONTRACT_LEGAL_IDENTITY")
    clauses = [m[1] for m in re.finditer(r"业绩比较基准(.{0,420})", body[:40000])]
    compatible = [
        c
        for c in clauses
        if re.search(r"(?<!\d)95%", c)
        and re.search(r"(?<!\d)5%", c)
        and "活期" in c
        and ("AU99.99" in c or "国内黄金现货价格收益率" in c)
    ]
    if not compatible or any(t in compatible[0] for t in ["上海金集中定价", "黄金产业"]):
        raise ValueError("GOLD_SAME_AU9999_95_CASH5_CLAUSE_REQUIRED")
    published = source.sources[digest].get("published_on")
    if not published or published > AS_OF:
        raise ValueError("GOLD_CONTRACT_PUBLICATION_REQUIRED")
    return {
        "fund_code": code,
        "source_sha256": digest,
        "period": period,
        "legal_name": next(iter(names)),
        "published_on": published,
        "clause": compatible[0],
    }


def history(source: Originals, member: dict, code: str, days: list[str]) -> tuple[dict, dict]:
    """完整窗只读同份额真实净值与事件；存在现金、拆分或未知状态即拒绝。"""
    rows = table(source, member["nav_ref"])
    events = table(source, member["dividends_ref"])
    params = dividend_query(source, member["dividends_ref"])
    if params.get("ts_code") != code + ".OF":
        raise ValueError("GOLD_DIVIDEND_QUERY_SHARE_IDENTITY")
    page = source.json(member["dividends_ref"])["data"]
    # 本适配只接受已证明单页完整的无现金场景；页尾未知或恰好触及上限时
    # 必须另补完整分页，不能把截断的空集或缺失页解释为“本基金无分红”。
    count = page.get("count")
    if (
        page.get("has_more") is not False
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count < 0
        or (not events and count != 0)
        or (count > 0 and count != len(events))
        or (params.get("limit") is not None and len(events) >= int(params["limit"]))
    ):
        raise ValueError("GOLD_DIVIDEND_PAGE_COMPLETENESS_REQUIRED")
    start, end = (economic_date(d, "window") for d in (days[0], days[-1]))
    cutoff = economic_date(AS_OF, "cutoff")
    seen_events = set()
    for event in events:
        if event.get("ts_code") != code + ".OF":
            raise ValueError("GOLD_WRONG_SHARE_EVENT")
        status = event.get("div_proc")
        if status not in {"实施", "预案", "股东大会通过", "终止"}:
            raise ValueError("GOLD_UNKNOWN_EVENT_STATUS")
        raw_ex = event.get("ex_date")
        if not raw_ex:
            if status == "实施":
                raise ValueError("GOLD_IMPLEMENTED_EVENT_DATE_MISSING")
            continue
        ex = economic_date(raw_ex, "event_ex_date")
        if start < ex <= end and status == "实施":
            announced = economic_date(event.get("ann_date"), "event_ann_date")
            if announced > cutoff:
                raise ValueError("GOLD_FUTURE_EVENT_PUBLICATION")
            key = (event["ts_code"], ex, str(event.get("div_cash")))
            seen_events.add(key)
    if seen_events:
        raise ValueError("GOLD_DIVIDEND_EVENT_ADAPTER_REQUIRED")
    by = {}
    for row in rows:
        if row.get("ts_code") != code + ".OF":
            raise ValueError("GOLD_WRONG_SHARE_HISTORY")
        day = str(economic_date(row["nav_date"], "nav_date"))
        if day in by:
            raise ValueError("GOLD_DUPLICATE_NAV_DATE")
        if economic_date(row.get("ann_date"), "nav_ann_date") > cutoff:
            raise ValueError("GOLD_FUTURE_NAV_PUBLICATION")
        values = [Decimal(str(row[k])) for k in ("unit_nav", "accum_nav", "adj_nav")]
        if any(not v.is_finite() or v <= 0 for v in values) or len(set(values)) != 1:
            raise ValueError("GOLD_ADJUSTMENT_REQUIRES_EVENT_REVIEW")
        by[day] = values[0]
    if not set(days) <= by.keys():
        raise ValueError("GOLD_REQUIRED_VALUATION_DATE_MISSING:" + code)
    return by, {
        "raw_rows": len(rows),
        "common_rows": len(days),
        "implemented_events": 0,
        "missing_rows_filled": 0,
        "extra_noncalendar_rows": sorted(by.keys() - set(days)),
    }


def material_review(source: Originals, inputs: dict) -> dict:
    """复读已人工核准的事项适用性工作底稿及其逐页原件，不推定全面无风险。

    事项分类是有来源的研究判断；程序核其身份、截止日、证据完整与声明
    一致性，不把管理人整改自动转换为本基金无重大未决，也不修改治理分。
    """
    digest = inputs.get("material_event_review_ref")
    if digest not in source.sources:
        raise ValueError("GOLD_MATERIAL_EVENT_REVIEW_SOURCE_REQUIRED")
    review = source.json(digest)
    if (
        review.get("fund_code") != inputs["fund_code"]
        or review.get("share_class") != inputs["share_class"]
        or review.get("as_of") != inputs["as_of_date"]
        or review.get("review_closed") is not True
        or review.get("material_event_pending") is not False
        or review.get("applicable_material_unresolved") != []
        or review.get("not_inferred_from_governance_score") is not True
        or review.get("no_normal_by_search_absence") is not True
    ):
        raise ValueError("GOLD_MATERIAL_EVENT_REVIEW_SCOPE_OR_PENDING")
    registry, evidence = review["source_registry"], review["evidence"]
    required = {
        "manager_2025_measure",
        "staff_2025_measures",
        "current_manager_operation",
        "current_custodian_operation",
        "current_material_event_section",
    }
    if not required <= evidence.keys() or not review.get("events"):
        raise ValueError("GOLD_MATERIAL_EVENT_KNOWN_FACTS_MISSING")
    for ref in registry.values():
        if ref["sha256"] not in source.raw:
            raise ValueError("GOLD_MATERIAL_EVENT_ORIGINAL_NOT_ARCHIVED")
    for item in evidence.values():
        ref = registry[item["source_id"]]
        if (
            item["source_sha256"] != ref["sha256"]
            or not item.get("original_page_text")
            or not item.get("published_on")
            or item["published_on"] > inputs["as_of_date"]
            or ref.get("published_on") != item["published_on"]
        ):
            raise ValueError("GOLD_MATERIAL_EVENT_EVIDENCE_SCOPE")
        for quote in item["original_page_text"]:
            source.claim({"source_hash": ref["sha256"], "quote": quote})
    for event in review["events"]:
        if not event.get("evidence_ids") or not set(event["evidence_ids"]) <= evidence.keys():
            raise ValueError("GOLD_MATERIAL_EVENT_CLASSIFICATION_UNSOURCED")
        if event.get("fund_specific_material_unresolved_confirmed") is not False or (
            event.get("applicability") == "FUND_CONFIRMED"
            and event.get("status") == "UNRESOLVED"
            and event.get("severity") in {"MATERIAL", "SEVERE"}
        ):
            raise ValueError("GOLD_MATERIAL_EVENT_PENDING")
    return {
        "source_hash": digest,
        "known_events_preserved": len(review["events"]),
        "original_evidence_sections": len(evidence),
        "independent_regulatory_acceptance": False,
        "governance_not_reclassified": True,
        "no_global_clean_bill": True,
    }


def calculate(inputs: dict, *, root: Path) -> dict:
    """逐次检查全部原字节；仅复用相同内容的解析结果，不延长资料有效日。"""
    if (
        inputs.get("version") != VERSION
        or inputs.get("fund_code") != "000216"
        or inputs.get("share_class") != "A"
        or inputs.get("as_of_date") != AS_OF
        or inputs.get("holding_date") != "2026-06-30"
        or inputs.get("basis") != PASSIVE_BASIS
        or inputs.get("scope") != "RETURN_STABILITY_ONLY_TOTAL_AND_GRADE_NULL"
    ):
        raise ValueError("GOLD_COMPONENT_SCOPE")
    if any(k in inputs for k in ("score", "dimensions", "observations", "quality_metrics", "packet")):
        raise ValueError("GOLD_ORIGINAL_INPUTS_REQUIRED")
    # 该布尔量是调用方明确核准并封存的研究状态，不是由缺少事件字段推断。
    # 原包没有该状态时须由指挥核原件后明确提供，不能默认为“没有重大未决”。
    if inputs.get("material_event_pending") is not False:
        raise ValueError("GOLD_EXPLICIT_MATERIAL_EVENT_REVIEW_REQUIRED")
    root = root.resolve()
    Originals(inputs, root)
    return deepcopy(_calculate_cached(canonical(inputs), str(root)))


@lru_cache(maxsize=4)
def _calculate_cached(serialized: str, root_name: str) -> dict:
    inputs, root = json.loads(serialized), Path(root_name)
    source = Originals(inputs, root)
    event_review = material_review(source, inputs)
    expected, cohort_proof = cohort(source, inputs)
    raw_calendar = table(source, inputs["calendar_ref"])
    if any(r["ts_code"] != "000846.CSI" for r in raw_calendar):
        raise ValueError("GOLD_CALENDAR_MARKET_IDENTITY")
    days = sorted({str(date.fromisoformat(r["trade_date"])) for r in raw_calendar})
    if len(days) != len(raw_calendar) or len(days) != 727 or days[0] != "2023-09-28" or days[-1] != AS_OF:
        raise ValueError("GOLD_FIXED_THREE_YEAR_CALENDAR")
    months = sorted({d[:7] for d in days})
    ends = [max(d for d in days if d[:7] == m) for m in months]
    if len(ends) != 37:
        raise ValueError("GOLD_37_MONTH_ENDS_REQUIRED")
    definition = inputs["target"]["index_definition_ref"]
    defined = plain("\n".join(source.pages(definition)))
    if (
        "华安易富黄金交易型开放式证券投资基金联接基金" not in defined[:3000]
        or "目前国内黄金现货价格收益率以AU99.99合约价格为基础进行计算" not in defined
    ):
        raise ValueError("GOLD_TARGET_AU9999_EXPLICIT_DEFINITION_REQUIRED")
    contracts, series, proofs, legal_names = {}, {}, {}, []
    members = {"000216": inputs["target"], **inputs["peers"]}
    for code, member in members.items():
        if set(member["contracts"]) != PERIODS or len(set(member["contracts"].values())) != 4:
            raise ValueError("GOLD_CROSS_PERIOD_CONTRACTS_REQUIRED")
        contracts[code] = [contract(source, h, code, period) for period, h in member["contracts"].items()]
        names = {r["legal_name"] for r in contracts[code]}
        if len(names) != 1:
            raise ValueError("GOLD_HISTORICAL_LEGAL_NAME_CONFLICT")
        legal_names.append(next(iter(names)))
        series[code], proofs[code] = history(source, member, code, days)
    if len(set(legal_names)) != len(members) or inputs["primary_ref"] != inputs["target"]["contracts"]["2026-06-30"]:
        raise ValueError("GOLD_INDEPENDENT_LEGAL_PRODUCTS_OR_PRIMARY")
    own_plain = plain("\n".join(source.pages(inputs["primary_ref"])))
    if plain(inputs["manager_name"]) not in own_plain[:15000]:
        raise ValueError("GOLD_PRIMARY_MANAGER_BINDING")
    with localcontext() as context:
        context.prec = 70
        annualization = Decimal("365.25") / Decimal((date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days)
        returns = {
            code: {
                "return_1y": values[ends[-1]] / values[ends[-13]] - 1,
                "return_3y": (values[days[-1]] / values[days[0]]) ** annualization - 1,
            }
            for code, values in series.items()
        }
        refs = sorted(source.sources)
        nodes = {
            name: {
                "kind": "rank",
                "direction": "higher",
                "target": str(returns["000216"][name]),
                "expected_products": ["CN_PUBLIC_FUND:" + c for c in expected],
                "peers": [{"product_id": "CN_PUBLIC_FUND:" + c, "value": str(returns[c][name])} for c in expected],
                "source_refs": refs,
            }
            for name in ("return_1y", "return_3y")
        }
        for name, step, count in (("monthly_success", 1, 36), ("annual_success", 12, 25)):
            rows = []
            for n in range(step, len(ends)):
                a, b = ends[n - step], ends[n]
                peer_values = [series[c][b] / series[c][a] - 1 for c in expected]
                rows.append(
                    {
                        "start": a,
                        "end": b,
                        "target": str(series["000216"][b] / series["000216"][a] - 1),
                        "reference": str(sum(peer_values) / len(peer_values)),
                    }
                )
            nodes[name] = {"kind": "windows", "expected_windows": count, "rows": rows, "source_refs": refs}
    registry = {h: {**ref, "path": str(root / ref["file"])} for h, ref in source.sources.items()}
    packet = {
        "basis": PASSIVE_BASIS,
        "fund_code": "000216",
        "as_of_date": AS_OF,
        "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
        "quality_metrics": {
            "target_product_aliases": ["CN_PUBLIC_FUND:" + c for c in sorted(TARGET_ALIASES)],
            "material_event_pending": inputs["material_event_pending"],
            "observations": nodes,
        },
        "r4_evidence_registry": registry,
        "quality_public": {"dataDates": {"nav": AS_OF, "holdings": "2026-06-30"}},
        "raw_archive": [{"kind": source.sources[h].get("archive_kind", "ORIGINAL"), "sha256": h} for h in refs],
    }
    result = _calculate_frozen(packet, PROTOCOL, {}, selected_dimensions={"return", "stability"})
    if any(result.get(k) is not None for k in ("score", "raw_score", "grade")):
        raise ValueError("GOLD_COMPONENT_TOTAL_LEAK")
    return {
        "dimensions": result["dimensions"],
        "observations": nodes,
        "standard_replay": result,
        "complete_score": None,
        "grade": None,
        "audit": {
            "material_event_review": event_review,
            "cohort": cohort_proof,
            "contracts": contracts,
            "history": proofs,
            "common_benchmark_imputed": False,
            "return_identity": "(RF-B)-median(RP-B)=RF-median(RP)",
            "stability_reference": "FROZEN_PASSIVE_PEER_ARITHMETIC_MEAN",
        },
    }


def prepare(session, request: dict, *, root: Path) -> dict:
    """候选prepare：从最新部分包合并两维，不写库、不登记、不发布。"""
    from app.services.fund_rating_pipeline import current_inputs
    from app.services.fund_rating_pipeline_prepare import archive_json, canonical_source
    from app.services.fund_rating_reviewed_snapshot import BASIS, validate_sources
    from app.services.fund_rating_reviewed_snapshot import calculate as snapshot_calculate

    code, raw = request["fund_code"], deepcopy(request["inputs"])
    previous = current_inputs(session, [code]).get(code)
    if not previous or str(previous["as_of_date"]) != request["as_of_date"]:
        raise ValueError("GOLD_CURRENT_PARTIAL_PERIOD_REQUIRED")
    bundle = deepcopy(next(m["bundle"] for m in previous["input_snapshot"]["members"] if m["fund_code"] == code))
    if (
        bundle.get("basis") != BASIS
        or bundle.get("family") != "INDEX"
        or raw.get("fund_code") != code
        or raw.get("as_of_date") != request["as_of_date"]
        or raw.get("share_class") != bundle["share_class"]
        or raw.get("holding_date") != bundle["holding_date"]
        or raw.get("legal_name") != bundle["identity"]["legal_name"]
        or raw.get("manager_name") != bundle["identity"]["manager_name"]
        or raw.get("primary_ref") != bundle["original_report_archive"]["sha256"]
        or any(
            bundle.get(k)
            for k in ("r5_calculation", "passive_components_calculation", "passive_performance_calculation")
        )
    ):
        raise ValueError("GOLD_INPUT_CURRENT_BINDING")
    for h, ref in raw["sources"].items():
        raw["sources"][h] = canonical_source(root, ref)
    calculated = calculate(raw, root=root)
    explanations = request["explanations"]
    document = bundle["reviewed_document"]
    for key in ("return", "stability"):
        explanation = explanations[key].strip()
        if len(explanation) < 40:
            raise ValueError("GOLD_EXPLANATION_REQUIRED")
        point = calculated["dimensions"][key]
        document["dimensions"][key] = {"lower": point, "upper": point, "candidate": point, "reason": explanation}
        bundle.setdefault("public_explanations", {})[key] = explanation
        bundle.get("public_metrics", {}).pop(key, None)
    bundle.update(
        research_format="single",
        research_archive=archive_json(root, document),
        gold_components_calculation={"inputs": raw, "input_archive": archive_json(root, raw)},
    )
    known = {r["sha256"] for r in bundle.setdefault("extra_archives", [])}
    bundle["extra_archives"].extend(ref for h, ref in raw["sources"].items() if h not in known)
    # 正式接入前，snapshot必须增加gold_components_calculation原件重放与输入归档一致性；
    # 这里先要求来源校验及当前部分评分，不将此局部适配当作正式发布授权。
    validate_sources(bundle, datetime.now(UTC), root=root)
    checked = snapshot_calculate(bundle, root=root)
    if checked["status"] != "PARTIAL" or checked["score"] is not None:
        raise ValueError("GOLD_CANNOT_CLAIM_COMPLETE")
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}]}
