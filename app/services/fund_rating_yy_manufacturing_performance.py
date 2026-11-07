"""018125三项业绩的原件桥接候选；重大事项三条件与执行缺项分别保留。

冻结人工审阅摘要，逐次验证原始字节。历史研究适配器只充当净值、分红和同行
算术子组件；本组件完成公告审阅后才产生三项点分，不产生八维总分或发布动作。
"""

import json
import re
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal as D
from functools import lru_cache
from pathlib import Path

from bs4 import BeautifulSoup

from app.services.fund_rating_direct_index import Originals
from app.services.fund_rating_inputs import canonical
from app.services.fund_rating_yy_manufacturing_liquidity import check_binding as known_binding
from app.services.fund_rating_yy_manufacturing_liquidity import original_pages, require
from app.services.fund_rating_yy_manufacturing_return_research import calculate as return_calculate
from app.services.fund_rating_yy_manufacturing_risk_stability_research import calculate as risk_calculate

VERSION = "R5_018125_PERFORMANCE_ORIGINALS_20261030_1"
REVIEW_SHA = "24e0d4279ad027be14f276ebce7e0be0fa7c8aff414416b4af554aa636b4c1a8"
MEANING = "CONFIRMED_FUND_SPECIFIC_MATERIAL_UNRESOLVED_EVENTS_ONLY"
DIMS = ("return", "risk", "stability")


def confirmed_material(events):
    """冻结R5三条件；未知重大性不自行提升为已确认重大。"""
    return [
        e
        for e in events
        if e.get("applicability") == "FUND_CONFIRMED"
        and e.get("status") == "UNRESOLVED"
        and e.get("severity") in {"MATERIAL", "SEVERE"}
    ]


def _source(inputs, root):
    require(
        inputs.get("version") == VERSION
        and inputs.get("fund_code") == "018125"
        and inputs.get("share_class") == "C"
        and inputs.get("as_of_date") == "2026-09-30"
        and inputs.get("holding_date") == "2026-06-30",
        "YYP_SCOPE",
    )
    require(inputs.get("material_event_review_ref") == REVIEW_SHA, "YYP_REVIEW_NOT_FROZEN")
    require(
        not any(k in inputs for k in ["score", "dimensions", "metrics", "observations"]), "YYP_NO_PRECOMPUTED_SCORES"
    )
    source = Originals(inputs, Path(root))
    review = source.json(REVIEW_SHA)
    expected = {
        REVIEW_SHA,
        *review["sources"],
        *inputs["return_inputs"]["sources"],
        *inputs["risk_stability_inputs"]["sources"],
    }
    require(set(inputs["sources"]) == expected, "YYP_COMPLETE_SOURCE_GRAPH")
    require(inputs["sources"][REVIEW_SHA].get("archive_kind") == "DERIVED", "YYP_REVIEW_IS_JUDGMENT_NOT_ORIGINAL")
    for graph in [review["sources"], inputs["return_inputs"]["sources"], inputs["risk_stability_inputs"]["sources"]]:
        for h, ref in graph.items():
            require(
                inputs["sources"][h]["sha256"] == h and inputs["sources"][h]["archive_kind"] == ref["archive_kind"],
                "YYP_SOURCE_ROLE",
            )
    require(
        type(inputs.get("material_event_pending")) is bool and inputs["material_event_pending"] is False,
        "YYP_EXPLICIT_REVIEW_GATE_REQUIRED",
    )
    return source, review


def _review(inputs, source, review):
    require(
        review.get("fund_code") == "018125"
        and review.get("share_class") == "C"
        and review.get("as_of") == inputs["as_of_date"]
        and review.get("review_complete") is True
        and review.get("event_catalog_and_originals_review_complete") is True,
        "YYP_REVIEW_SCOPE_OR_CLOSURE",
    )
    entries = review["entries"]
    require(len(entries) == 84 and len({e["announcement_id"] for e in entries}) == 84, "YYP_84_ORIGINAL_REVIEWS")
    catalog = {}
    for h in review["catalog_refs"]:
        value = source.json(h)
        require(value.get("ErrCode") == 0, "YYP_CATALOG_FAILURE_NOT_EMPTY")
        for row in value["Data"]:
            require(row["FUNDCODE"] == "018125", "YYP_CATALOG_WRONG_SHARE")
            catalog[row["ID"]] = row
    require(len(catalog) == 109, "YYP_CATALOG_COUNT")
    wanted = {k for k, r in catalog.items() if "2023-09-28" <= r["PUBLISHDATEDesc"][:10] <= "2026-09-30"}
    require({e["announcement_id"] for e in entries} == wanted, "YYP_CATALOG_WINDOW_NOT_CLOSED")
    for e in entries:
        require(
            e["title"] == catalog[e["announcement_id"]]["TITLE"]
            and len(e["manual_judgment"]) >= 25
            and e["source_hash"] in review["sources"]
            and review["sources"][e["source_hash"]]["archive_kind"] == "ORIGINAL"
            and e["original_publication"] <= inputs["as_of_date"],
            "YYP_ORIGINAL_REVIEW_BINDING",
        )
    cache = {}
    ids = set()

    def compact(text):
        return re.sub(r"\s+", "", text)

    for c in review["claims"]:
        require(
            c["claim_id"] not in ids and review["sources"][c["source_hash"]]["archive_kind"] == "ORIGINAL",
            "YYP_CLAIM_ID_OR_ROLE",
        )
        ids.add(c["claim_id"])
        h = c["source_hash"]
        if h not in cache:
            raw = source.raw[h]
            if raw.startswith(b"%PDF"):
                body = "\n".join(original_pages(raw))
            elif raw.lstrip().startswith(b"<"):
                body = BeautifulSoup(raw, "html.parser").get_text("\n", strip=True)
            else:
                value = source.json(h)
                require(
                    value.get("success") == 1 and value["data"].get("is_ai_summary", 0) == 0,
                    "YYP_ACTUAL_NOTICE_NOT_SUMMARY",
                )
                body = BeautifulSoup(value["data"]["notice_content"], "html.parser").get_text("\n", strip=True)
            cache[h] = compact(body)
        require(compact(c["quote"]) and compact(c["quote"]) in cache[h], "YYP_QUOTE_NOT_IN_ORIGINAL:" + c["claim_id"])
    events = review["fund_gate_events"]
    require(len(events) == 86 and len({e["event_id"] for e in events}) == 86, "YYP_PRESERVED_EVENT_COUNT")
    for event in events:
        require(
            event["source_refs"]
            and set(event["source_refs"]) <= review["sources"].keys()
            and event["claim_ids"]
            and set(event["claim_ids"]) <= ids,
            "YYP_EVENT_SOURCE_CLAIMS",
        )
    pending = bool(confirmed_material(events))
    require(
        review["material_event_pending_meaning"] == MEANING
        and review["material_event_pending"] is pending
        and inputs["material_event_pending"] is pending
        and review["whole_fund_safety_inferred"] is False,
        "YYP_GATE_DERIVATION",
    )
    conflict = next(e for e in events if e["event_id"] == "603179-20260630-single-issuer-conflict")
    require(
        conflict["status"] == "UNRESOLVED" and conflict["severity"] is None and review["preserved_blockers"],
        "YYP_EXECUTION_GAP_NOT_CLEARED",
    )
    return {
        "review_ref": REVIEW_SHA,
        "catalog_entries": 84,
        "verified_claims": len(ids),
        "preserved_events": len(events),
        "material_event_pending": pending,
        "material_event_pending_meaning": MEANING,
        "whole_fund_safety_inferred": False,
        "preserved_blockers": review["preserved_blockers"],
    }


@lru_cache(maxsize=1)
def _cached(serialized, root):
    inputs = json.loads(serialized)
    source, review = _source(inputs, root)
    audit = _review(inputs, source, review)
    require(not audit["material_event_pending"], "YYP_CONFIRMED_MATERIAL_PENDING")
    a = return_calculate(inputs["return_inputs"], root=root)
    b = risk_calculate(inputs["risk_stability_inputs"], root=root)
    scores = {**a["research_scores"], **b["research_scores"]}
    require(set(scores) == set(DIMS) and all(D(0) <= D(x) <= D(100) for x in scores.values()), "YYP_MATH_RESULT_SCOPE")
    return {
        "dimensions": {k: {"lower": scores[k], "upper": scores[k], "candidate": scores[k]} for k in DIMS},
        "component_scores": scores,
        "complete_score": None,
        "grade": None,
        "performance_release_permitted": True,
        "material_event_pending": False,
        "audit": {"material_review": audit, "return": a["audit"], "risk_stability": b["audit"]},
    }


def calculate(inputs, *, root):
    """每次核全来源字节再复用纯计算缓存，禁止热缓存掩盖原件被改。"""
    _source(inputs, root)
    return deepcopy(_cached(canonical(inputs), str(Path(root).resolve())))


def check_binding(bundle, inputs):
    known_binding(bundle, inputs)
    require(not bundle.get("r5_calculation"), "YYP_REVIEWED_PARTIAL_BINDING")


def replay_snapshot_points(bundle, dimensions, *, root):
    inputs = bundle["yy_manufacturing_performance_calculation"]["inputs"]
    check_binding(bundle, inputs)
    expected = calculate(inputs, root=root)
    for key in DIMS:
        require(
            all(
                dimensions[key][field] is not None
                and D(dimensions[key][field]) == D(expected["dimensions"][key][field])
                for field in ["lower", "upper", "candidate"]
            ),
            "YYP_SNAPSHOT_POINT_MISMATCH:" + key,
        )


def prepare(session, request, *, root):
    """只读最新指针，在调用者指定根预演三项替换，其他五维不变。"""
    from app.services.fund_rating_pipeline import current_inputs
    from app.services.fund_rating_pipeline_prepare import archive_json, canonical_source
    from app.services.fund_rating_reviewed_snapshot import BASIS, validate_sources
    from app.services.fund_rating_reviewed_snapshot import calculate as snapshot_calculate

    root = Path(root).resolve()
    raw = deepcopy(request["inputs"])
    require(request["fund_code"] == "018125" and request["as_of_date"] == raw["as_of_date"], "YYP_REQUEST_SCOPE")
    previous = current_inputs(session, ["018125"]).get("018125")
    require(
        previous
        and str(previous["as_of_date"]) == raw["as_of_date"]
        and request.get("expected_current_ref") == previous["batch_id"],
        "YYP_CURRENT_CHANGED_REVIEW_REQUIRED",
    )
    bundle = deepcopy(next(m["bundle"] for m in previous["input_snapshot"]["members"] if m["fund_code"] == "018125"))
    check_binding(bundle, raw)
    require(bundle["basis"] == BASIS, "YYP_BASIS")
    raw["sources"] = {h: canonical_source(root, ref) for h, ref in raw["sources"].items()}
    for name in ["return_inputs", "risk_stability_inputs"]:
        raw[name]["sources"] = {h: raw["sources"][h] for h in raw[name]["sources"]}
    value = calculate(raw, root=root)
    doc = bundle["reviewed_document"]
    before = deepcopy(doc["dimensions"])
    for key in DIMS:
        reason = request["explanations"][key].strip()
        require(len(reason) >= 40, "YYP_BUSINESS_EXPLANATION")
        doc["dimensions"][key] = {**value["dimensions"][key], "reason": reason}
        bundle.setdefault("public_explanations", {})[key] = reason
        bundle.get("public_metrics", {}).pop(key, None)
    bundle.update(
        research_format="single",
        research_archive=archive_json(root, doc),
        yy_manufacturing_performance_calculation={"inputs": raw, "input_archive": archive_json(root, raw)},
    )
    existing = {r["sha256"] for r in bundle.setdefault("extra_archives", [])}
    bundle["extra_archives"].extend(ref for h, ref in raw["sources"].items() if h not in existing)
    require(all(doc["dimensions"][k] == v for k, v in before.items() if k not in DIMS), "YYP_OTHER_DIMENSIONS_CHANGED")
    validate_sources(bundle, datetime.now(UTC), root=root)
    replay_snapshot_points(bundle, doc["dimensions"], root=root)
    actual = snapshot_calculate(bundle, root=root)
    require(actual["status"] == "PARTIAL" and actual["score"] is None, "YYP_NO_EIGHT_DIMENSION_SCORE")
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}]}
