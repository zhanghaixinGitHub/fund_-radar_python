"""有版本统一行业研究映射及信用判断后的018125持仓点分候选。

完整保留旧原件审阅与真实敏感性。行业归类为明确的人工研究映射，不冒充协会
发布的港股分类通知；质量50来自已证信用事实而非资料缺失或区间中点。
"""

import json
import re
import unicodedata
from collections import defaultdict
from copy import deepcopy
from decimal import Decimal as D
from decimal import localcontext
from functools import lru_cache
from pathlib import Path

from app.services.fund_rating_direct_index import Originals
from app.services.fund_rating_inputs import canonical
from app.services.fund_rating_r5 import exposure_factor, peer_gap_score
from app.services.fund_rating_yy_manufacturing_holdings_bound import (
    calculate as original_review_calculate,
)
from app.services.fund_rating_yy_manufacturing_holdings_bound import (
    check_binding as original_binding,
)
from app.services.fund_rating_yy_manufacturing_holdings_bound import (
    verify_registered_original as original_verify,
)
from app.services.fund_rating_yy_manufacturing_liquidity import require
from app.services.fund_rating_yy_manufacturing_market_liquidity import PEERS, _books, _pages

VERSION = "R5_018125_REVIEWED_HOLDINGS_POINT_20261030_1"
REVIEW = "001a7670bd5c333e5db229908477fa6d792773190e4f11ad51478dfb4218e419"
MAPPINGS = {"00179.HK": "C38", "02507.HK": "C37", "09880.HK": "C34"}


def compact(text):
    """仅统一原件空白与全半角，不改动数值或业务文字。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def _source(inputs, root):
    require(
        inputs.get("version") == VERSION
        and inputs.get("fund_code") == "018125"
        and inputs.get("share_class") == "C"
        and inputs.get("as_of_date") == "2026-09-30",
        "YYHR_SCOPE",
    )
    require(
        inputs.get("review_ref") == REVIEW
        and not any(k in inputs for k in ["score", "dimensions", "Q", "industry_HHI"]),
        "YYHR_FROZEN_REVIEW_NO_SCORES",
    )
    source = Originals(inputs, Path(root))
    review = source.json(REVIEW)
    require(
        set(inputs["sources"]) == set(inputs["original_inputs"]["sources"]) | set(review["sources"]) | {REVIEW},
        "YYHR_COMPLETE_SOURCE_GRAPH",
    )
    require(
        review["formula_weights_and_peer_group_changed"] is False
        and review["professional_judgment_not_automatic_keyword_rating"] is True
        and review["as_of"] == inputs["as_of_date"]
        and review["holding_date"] == "2026-06-30",
        "YYHR_REVIEW_SCOPE",
    )
    require(inputs["sources"][REVIEW]["archive_kind"] == "DERIVED", "YYHR_REVIEW_NOT_ORIGINAL")
    for h, ref in review["sources"].items():
        require(
            inputs["sources"][h]["archive_kind"] == "ORIGINAL" and ref["archive_kind"] == "ORIGINAL",
            "YYHR_GENUINE_NEW_ORIGINALS",
        )
    return source, review


def _review(source, review):
    ids = set()
    cache = {}
    for claim in review["claims"]:
        h = claim["source_hash"]
        require(claim["claim_id"] not in ids and h in review["sources"], "YYHR_CLAIM_SCOPE")
        ids.add(claim["claim_id"])
        if h not in cache:
            cache[h] = _pages(source, h)
        page = claim["physical_page"]
        require(
            1 <= page <= len(cache[h]) and compact(claim["quote"]) in compact(cache[h][page - 1]),
            "YYHR_ORIGINAL_CLAIM:" + claim["claim_id"],
        )
    require(
        len(ids) == 28 and {k: v["industry_code"] for k, v in review["mappings"].items()} == MAPPINGS,
        "YYHR_MAPPING_VERSION",
    )
    for node in review["mappings"].values():
        require(
            node["official_company_assignment_claimed"] is False
            and len(node["judgment"]) >= 80
            and set(node["claim_ids"]) <= ids,
            "YYHR_EXPLICIT_RESEARCH_MAPPING",
        )
    q = review["quality_resolution"]
    require(
        q["security"] == "09880.HK"
        and q["manual_q"] == "50"
        and q["new_numeric_materiality_threshold_created"] is False
        and q["alternative_75_sensitivity_preserved"] is True
        and set(q["claim_ids"]) <= ids,
        "YYHR_CREDIT_JUDGMENT",
    )


def _industry(inputs, books, mapping):
    old = inputs["original_inputs"]
    sectors = {code: row["industry_code"] for code, row in old["classification_rows"].items()}
    sectors.update({code: row["industry_code"] for code, row in old["IPO_classifications"].items()})
    result = {}
    for code, book in books.items():
        buckets = defaultdict(lambda: D(0))
        for security, value in book["holdings"].items():
            if security.endswith(".HK"):
                sector = old["HK_legal"][security]["industry"] if security == "06869.HK" else mapping[security]
            else:
                sector = sectors[security[:6]]
            require(re.fullmatch("[A-S]\\d{2}", sector) is not None, "YYHR_SAME_GRAIN_REQUIRED")
            buckets[sector] += value
        require(sum(buckets.values()) == book["total"], "YYHR_FULL_INDUSTRY_AMOUNT")
        h = sum((v / book["total"]) ** 2 for v in buckets.values())
        result[code] = {"industry_HHI": h, "buckets_CNY": dict(buckets), "stock_total_CNY": book["total"]}
    target = result["018125"]["industry_HHI"]
    score = peer_gap_score(target, [result[p]["industry_HHI"] for p in PEERS]) * exposure_factor(
        "industry_concentration", target
    )
    return score, result


def _compile(inputs, source, review, root):
    _review(source, review)
    # 沿旧组件从所有财报/完整目录/法律主体及非股票风险逐一重放；不绕开原件审阅。
    prior = original_review_calculate(inputs["original_inputs"], root=root)
    with localcontext() as ctx:
        ctx.prec = 65
        books = _books(source, inputs["original_inputs"]["capacity"])
        industry, structure = _industry(inputs, books, MAPPINGS)
        target = books["018125"]
        weighted = D(0)
        qs = {}
        for r in inputs["original_inputs"]["manual_asset_quality"]:
            q = D(review["quality_resolution"]["manual_q"]) if r["security"] == "09880.HK" else D(r["manual_q"])
            weighted += target["holdings"][r["security"]] * q
            qs[r["security"]] = str(q)
        quality = weighted / target["total"]
        issuer = D(prior["issuer_score"])
        point = D(".35") * issuer + D(".35") * industry + D(".30") * quality
        sensitivities = {}
        for key, changes in [("UB_industry_C39", {"09880.HK": "C39"}), ("JE_industry_C36", {"00179.HK": "C36"})]:
            score, _ = _industry(inputs, books, {**MAPPINGS, **changes})
            sensitivities[key] = D(".35") * issuer + D(".35") * score + D(".30") * quality
        alternate_q = quality + target["holdings"]["09880.HK"] / target["total"] * D(25)
        sensitivities["UB_q75"] = D(".35") * issuer + D(".35") * industry + D(".30") * alternate_q

    def strings(value):
        if isinstance(value, D):
            return str(value)
        if isinstance(value, dict):
            return {k: strings(v) for k, v in value.items()}
        if isinstance(value, list):
            return [strings(v) for v in value]
        return value

    return strings(
        {
            "dimensions": {"holdings": {"lower": point, "upper": point, "candidate": point}},
            "component_scores": {"holdings": point},
            "complete_score": None,
            "grade": None,
            "issuer_score": issuer,
            "industry_score": industry,
            "asset_quality_Q": quality,
            "sensitivity_not_alternate_admitted_points": sensitivities,
            "audit": {
                "review_ref": REVIEW,
                "manual_research_mappings": MAPPINGS,
                "official_HK_classification_claimed": False,
                "structure": structure,
                "all_24_quality_judgments": qs,
                "full_stock_amount_coverage": "1",
                "original_review": prior["audit"],
                "quality_50_not_missing_default": True,
                "formula_weights_and_peer_group_changed": False,
            },
        }
    )


@lru_cache(maxsize=1)
def _cached(serialized, root):
    inputs = json.loads(serialized)
    source, review = _source(inputs, root)
    return _compile(inputs, source, review, root)


def calculate(inputs, *, root):
    _source(inputs, root)
    return deepcopy(_cached(canonical(inputs), str(Path(root).resolve())))


def check_binding(bundle, inputs):
    original_binding(bundle, inputs["original_inputs"])


def verify_registered_original(bundle, ref, *, root):
    inputs = bundle["yy_manufacturing_reviewed_holdings_calculation"]["inputs"]
    check_binding(bundle, inputs)
    facade = {**bundle, "yy_manufacturing_holdings_bound_calculation": {"inputs": inputs["original_inputs"]}}
    original_verify(facade, ref, root=root)


def replay_snapshot_points(bundle, dimensions, *, root):
    inputs = bundle["yy_manufacturing_reviewed_holdings_calculation"]["inputs"]
    check_binding(bundle, inputs)
    expected = calculate(inputs, root=root)["dimensions"]["holdings"]
    require(
        all(
            dimensions["holdings"][k] is not None and D(dimensions["holdings"][k]) == D(expected[k])
            for k in ["lower", "upper", "candidate"]
        ),
        "YYHR_SNAPSHOT_ORIGINAL_POINT",
    )


def prepare(session, request, *, root):
    from app.services.fund_rating_pipeline import current_inputs
    from app.services.fund_rating_pipeline_prepare import archive_json, canonical_source
    from app.services.fund_rating_reviewed_snapshot import BASIS

    root = Path(root).resolve()
    raw = deepcopy(request["inputs"])
    current = current_inputs(session, ["018125"]).get("018125")
    require(
        current
        and current["batch_id"] == request["expected_current_ref"]
        and request["as_of_date"] == str(current["as_of_date"]),
        "YYHR_CURRENT_CHANGED",
    )
    bundle = deepcopy(next(m["bundle"] for m in current["input_snapshot"]["members"] if m["fund_code"] == "018125"))
    check_binding(bundle, raw)
    require(bundle["basis"] == BASIS and not bundle.get("r5_calculation"), "YYHR_PARTIAL_CURRENT")
    raw["sources"] = {h: canonical_source(root, ref) for h, ref in raw["sources"].items()}
    old = raw["original_inputs"]
    old["sources"] = {h: raw["sources"][h] for h in old["sources"]}
    value = calculate(raw, root=root)
    doc = bundle["reviewed_document"]
    before = deepcopy(doc["dimensions"])
    reason = request["explanations"]["holdings"].strip()
    require(len(reason) >= 40, "YYHR_EXPLANATION")
    doc["dimensions"]["holdings"] = {**value["dimensions"]["holdings"], "reason": reason}
    bundle.setdefault("public_explanations", {})["holdings"] = reason
    bundle.get("public_metrics", {}).pop("holdings", None)
    bundle.pop("yy_manufacturing_holdings_bound_calculation", None)
    bundle.update(
        research_format="single",
        research_archive=archive_json(root, doc),
        yy_manufacturing_reviewed_holdings_calculation={"inputs": raw, "input_archive": archive_json(root, raw)},
    )
    known = {r["sha256"] for r in bundle.setdefault("extra_archives", [])}
    bundle["extra_archives"].extend(ref for h, ref in raw["sources"].items() if h not in known)
    require(
        all(doc["dimensions"][k] == v for k, v in before.items() if k != "holdings"), "YYHR_OTHER_DIMENSIONS_CHANGED"
    )
    replay_snapshot_points(bundle, doc["dimensions"], root=root)
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}]}
