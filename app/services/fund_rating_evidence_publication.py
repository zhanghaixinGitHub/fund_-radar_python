"""从已核指数原件生成正式的一年口径评分包。"""

from copy import deepcopy
from pathlib import Path

from app.services.fund_market_review import rating_review_required
from app.services.fund_rating_evidence_preview import REVISION, preview
from app.services.fund_rating_pipeline import archive_json
from app.services.fund_rating_reviewed_quality import PASSIVE_BASIS, policy
from app.services.fund_rating_reviewed_snapshot import BASIS, KEYS

PUBLIC_EXPLANATIONS = {
    "return": "按近一年本份额总回报与合同基准的差值评价。",
    "risk": "按近一年净值从高点到随后低点的最大跌幅评价。",
    "execution": "按近一年的日净值相对合同基准的跟踪误差评价。",
    "stability": "按近十二个完整月达到或超过合同基准的月份比例评价。",
    "management": "按已核对的经理职责、交接及适用的管理事项评价。",
    "holdings": "按公开报告主要股票中符合目标指数成分的持仓金额比例评价。联接基金先核对目标基金。",
    "liquidity": "按合同赎回安排及报告期内核实的实际运作评价。",
    "fee": "按本份额一万元持有一年情景下的公开费率评价。",
}


def prepare_entry(bundle: dict, root: Path) -> dict:
    """复用身份与考察期，独立封存本规则使用的资料。"""
    code = bundle["fund_code"]
    if bundle["family"] != "INDEX" or rating_review_required(code):
        raise ValueError("RATING_EVIDENCE_V1_SCOPE_OR_EVENT_REVIEW")
    if bundle.get("evidence_v1_calculation"):
        return {k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}
    raw = bundle.get("r5_calculation") or bundle.get("passive_components_calculation")
    if raw is None:
        raise ValueError("RATING_EVIDENCE_V1_NEEDS_INDEX_ORIGINALS")
    packet = raw["packet"]
    inputs = packet.get("direct_index_compilation") or packet.get("passive_index_compilation")
    if not inputs or packet["basis"] != PASSIVE_BASIS:
        raise ValueError("RATING_EVIDENCE_V1_NEEDS_INDEX_ORIGINALS")
    frozen = {
        "revision": REVISION,
        "inputs": deepcopy(inputs),
        "events": raw["anchors"].get("material_events", {}).get(code, []),
    }
    result = preview({**bundle, "evidence_v1_calculation": frozen}, root, review_required=False)
    if result["status"] != "CANDIDATE" or result["score"] is None:
        raise ValueError("RATING_EVIDENCE_V1_INCOMPLETE:" + ",".join(result.get("missing", [])))
    retained = set(result["facts"]["source_sha256"])
    frozen["inputs"]["sources"] = {k: v for k, v in inputs["sources"].items() if k in retained}
    for name in ("cohort", "market", "peers"):
        frozen["inputs"].pop(name, None)
    weights = policy(PASSIVE_BASIS)["weights"]
    doc = {
        "code": code,
        "as_of": bundle["as_of_date"],
        "outer_weights_percent": {key: weights[value] for key, value in KEYS.items()},
        "dimensions": {key: result["dimensions"][value] for key, value in KEYS.items()},
    }
    entry = {
        key: bundle[key]
        for key in (
            "fund_code",
            "family",
            "public_name",
            "share_class",
            "as_of_date",
            "holding_date",
            "verified_at",
            "identity",
            "identity_archive",
            "original_report_archive",
            "report_title",
            "report_url",
            "report_published_on",
            "data_dates",
        )
    }
    if "existing_catalog_binding" in bundle:
        entry["existing_catalog_binding"] = bundle["existing_catalog_binding"]
    return {
        **entry,
        "research_format": "single",
        "research_archive": archive_json(root, doc),
        "evidence_v1_calculation": frozen,
        "evidence_v1_archive": archive_json(root, frozen),
        "public_explanations": PUBLIC_EXPLANATIONS,
        "public_limitations": ["评分以近一年及已披露的资料为准。持仓和赎回评价以所引资料期间为限。"],
        "comparison_product_count": 1,
    }


def dossier_for(bundles: list[dict], root: Path) -> dict:
    return {"basis": BASIS, "entries": [prepare_entry(bundle, root) for bundle in bundles]}
