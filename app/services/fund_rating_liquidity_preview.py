"""按固定 R5 试验口径重放流动性，产出与正式发布隔离的对照结果。"""

from datetime import UTC, datetime
from decimal import Decimal, localcontext
from pathlib import Path

from app.services.fund_rating_r5 import calculate as calculate_r5
from app.services.fund_rating_r5 import number
from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS, PASSIVE_BASIS, policy
from app.services.fund_rating_reviewed_snapshot import KEYS, calculate, validate_sources
from app.services.material_content import material_read_cache

REVISION = "R5_LIQUIDITY_DISCLOSURE_PREVIEW_V1"
REQUIRED = {"market_capacity": 40, "restricted": 20, "holders": 15}
BASES = {"ACTIVE_EQUITY": ACTIVE_BASIS, "INDEX": PASSIVE_BASIS}


def score(dimensions: dict, subscores: dict, weights: dict, *, event_review: bool) -> dict:
    """付款项是否存在不改变计分口径，也不改变该项的核查状态。"""
    missing = [key for key in REQUIRED if subscores.get(key) is None]
    payment = subscores.get("known_payment")
    result = dict(dimensions)
    result["liquidity"] = None
    with localcontext() as context:
        context.prec = 60
        if not missing:
            points = {key: number(subscores[key]) for key in REQUIRED}
            if any(not 0 <= value <= 100 for value in points.values()):
                raise ValueError("RATING_LIQUIDITY_PREVIEW_SUBSCORE_RANGE")
            weighted = sum(points[key] * weight for key, weight in REQUIRED.items())
            result["liquidity"] = str(weighted / sum(REQUIRED.values()))
        total = None
        if not event_review and all(result[key] is not None for key in weights):
            total = sum(number(result[key]) * weight / 100 for key, weight in weights.items()).quantize(
                Decimal(".000000000001")
            )
    return {
        "dimensions": result,
        "score": str(total) if total is not None else None,
        "missing_required_liquidity_items": missing,
        "payment": {"status": "VERIFIED" if payment is not None else "PENDING_EVIDENCE", "point": payment},
        "event_review_required": event_review,
    }


@material_read_cache
def preview(bundle: dict, *, root: Path, event_review: bool, mask_payment_check: bool = False) -> dict:
    """只使用现有评分包或已接入的原件重放，拒绝从区间中点反推子项。"""
    basis = BASES.get(bundle.get("family"))
    if basis is None:
        return {"revision": REVISION, "status": "UNSUPPORTED_FAMILY"}
    validate_sources(bundle, datetime.now(UTC), root=root)
    weights = policy(basis)["weights"]
    recorded_weights = bundle["reviewed_document"]["outer_weights_percent"]
    if set(recorded_weights) != set(KEYS) or any(
        number(recorded_weights[key]) != weights[value] for key, value in KEYS.items()
    ):
        return {"revision": REVISION, "status": "UNSUPPORTED_WEIGHTS"}
    raw = bundle.get("r5_calculation")
    strict_score = None
    subscores = {}
    if raw:
        packet = raw["packet"]
        if packet["basis"] != basis or any(packet[key] != bundle[key] for key in ("fund_code", "as_of_date")):
            raise ValueError("RATING_LIQUIDITY_PREVIEW_PACKET_SCOPE")
        strict = calculate_r5(packet, raw["anchors"], root=root)
        dimensions = strict["dimensions"]
        strict_score = strict["score"]
        subscores = strict["subscores"]
        event_review = event_review or bool(strict["confirmed_material_event_restrictions"])
    else:
        strict = calculate(bundle, root=root)
        dimensions = {KEYS[key]: item["candidate"] for key, item in strict["dimensions"].items()}
        if bundle.get("health_stock_liquidity_interval_calculation"):
            from app.services.fund_rating_health_stock_liquidity_interval import replay_snapshot_interval

            interval = replay_snapshot_interval(bundle, strict["dimensions"], root=root)
            subscores = interval["subscores"]

    candidate = score(dimensions, subscores, weights, event_review=event_review)
    status = "EVENT_REVIEW" if event_review else "CANDIDATE" if candidate["score"] is not None else "MISSING_INPUT"
    result = {
        "revision": REVISION,
        "status": status,
        "as_of_date": bundle["as_of_date"],
        "strict_score": strict_score,
        "strict_liquidity": dimensions["liquidity"],
        "verified_liquidity_items": {key: subscores[key] for key in REQUIRED if key in subscores},
        **candidate,
    }
    if mask_payment_check and subscores.get("known_payment") is not None:
        masked = {key: value for key, value in subscores.items() if key != "known_payment"}
        result["payment_mask_check"] = score(dimensions, masked, weights, event_review=event_review)
    return result
