from app.services.fund_rating_liquidity_preview import score
from app.services.fund_rating_reviewed_quality import PASSIVE_BASIS, policy

WEIGHTS = policy(PASSIVE_BASIS)["weights"]
DIMENSIONS = dict.fromkeys(WEIGHTS, "70")
SUBSCORES = {"market_capacity": "90", "restricted": "60", "holders": "80"}


def test_payment_evidence_does_not_change_the_fixed_scoring_formula():
    verified = score(DIMENSIONS, {**SUBSCORES, "known_payment": "0"}, WEIGHTS, event_review=False)
    pending = score(DIMENSIONS, SUBSCORES, WEIGHTS, event_review=False)
    assert verified["dimensions"]["liquidity"] == pending["dimensions"]["liquidity"] == "80"
    assert verified["score"] == pending["score"] == "71.000000000000"
    assert verified["payment"] == {"status": "VERIFIED", "point": "0"}
    assert pending["payment"] == {"status": "PENDING_EVIDENCE", "point": None}


def test_required_data_and_event_review_still_block_total():
    missing = score(DIMENSIONS, {"restricted": "60", "holders": "80"}, WEIGHTS, event_review=False)
    assert missing["score"] is None
    assert missing["dimensions"]["liquidity"] is None
    assert missing["missing_required_liquidity_items"] == ["market_capacity"]
    assert score({**DIMENSIONS, "risk": None}, SUBSCORES, WEIGHTS, event_review=False)["score"] is None
    assert score(DIMENSIONS, SUBSCORES, WEIGHTS, event_review=True)["score"] is None
