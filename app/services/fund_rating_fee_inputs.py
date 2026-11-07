"""本份额公开费用原件独立编译；复用R5万元持有365天公式，无同行依赖。"""

from decimal import localcontext
from pathlib import Path

from app.services.fund_rating_direct_index import Originals
from app.services.fund_rating_market_review import annual_fee_score, fee_profile, normalized_text
from app.services.fund_rating_r5 import PROTOCOL, fee_score
from app.services.fund_rating_reviewed_quality import policy

VERSION = "R5_PUBLIC_FEE_ORIGINALS_20261029_1"


def calculate(inputs: dict, *, root: Path) -> dict:
    """只从真实本份额概要取费率，保留日期、申购方式和金额分解，不接收分数。"""
    if inputs.get("version") != VERSION or policy(inputs["basis"])["terms"]["cost"] != [["cost_365", 100]]:
        raise ValueError("R5_FEE_FORMULA_OR_SCOPE")
    source = Originals(inputs, root)
    digest = inputs["summary_ref"]
    reference = source.sources[digest]
    if reference.get("archive_kind") == "DERIVED" or not reference.get("published_on"):
        raise ValueError("R5_FEE_ORIGINAL_REQUIRED")
    if inputs.get("prospectus_ref") and source.sources[inputs["prospectus_ref"]].get("archive_kind") == "DERIVED":
        raise ValueError("R5_FEE_ORIGINAL_REQUIRED")
    summary = "\n".join(source.pages(digest))
    supporting = "\n".join(source.pages(inputs["prospectus_ref"])) if inputs.get("prospectus_ref") else ""
    if supporting:
        header = normalized_text(supporting)[:1000]
        if normalized_text(inputs['legal_name']) not in header or '招募说明书' not in header:
            raise ValueError('R5_FEE_SUPPORTING_PRODUCT_IDENTITY')
    profile = fee_profile(
        summary,
        code=inputs["fund_code"],
        share_class=inputs["share_class"],
        legal_name=inputs["legal_name"],
        supporting_text=supporting,
        published_on=reference["published_on"],
    )
    if profile["published_on"] > inputs["as_of_date"]:
        raise ValueError("R5_FEE_AFTER_CUTOFF")
    cost = annual_fee_score(profile["fees"])
    with localcontext() as context:
        context.prec = 60
        score = fee_score(cost["cost_cny"], PROTOCOL["parameters"]["annual_cost_unit_per_10000_CNY"])
    return {
        "score": str(score),
        "fees": profile["fees"],
        "cost": cost,
        "published_on": profile["published_on"],
        "calculation_revision": PROTOCOL["revision"],
        "scope": "PUBLIC_FEES_10000_CNY_365_DAYS_NO_GAIN_NO_UNPUBLISHED_TRADING_COST_ESTIMATE",
    }
