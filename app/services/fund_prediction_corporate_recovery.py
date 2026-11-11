"""逐次重验已完成公司事项的报价用途；不生成替代价格、基金收款或换股数量。"""

from copy import deepcopy

from app.services.direction_1d_protocol import window
from app.services.fund_corporate_actions import verified_security_quote_exception
from app.services.fund_exposure_common import now


def verified_codes(material, base):
    at = now()
    current = window(at)
    if current["base_nav_date"] != base:
        return set()
    data = {**material, "window": current, "as_of": at.isoformat()}
    documents = {d["id"]: d for d in material.get("documents", [])}
    return {
        c["code"]
        for c in material.get("companies", [])
        if c["code"] in material.get("market_quote_exceptions", {})
        and verified_security_quote_exception(data, documents, c)
    }


def refresh(material, current):
    receipts = []
    documents = {d["id"]: d for d in material.get("documents", [])}
    companies = {c["code"]: c for c in material.get("companies", [])}
    for security, item in material.get("market_quote_exceptions", {}).items():
        previous = deepcopy(item)
        item.update(base_date=current["base_nav_date"], target_date=current["target_nav_date"])
        data = {**material, "window": current, "as_of": now().isoformat()}
        proof = verified_security_quote_exception(data, documents, companies[security])
        if proof is None:
            item.clear()
            item.update(previous)
        receipts.append(
            {
                "security": security,
                "status": "REVALIDATED_NOT_APPLICABLE" if proof else "ACTION_REQUIRED",
                "proof": proof,
                "replacement_price_created": False,
                "fund_current_cash_or_share_quantity_claimed": False,
            }
        )
    return receipts
