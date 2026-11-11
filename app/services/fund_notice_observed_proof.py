"""同日公告的公开时间上界：以真实首次公开响应证明截止前可见，不伪造发布时间。"""

import json
import re
from datetime import datetime

from app.integrations.market_fund_reports import read_source
from app.integrations.public_fund_reports import BODY_URL
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import read
from app.services.fund_workflow_materials import workflow_root


def validate(document, code, cutoff, *, root=None):
    proof = document.get("publication_observed_proof") or {}
    article = proof.get("article_id", "")
    if (
        proof.get("kind") != "PUBLIC_RESPONSE_FIRST_OBSERVED"
        or not re.fullmatch(r"AN\d{18}", article)
        or document.get("kind") != "ANNOUNCEMENT"
        or code not in document.get("codes", [])
    ):
        return False
    root = workflow_root(root) / "news"
    try:
        receipt = read(root / "body-receipts" / code / (article + ".json"))
        state = read(root / "funds" / (code + ".json"))
        item = state["items"][article]
        observed = datetime.fromisoformat(receipt["received_at"])
        raw = json.loads(read_source(root, receipt))
        data = raw.get("data") or {}
        if (
            receipt.get("url") != BODY_URL
            or receipt.get("params", {}).get("art_code") != article
            or digest(receipt) != proof.get("receipt_hash")
            or item.get("source") != receipt
            or receipt["sha256"] != document.get("raw_sha256")
            or proof.get("observed_at") != receipt["received_at"]
            or document.get("available_at") != receipt["received_at"]
            or observed.tzinfo is None
            or observed > cutoff
            or document.get("published_date") != item["entry"]["published"]
            or raw.get("success") != 1
            or data.get("art_code") != article
            or data.get("notice_content") != document.get("body")
        ):
            return False
        if code not in {s.get("stock") for s in data.get("security", [])}:
            from app.services.fund_news_identity import checked_applicability

            if checked_applicability(root, code, item["entry"], receipt) is None:
                return False
        return True
    except (ValueError, KeyError, TypeError, OSError):
        return False
