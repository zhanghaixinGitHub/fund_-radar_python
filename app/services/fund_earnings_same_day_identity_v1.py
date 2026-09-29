"""同日季报正文为全文补证身份；不把两个载体当作两次独立业绩披露。"""

import re
from datetime import date, timedelta

from app.services.fund_earnings_batch_v4 import issuer_identity
from app.services.fund_earnings_identity_v1 import occurrence
from app.services.fund_information_history_v1 import normalize


def review_same_day_quarterly_identity(target, reference, legal_name):
    """严格核对同主体、同组织代码、同公开日、同一期全文与正文。

    legal_name 必须是两份原件首页的完整法定名称，不能只匹配名称后缀。
    参考正文须自行通过证券代码检查。修订附件不走此通道，避免借普通季报
    掩盖载体版本差异；金额一致、完整修订链及预测准入仍需另外核验。
    """
    a, b = target["row"], reference["row"]
    published = date.fromisoformat(a["published_date"])
    if not re.fullmatch(r"[\u4e00-\u9fff（）()]+股份有限公司", legal_name):
        raise ValueError("FULL_LEGAL_NAME_REQUIRED")
    if (
        any(not a.get(k) or a[k] != b.get(k) for k in ("secCode", "orgId", "published_date"))
        or not re.fullmatch(r"\d{6}", a["secCode"])
        or a["announcementId"] == b["announcementId"]
        or target["receipt"]["sha256"] == reference["receipt"]["sha256"]
    ):
        raise ValueError("SAME_DAY_DISTINCT_CARRIERS_REQUIRED")
    match = re.fullmatch(r"(20\d{2}年(?:第一|第三)季度报告)全文", normalize(a["title_plain"]))
    if not match or normalize(b["title_plain"]) != match[1] + "正文":
        raise ValueError("EXACT_QUARTERLY_CARRIERS_REQUIRED")
    year, quarter = int(match[1][:4]), 1 if "第一" in match[1] else 3
    end = date(year, 3, 31) if quarter == 1 else date(year, 9, 30)
    if published <= end:
        raise ValueError("QUARTER_NOT_ENDED_AT_DISCLOSURE")
    if target.get("revision_issues") or reference.get("revision_issues"):
        raise ValueError("UNRESOLVED_ORIGINAL_REVISION")
    direct_a, direct_b = [issuer_identity(d["row"], d["pages"]) for d in (target, reference)]
    if direct_a.get("reason") == "EXPLICIT_BODY_ISSUER_DIFFERS_FROM_CATALOG" or not direct_b["passed"]:
        raise ValueError("REFERENCE_CODE_OR_TARGET_ISSUER_CONFLICT")
    anchors = []
    for document in (target, reference):
        # 同页完整标题行防止把某个更长公司名称中的子串当作目标主体。
        lines = [normalize(line) for line in document["pages"][0].splitlines() if normalize(line)]
        titles = {legal_name, legal_name + match[1], legal_name + match[1] + "全文", legal_name + match[1] + "正文"}
        if not any(line in titles for line in lines):
            raise ValueError("EXACT_COVER_LEGAL_NAME_NOT_FOUND")
        if match[1] not in normalize(document["pages"][0]):
            raise ValueError("COVER_QUARTER_MISMATCH")
        anchors.append(
            {
                "document_id": document["row"]["announcementId"],
                "source_sha256": document["receipt"]["sha256"],
                "legal_name": occurrence(document["pages"], 1, legal_name),
                "period": occurrence(document["pages"], 1, match[1]),
            }
        )
    return {
        "passed": True,
        "rule": "SAME_DAY_QUARTERLY_BODY_CODE_AND_EXACT_LEGAL_COVERS",
        "document_id": a["announcementId"],
        "reference_id": b["announcementId"],
        "stock": a["secCode"],
        "legal_name": legal_name,
        "published_date": str(published),
        "available_at": str(published + timedelta(days=1)) + "T08:00:00+08:00",
        "anchors": anchors,
        "reference_code_proof": direct_b,
        "published_date_unchanged": True,
        "original_carriers_preserved": True,
        "independent_disclosure_pair": False,
        "semantic_verified": False,
        "version_history_verified": False,
        "training_ready": False,
    }
