"""从修正公告追索前次原件时，只接受精确的主体、日期、标题及公告编号。"""

from app.services.fund_earnings_batch_v4 import issuer_identity
from app.services.fund_earnings_history_v2 import canonical_title
from app.services.fund_information_history_v1 import normalize


def select_reference(rows, spec):
    """不拿修正公告、下一年度或另一公司替代原件；重复匹配留为不确定。"""
    matches = [
        r
        for r in rows
        if r["secCode"] == spec["stock"]
        and r["published_date"] == spec["published_date"]
        and canonical_title(r["title_plain"]) == canonical_title(spec["title"])
    ]
    if len(matches) != 1:
        raise ValueError("REFERENCED_ORIGINAL_NOT_UNIQUE")
    return matches[0]


def reference_identity(row, pages, spec):
    """公开目录与原件公告编号共同约束身份，不能仅按追述金额相同来配对。"""
    if select_reference([row], spec) != row:
        raise ValueError("REFERENCE_METADATA_MISMATCH")
    proof = issuer_identity(row, pages)
    wanted = normalize("公告编号：" + spec["notice_number"])
    front = normalize("\n".join(pages[:2]))
    if not proof["passed"] or front.count(wanted) != 1:
        raise ValueError("REFERENCED_ORIGINAL_BODY_IDENTITY_UNRESOLVED")
    return {
        **proof,
        "reference_notice_number": spec["notice_number"],
        "notice_number_anchor": wanted,
        "public_date_backfilled": False,
    }
