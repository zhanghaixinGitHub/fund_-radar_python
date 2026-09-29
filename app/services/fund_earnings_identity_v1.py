"""报告身份补证：公司简介中的股票代码与法定名称，或已核实的同名主体。

仅补证原件属于谁、哪期报告；不改变公告公开日，不把“更正前”附件倒填为
早期原件，不把身份通过当作金额、版本链或训练准入通过。
"""

import re

from app.services.fund_earnings_batch_v4 import issuer_identity
from app.services.fund_earnings_history_v2 import canonical_title
from app.services.fund_earnings_titles_v3 import identity_matching_text
from app.services.fund_information_history_v1 import normalize


def occurrence(pages, number, text):
    """身份名称可在页眉重复；保留全部出现位置，不冒充金额的唯一定位锚点。"""
    raw, wanted = normalize(pages[number - 1]), normalize(text)
    offsets = [m.start() for m in re.finditer(re.escape(wanted), raw)]
    if not wanted or not offsets:
        raise ValueError("IDENTITY_ANCHOR_MISSING")
    return {"page": number, "text": wanted, "offsets": offsets}


def title_proof(row, pages):
    """全文/正文是载体说明；摘要仍须明确匹配。原始修订后缀另行完整保留。"""
    text = identity_matching_text(row["title_plain"])
    text = re.sub(r"[（(]更正前[）)]", "", text)
    text = text.replace("报告全文", "报告").replace("报告正文", "报告")
    text = text.replace("半年报业绩预告", "半年度业绩预告")
    title = canonical_title(text)
    start = re.search(r"20\d{2}(?:年|年度)", title)
    wanted = title[start.start() :] if start else title
    hits = [i + 1 for i, p in enumerate(pages[:2]) if wanted in canonical_title(p)]
    return {
        "passed": bool(wanted and hits),
        "equivalent_title": wanted,
        "pages": hits,
        "original_title": row["title_plain"],
    }


def profile_proof(document):
    """在前十五页公司信息表核对同页代码及法定名称，并回核封面主体名称。"""
    row, pages = document["row"], document["pages"]
    title = title_proof(row, pages)
    if not title["passed"] or document.get("revision_issues"):
        return None
    if issuer_identity(row, pages).get("reason") == "EXPLICIT_BODY_ISSUER_DIFFERS_FROM_CATALOG":
        return None
    code = re.compile(r"股票\s*代码\s*" + re.escape(row["secCode"]) + r"(?!\d)")
    for number, page in enumerate(pages[:15], 1):
        match = code.search(page)
        name = re.search(r"公司(?:的)?中文名称([\u4e00-\u9fff（）()]+?股份有限公司)", normalize(page))
        if not match or not name:
            continue
        legal = name[1]
        front = next((i + 1 for i, p in enumerate(pages[:2]) if legal in normalize(p)), None)
        if front is None:
            continue
        return {
            "passed": True,
            "rule": "SAME_PDF_COMPANY_PROFILE_CODE_LEGAL_NAME_AND_COVER",
            "legal_name": legal,
            "title": title,
            "anchors": [
                occurrence(pages, number, match[0]),
                occurrence(pages, number, name[0]),
                occurrence(pages, front, legal),
            ],
        }
    return None


def supplement_identity(document, profile_references):
    """无代码季报可依同名封面和此前已核实公司简介补证；不依赖后来的更名。"""
    row, pages = document["row"], document["pages"]
    direct, title = issuer_identity(row, pages), title_proof(row, pages)
    base = {
        "document_id": row["announcementId"],
        "stock": row["secCode"],
        "published_date": row["published_date"],
        "source_sha256": document["receipt"]["sha256"],
        "original_title": row["title_plain"],
        "title": title,
        "published_date_unchanged": True,
        "semantic_verified": False,
        "version_history_verified": False,
        "training_ready": False,
    }
    if document.get("revision_issues") or direct.get("reason") == "EXPLICIT_BODY_ISSUER_DIFFERS_FROM_CATALOG":
        return {**base, "passed": False, "reason": "SOURCE_IDENTITY_OR_DATE_CONFLICT"}
    profile = profile_proof(document)
    if profile:
        return {**base, **profile}
    if not title["passed"]:
        return {**base, "passed": False, "reason": "REPORT_TITLE_UNRESOLVED"}
    if direct["stock_verified"]:
        return {
            **base,
            "passed": True,
            "rule": "EXISTING_CODE_PROOF_WITH_EXPLICIT_TITLE_EQUIVALENCE",
            "anchors": [
                occurrence(pages, i + 1, row["secCode"])
                for i, p in enumerate(pages[:2])
                if row["secCode"] in normalize(p)
            ],
        }
    references = []
    for reference in profile_references:
        other, proof = reference["document"], reference["proof"]
        if (
            other["row"]["secCode"] != row["secCode"]
            or other["row"]["published_date"] > row["published_date"]
            or not proof["passed"]
        ):
            continue
        legal = proof["legal_name"]
        page = next((i + 1 for i, p in enumerate(pages[:2]) if legal in normalize(p)), None)
        if page:
            references.append(
                {
                    "document_id": other["row"]["announcementId"],
                    "source_sha256": other["receipt"]["sha256"],
                    "legal_name": legal,
                    "profile_proof": proof,
                    "target_anchor": occurrence(pages, page, legal),
                }
            )
    if len({r["legal_name"] for r in references}) == 1:
        return {
            **base,
            "passed": True,
            "rule": "COVER_LEGAL_NAME_AND_EARLIER_VERIFIED_COMPANY_PROFILE",
            "references": references,
        }
    return {**base, "passed": False, "reason": "NO_UNAMBIGUOUS_EARLIER_LEGAL_NAME_REFERENCE"}
