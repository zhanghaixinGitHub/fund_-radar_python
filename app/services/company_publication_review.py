"""复用已完成的当前公告逐份核验；只补披露事实，不授予训练或财务语义资格。"""

import hashlib
import json
import re
import unicodedata
from datetime import datetime

from app.services.fund_exposure_common import ROOT
from app.services.fund_exposure_supplement import verified_bytes

# 这是已验收专项证据的固定入口，不扫描其他研究目录，也不运行补数/OCR脚本。
REVIEW_ROOT = ROOT / "data-completion/20260930-v1"
_BINDING = ("announcementId", "secCode", "adjunctUrl", "adjunctSize", "announcementTitle", "announcementTime")


def _read(path, expected_hash=None):
    raw = path.read_bytes()
    if expected_hash is not None and hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError("PUBLICATION_REVIEW_HASH_MISMATCH")
    return json.loads(raw)


def _norm(value):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold()


def _anchors_match(anchors, pages):
    """逐页重验当时留存的原文位置；不能只信任核验通过布尔值。"""
    return bool(anchors) and all(
        1 <= anchor["page"] <= len(pages) and anchor["offset"] >= 0
        and _norm(pages[anchor["page"] - 1])[anchor["offset"]:].startswith(anchor["literal"])
        and bool(anchor["literal"])
        for anchor in anchors
    )


def reviewed_publication(item, document, checked_at):
    """仅接受仍匹配当前目录、原件摘要、来源时间及正文定位的已验收证据。

    没有专项证据返回 None，继续普通核验；存在但被修改或版本不一致则明确拒绝。
    路径按数字公告号构造，不跟随证据中的绝对路径，防止串用其他文件。
    """
    announcement_id = str(item["announcementId"])
    if not re.fullmatch(r"\d+", announcement_id):
        return None
    audit_path = REVIEW_ROOT / "current-audit-v2" / (announcement_id + ".json")
    if not audit_path.exists():
        return None
    audit = _read(audit_path)
    source = _read(REVIEW_ROOT / "current" / (announcement_id + ".json"), audit["source_sha256"])
    rows = audit["catalog_rows"]
    valid = (
        len(rows) == 1 and all(rows[0].get(key) == item.get(key) for key in _BINDING)
        and audit["id"] == source["id"] == announcement_id
        and audit["stock_code"] == source["stock_code"] == item["secCode"]
        and audit["raw_sha256"] == source["receipt"]["sha256"] == document["receipt"]["sha256"]
        and audit["source_announced_at"] == source["published_at"] == item["announced_at_source"]
        and audit["title"] == source["title"] == item["title_plain"]
        and source["receipt"]["url"] == document["receipt"]["url"]
        and audit["source_publication_verified"] is True
        and audit["current_document_nature_verified"] is True
        and not audit["current_publication_gaps"] and not audit["version_issues"]
        and audit["historical_training_eligible"] is False
        and audit["numeric_table_semantics_verified"] is False
        and audit["document_kind"] != "UNCLASSIFIED"
    )
    reviewed_at = datetime.fromisoformat(audit["at"])
    valid = valid and reviewed_at.tzinfo is not None and reviewed_at <= checked_at
    pages = source["normalized_pages"]
    valid = valid and _anchors_match(audit["identity_anchors"], pages) and _anchors_match(audit["kind_anchors"], pages)
    valid = valid and any(len(_norm(page)) >= 100 for page in pages)
    # 收购标的审计正文通过关联文书证明与上市公司的关系，不能把标的身份当上市公司身份。
    related = audit.get("related_entity_proof")
    if related:
        if not re.fullmatch(r"\d+", str(related["id"])):
            raise ValueError("PUBLICATION_REVIEW_RELATED_ID_INVALID")
        proof = _read(REVIEW_ROOT / "current" / (str(related["id"]) + ".json"))
        valid = valid and proof["receipt"]["sha256"] == related["sha256"]
        valid = valid and proof["stock_code"] == item["secCode"]
        valid = valid and _anchors_match(related["anchors"], proof["normalized_pages"])
        verified_bytes(proof["receipt"])
    if not valid:
        raise ValueError("PUBLICATION_REVIEW_VERSION_OR_IDENTITY_MISMATCH")
    verified_bytes(source["receipt"])
    return {
        "rule": "EXACT_ORIGINAL_PUBLICATION_REVIEW_V1",
        "review_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "source_sha256": audit["source_sha256"], "raw_sha256": audit["raw_sha256"],
        "document_kind": audit["document_kind"], "business_boundary": audit["business_boundary"],
        "identity_anchors": audit["identity_anchors"], "kind_anchors": audit["kind_anchors"],
        "reviewed_at": audit["at"], "numeric_table_semantics_verified": False,
        "historical_training_eligible": False,
    }
