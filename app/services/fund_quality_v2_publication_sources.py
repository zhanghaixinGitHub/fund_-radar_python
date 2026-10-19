"""从保存的发行人目录和正式公告索引复核招募书公开日期。

证据只用于事后核定历史文件的公开日期，不把索引中的后来财务或事项
用作更早考察日的计分输入，也不声称已遍历发行人的全部目录。
"""

import hashlib
import json
import re
import unicodedata
from datetime import date
from pathlib import Path
from urllib.parse import urljoin, urlsplit


def publication_proof_refs(refs: list[dict]) -> list[dict]:
    """展开有界的原始证明文件；来源角色由本模块确定，不接受核验布尔标志。

    原经济报告已经登记时保留其原角色；相同摘要不重复统计成新原件。
    证明文件仍由材料入口检查路径、字节摘要、取得/观测时刻和文件大小。
    """
    result = {}
    original_hashes = {ref["sha256"] for ref in refs}
    for ref in refs:
        if "publication_evidence" not in ref:
            continue
        evidence = ref["publication_evidence"]
        if not isinstance(evidence, dict) or set(evidence) != {"catalog", "notice_index", "article_id"}:
            raise ValueError("PUBLICATION_EVIDENCE_SCHEMA")
        if type(evidence["article_id"]) is not int or evidence["article_id"] <= 0:
            raise ValueError("PUBLICATION_EVIDENCE_ARTICLE_ID")
        for name in ("catalog", "notice_index"):
            proof = evidence[name]
            if not isinstance(proof, dict) or not all(
                isinstance(proof.get(key), str) and proof[key] for key in ("sha256", "file", "url", "acquired_at")
            ):
                raise ValueError("PUBLICATION_EVIDENCE_REFERENCE")
            if proof["sha256"] not in original_hashes:
                result[proof["sha256"]] = {**proof, "label": "publisher-" + name.replace("_", "-")}
    return list(result.values())


def _read_proof(ref: dict, root: Path) -> bytes:
    path = (root / ref["file"]).resolve()
    if not path.is_relative_to(root.resolve() / "sources") or not path.is_file() or path.stat().st_size > 20_000_000:
        raise ValueError("PUBLICATION_EVIDENCE_PATH_SIZE")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("PUBLICATION_EVIDENCE_HASH")
    return raw


def _clean(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold()


def verify_prospectus_publication(ref: dict, root: Path, code: str, as_of: date) -> dict:
    """重读目录实际条目及同产品法定报告的公告索引，返回日期与原文位置证据。

    当前仅实现华宝公开目录格式；其他发行人不可借同域名或自填成功标志
    越过格式/来源核验。基金身份另由招募书与含份额代码的报告交叉核对。
    """
    from app.services.fund_rating_public_market import public_document

    evidence = ref.get("publication_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("PROSPECTUS_PUBLICATION_EVIDENCE_REQUIRED")
    publication_proof_refs([ref])
    origin = urlsplit(ref["url"])
    notice = evidence["notice_index"]
    notice_origin = urlsplit(notice["url"])
    if any(
        source.scheme != "https" or source.hostname != "www.fsfund.com" or source.port not in (None, 443)
        or source.username or source.password
        or source.query or source.fragment
        for source in (origin, notice_origin)
    ):
        raise ValueError("PROSPECTUS_PUBLICATION_ISSUER_UNSUPPORTED")
    catalog_raw = _read_proof(evidence["catalog"], root)
    catalog = json.loads(catalog_raw)
    if (
        not isinstance(catalog, dict) or catalog.get("code") != "0000"
        or not isinstance(catalog.get("data"), list) or len(catalog["data"]) > 100
        or any(not isinstance(row, dict) for row in catalog["data"])
    ):
        raise ValueError("PROSPECTUS_PUBLICATION_CATALOG_SCHEMA")
    rows = [row for row in catalog["data"] if row.get("id") == evidence["article_id"]]
    if len(rows) != 1:
        raise ValueError("PROSPECTUS_PUBLICATION_CATALOG_IDENTITY")
    row = rows[0]
    published = str(row.get("publishDate", ""))
    if not re.fullmatch(r"\d{8}", published):
        raise ValueError("PROSPECTUS_PUBLICATION_DATE")
    actual_date = date(int(published[:4]), int(published[4:6]), int(published[6:]))
    title = ref.get("title", "")
    if (
        str(actual_date) != ref.get("published_on") or actual_date > as_of
        or _clean(row.get("title", "")) != _clean(title) or "招募说明书" not in title
        or urljoin("https://www.fsfund.com/", row.get("articleFile", "")) != ref["url"]
    ):
        raise ValueError("PROSPECTUS_PUBLICATION_ROW_CONFLICT")
    # 已观察到更晚的适用版本时不得悄悄退回旧书；未来公开版本不提前使用。
    newer = [r for r in catalog["data"] if _clean(r.get("title", "")) == _clean(title)
             and published < str(r.get("publishDate", "")) <= as_of.strftime("%Y%m%d")]
    if newer:
        raise ValueError("PROSPECTUS_NEWER_LISTED_VERSION_AVAILABLE")
    raw = _read_proof(notice, root)
    if not raw.startswith(b"%PDF"):
        raise ValueError("PROSPECTUS_PUBLICATION_NOTICE_PDF_REQUIRED")
    pages, _ = public_document(raw, notice, code, "年度报告")
    body = _clean("\n".join(pages))
    product_title = _clean(title).split("招募说明书", 1)[0]
    if product_title not in body[:3000] or "年度报告" not in body[:3000] or code not in body[:16000]:
        raise ValueError("PROSPECTUS_PUBLICATION_NOTICE_PRODUCT")
    qualified = []
    for match in re.finditer(re.escape(_clean(title)), body):
        # 必须是本条标题紧邻的渠道和第一个日期，不能跨到下一条公告挑日期。
        suffix = body[match.end():match.end() + 90]
        item_date = re.match(r"基金管理人网站(?:[,，、][\u4e00-\u9fff]+){0,5}(20\d{2}-\d{2}-\d{2})", suffix)
        if item_date and item_date[1] == str(actual_date):
            qualified.append(body[max(0, match.start() - 15):match.end() + item_date.end()])
    if not qualified:
        raise ValueError("PROSPECTUS_PUBLICATION_NOTICE_ROW_MISSING")
    return {
        "published_on": str(actual_date), "catalog_article_id": evidence["article_id"],
        "source_hashes": [evidence["catalog"]["sha256"], notice["sha256"]],
        "notice_excerpt": qualified[-1],
        "scope": "RETROSPECTIVE_PUBLICATION_METADATA_ONLY_NOT_ECONOMIC_INPUT_OR_COMPLETE_DIRECTORY_PROOF",
    }
