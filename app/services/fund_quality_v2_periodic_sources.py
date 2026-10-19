"""V2定期披露的逐件准入与明确来源选择；不扩展第一版来源或证明全文等价。"""

import hashlib
import json
import re
import unicodedata
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlsplit

VERSION = "QUALITY_PERIODIC_SOURCE_SELECTION_20261017_1"
AUTHORITY = {"ISSUER_ORIGINAL": 0, "REGULATORY_DISCLOSURE_PDF": 1,
             "BOUND_PROVIDER_PDF": 2, "FULL_PUBLIC_REPRINT": 3}


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def verify_cninfo_periodic(raw: bytes, pages: list[str], ref: dict, *, refs: list[dict],
                           root: Path, code: str, as_of: date) -> dict:
    """重读下载收据和PDF主体，绑定日期、基金名称、主代码与所属份额。

    本函数不信任传入的verified标志。只接纳已保存的实际HTTP200收据与
    对应原字节；“送出日=网址日期=登记公布日”只证明此次取得的公开原件
    口径一致，不证明当年HTTP首发内容从未修订。目录代理或失败HTML均拒收。
    """
    address = urlsplit(ref.get("url", ""))
    route = re.fullmatch(r"/finalpage/(20\d{2}-\d{2}-\d{2})/(\d{8,14})\.PDF", address.path)
    if (address.scheme != "https" or address.hostname != "static.cninfo.com.cn" or not route
            or address.port not in {None, 443} or address.query or address.fragment
            or address.username or address.password
            or ref.get("label") not in {"historical-annual", "historical-half", "annual", "half"}):
        raise ValueError("QUALITY_CNINFO_PERIODIC_ROUTE_OR_ROLE")
    if not raw.startswith(b"%PDF") or hashlib.sha256(raw).hexdigest() != ref.get("sha256"):
        raise ValueError("QUALITY_CNINFO_ACTUAL_PDF_HASH")
    proofs = [r for r in refs if r.get("label") == "cninfo-download-receipt"
              and r.get("sha256") == ref.get("cninfo_receipt_sha256")]
    if len(proofs) != 1:
        raise ValueError("QUALITY_CNINFO_HTTP_RECEIPT_ORIGINAL_REQUIRED")
    proof_ref = proofs[0]
    proof_path = (root/proof_ref["file"]).resolve()
    if (not proof_path.is_relative_to(root.resolve()/"sources") or not proof_path.is_file()
            or proof_path.stat().st_size > 100_000):
        raise ValueError("QUALITY_CNINFO_HTTP_RECEIPT_PATH_OR_SIZE")
    proof_raw = proof_path.read_bytes()
    if hashlib.sha256(proof_raw).hexdigest() != proof_ref["sha256"]:
        raise ValueError("QUALITY_CNINFO_HTTP_RECEIPT_HASH")
    # 收据沿用项目带摘要的版本化原始文件，原封不动归档，不另造下载回执。
    from app.services.fund_rating_inputs import digest

    stored = json.loads(proof_raw)
    if (not isinstance(stored, dict) or set(stored) != {"hash", "payload"}
            or not isinstance(stored["payload"], dict) or digest(stored["payload"]) != stored["hash"]):
        raise ValueError("QUALITY_CNINFO_HTTP_RECEIPT_ENVELOPE")
    receipt = stored["payload"]
    if (receipt.get("status") != 200 or receipt.get("url") != ref["url"]
            or receipt.get("sha256") != ref["sha256"] or receipt.get("bytes") != len(raw)
            or receipt.get("received_at") != ref.get("acquired_at")
            or proof_ref.get("url") != ref["url"]
            or proof_ref.get("acquired_at") != receipt.get("received_at")
            or proof_ref.get("published_on") != ref.get("published_on")):
        raise ValueError("QUALITY_CNINFO_HTTP_RECEIPT_BINDING")
    received = datetime.fromisoformat(receipt["received_at"])
    if received.tzinfo is None:
        raise ValueError("QUALITY_CNINFO_HTTP_RECEIPT_TIMEZONE")
    from app.services.fund_rating_public_market import publication

    sent = publication(pages)
    if sent != route[1] or ref.get("published_on") != sent or date.fromisoformat(sent) > as_of:
        raise ValueError("QUALITY_CNINFO_PUBLICATION_ROUTE_COVER_OR_CUTOFF")
    if received.date() < date.fromisoformat(sent):
        raise ValueError("QUALITY_CNINFO_ACQUISITION_BEFORE_PUBLICATION")
    body, cover = _compact("\n".join(pages))[:18000], _compact("\n".join(pages[:2]))
    name = re.search(r"基金名称(.{6,100}?)基金简称", body)
    master = re.search(r"基金主代码([0-9]{6})(?!\d)", body)
    cover_identity = (re.match(r"^(?:\d{1,3})?" + re.escape(name[1])
                              + r"(?:20\d{2}年)?(?:年度|中期)报告", cover) if name else None)
    if not name or not master or not cover_identity or not re.fullmatch(r"[0-9]{6}", code):
        raise ValueError("QUALITY_CNINFO_COVER_AND_BASIC_PRODUCT_IDENTITY")
    # 份额只取基金基本情况中的明确交易代码行，不能从任意持仓/正文命中代码。
    basic = body[name.start():name.start()+2500]
    shares = re.search(r"下属分级基金的交易代码((?:[0-9]{6}){1,12})(?=报告期|基金|下属|注|$)", basic)
    share_codes = re.findall(r"[0-9]{6}", shares[1]) if shares else []
    if code != master[1] and code not in share_codes:
        raise ValueError("QUALITY_CNINFO_REQUESTED_SHARE_NOT_IN_BASIC_TABLE")
    title_kind = "年度报告" if "annual" in ref["label"] else "中期报告"
    if title_kind not in cover:
        raise ValueError("QUALITY_CNINFO_ORIGINAL_REPORT_KIND")
    return {"version": VERSION, "authority": "REGULATORY_DISCLOSURE_PDF", "announcement_id": route[2],
            "legal_fund_name": name[1], "master_code": master[1], "share_codes": share_codes,
            "fund_name_original": name[0], "master_code_original": master[0],
            "share_code_original": shares[0] if shares else None, "original_sent_on": sent,
            "source_hashes": [ref["sha256"], proof_ref["sha256"]],
            "publication_scope": "RETROSPECTIVE_PUBLIC_ORIGINAL_NOT_PROOF_OF_UNREVISED_HISTORICAL_HTTP_FIRST_COPY"}


def select_periodic_document(documents: list[dict]) -> tuple[dict | None, dict]:
    """先按实际经济期和公开日选组，再用明确来源优先级；同级不同原件拒收。

    PDF优先于转载是资料来源取舍，不声称全文相同。不同摘要只用于判断
    原字节是否相同，绝不比较摘要大小来决定更好的原件。冲突组不退回旧期。
    """
    if not documents:
        raise ValueError("QUALITY_PERIODIC_NO_VALID_DOCUMENTS")
    def dates(document):
        return (document.get("economic_period", ["", document["ref"].get("report_end", "")])[1],
                document["ref"].get("published_on", ""))

    latest = max(map(dates, documents))
    contemporaries = [d for d in documents if dates(d) == latest]
    if any(d.get("source_authority") not in AUTHORITY for d in contemporaries):
        raise ValueError("QUALITY_PERIODIC_SOURCE_AUTHORITY_UNCLASSIFIED")
    rank = min(AUTHORITY[d["source_authority"]] for d in contemporaries)
    preferred = [d for d in contemporaries if AUTHORITY[d["source_authority"]] == rank]
    hashes = {d["ref"]["sha256"] for d in preferred}
    evidence = {"version": VERSION, "economic_period_end": latest[0], "published_on": latest[1],
        "rule": "LATEST_PERIOD_THEN_PUBLICATION_THEN_EXPLICIT_SOURCE_PRIORITY_NO_DIGEST_RANKING",
        "counterparts": [{"sha256": d["ref"]["sha256"], "url": d["ref"]["url"],
                          "authority": d["source_authority"]} for d in contemporaries],
        "whole_document_equivalence_proven": False}
    if len(hashes) != 1:
        return None, {**evidence, "error": "QUALITY_PERIODIC_SAME_PRIORITY_DIFFERENT_ORIGINALS_CONFLICT",
                      "conflicting_sha256": sorted(hashes)}
    selected = preferred[0]
    return selected, {**evidence, "selected_sha256": selected["ref"]["sha256"],
                      "selected_authority": selected["source_authority"]}
