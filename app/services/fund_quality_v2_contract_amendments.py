"""基金合同具名修订公告的原件核验；不将修订日等同完整生效版本链。"""

import hashlib
import re
import unicodedata
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from app.services.fund_exposure_common import read

VERSION = "QUALITY_CONTRACT_AMENDMENT_EVIDENCE_20261017_1"
FEE_TITLE = "招商基金管理有限公司关于调低旗下部分基金费率并修订基金合同的公告"


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def cmf_fee_amendment_html(raw: bytes, *, legal_name: str, published_on: str) -> dict:
    """读取官网实际标题、公布日、正文生效条文和具名附件行，不依赖目录标志。

    只适配该类费率/主体资料修订公告。两种费率及修改范围保留原样，不能
    因公告称其余要素不变而证明之后从未再修订合同或基金运作始终合规。
    """
    if len(raw) > 4_000_000:
        raise ValueError("CONTRACT_NOTICE_HTML_SIZE_LIMIT")
    soup = BeautifulSoup(raw, "html.parser")
    heads = soup.select(".article_detail_title .wrapfix")
    bodies = soup.select(".article_detail_box")
    if len(heads) != 1 or len(bodies) != 1:
        raise ValueError("CONTRACT_NOTICE_UNIQUE_ISSUER_ARTICLE_BODY_REQUIRED")
    titles, dates = heads[0].select("h2"), heads[0].select(".data")
    if (len(titles) != 1 or _compact(titles[0].get_text()) != FEE_TITLE or len(dates) != 1
            or dates[0].get_text(strip=True) != published_on):
        raise ValueError("CONTRACT_NOTICE_ISSUER_TITLE_OR_PUBLICATION_UNBOUND")
    body = _compact(bodies[0].get_text("\n", strip=True))
    effective = list(re.finditer(
        r"决定自(20\d{2})年(\d{1,2})月(\d{1,2})日起[,，]调低旗下部分基金的管理费率及托管费率"
        r"并对基金合同有关条款进行修订", body))
    if len(effective) != 1:
        raise ValueError("CONTRACT_NOTICE_EFFECTIVE_REVISION_SENTENCE_UNPARSED")
    day = date(*map(int, effective[0].groups()))
    if day < date.fromisoformat(published_on):
        raise ValueError("CONTRACT_NOTICE_REVISION_PRECEDES_PUBLICATION")
    unchanged = "除上述要素外,其余要素不作变更"
    if unchanged not in body or "同时对相关基金的基金管理人、基金托管人信息进行必要更新" not in body:
        raise ValueError("CONTRACT_NOTICE_CHANGE_SCOPE_UNPARSED")
    rows = []
    for row in bodies[0].select("tr"):
        cells = [_compact(cell.get_text()) for cell in row.find_all(["td", "th"], recursive=False)]
        if len(cells) == 6 and cells[1] == legal_name and re.fullmatch(r"\d+", cells[0]):
            if all(re.fullmatch(r"\d+(?:\.\d+)?%", cell) for cell in cells[2:]):
                rows.append(cells)
    if len(rows) != 1:
        raise ValueError("CONTRACT_NOTICE_NAMED_FUND_ROW_NOT_UNIQUE")
    return {"version": VERSION, "title": FEE_TITLE, "published_on": published_on, "effective_on": str(day),
            "legal_name": legal_name, "named_fund_fee_row": rows[0], "effective_excerpt": effective[0][0],
            "unchanged_except_excerpt": unchanged, "event_kind": "FEE_AND_ORGANIZATION_INFORMATION_REVISION",
            "scope": "NAMED_REVISION_EFFECTIVE_DATE_NOT_COMPLETE_LATER_VERSION_CHAIN_OR_COMPLIANCE"}


def verify_cmf_fee_amendment(raw: bytes, ref: dict, *, refs: list[dict], root: Path,
                             legal_name: str, as_of: date) -> dict:
    """核原页与实际下载收据原字节、URL/日期/大小/时刻/摘要的一一绑定。"""
    address = urlsplit(ref["url"])
    if (address.scheme != "https" or address.hostname != "www.cmfchina.com" or address.username
            or address.password or address.port not in {None, 443} or address.query or address.fragment
            or not re.fullmatch(r"/web/fundnoticedetail/\d+/index\.html", address.path)
            or ref.get("label") != "contract-amendment-notice"):
        raise ValueError("CONTRACT_NOTICE_ISSUER_ENDPOINT_OR_ROLE_UNSUPPORTED")
    sha = hashlib.sha256(raw).hexdigest()
    if sha != ref["sha256"] or date.fromisoformat(ref["published_on"]) > as_of:
        raise ValueError("CONTRACT_NOTICE_ACTUAL_HASH_OR_PUBLICATION")
    proofs = [p for p in refs if p.get("sha256") == ref.get("contract_notice_receipt_sha256")
              and p.get("label") == "contract-notice-download-receipt"]
    if len(proofs) != 1:
        raise ValueError("CONTRACT_NOTICE_DOWNLOAD_RECEIPT_ORIGINAL_REQUIRED")
    proof = proofs[0]
    file = (root/proof["file"]).resolve()
    if not file.is_relative_to((root/"sources").resolve()) or not file.is_file() or file.stat().st_size > 100_000:
        raise ValueError("CONTRACT_NOTICE_RECEIPT_PATH_OR_SIZE")
    if hashlib.sha256(file.read_bytes()).hexdigest() != proof["sha256"]:
        raise ValueError("CONTRACT_NOTICE_RECEIPT_ACTUAL_HASH")
    receipt = read(file)
    if not isinstance(receipt, dict) or not isinstance(receipt.get("received_at"), str):
        raise ValueError("CONTRACT_NOTICE_RECEIPT_STRUCTURE_OR_TIME")
    if (type(receipt.get("status")) is not int or type(receipt.get("bytes")) is not int
            or receipt.get("status") != 200 or receipt.get("url") != ref["url"] or receipt.get("sha256") != sha
            or receipt.get("bytes") != len(raw) or receipt.get("received_at") != ref.get("acquired_at")
            or proof.get("acquired_at") != ref.get("acquired_at")):
        raise ValueError("CONTRACT_NOTICE_RECEIPT_CONTENT_NOT_BOUND")
    received = datetime.fromisoformat(receipt["received_at"])
    if received.tzinfo is None or received.date() < date.fromisoformat(ref["published_on"]):
        raise ValueError("CONTRACT_NOTICE_RECEIPT_TIME")
    facts = cmf_fee_amendment_html(raw, legal_name=legal_name, published_on=ref["published_on"])
    return facts | {"source_hashes": [sha, proof["sha256"]], "source_url": ref["url"],
                    "received_at": receipt["received_at"],
                    "history_scope": "RETROSPECTIVE_ORIGINAL_READ_NOT_PROOF_OF_UNREVISED_FIRST_HTTP_VERSION"}
