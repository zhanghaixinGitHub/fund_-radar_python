"""六份已定位的公开历史报告补查；只留存在本轮目录，不更新原资料或调用付费接口。"""

import hashlib
import io
import re
from datetime import date, datetime, timedelta
from urllib.parse import urljoin, urlparse

import httpx
from app.integrations.fund_report_sections_v2 import parse_text
from app.services.fund_002112_peer_material_repair import OUTPUT
from app.services.fund_002112_zero_fit_review import digest, file_hash, read_json, save_once
from bs4 import BeautifulSoup
from pypdf import PdfReader

# 全部 URL 来自本轮搜索或管理人公告页面；不猜下载标识，不遍历站点或抓取新闻。
TARGETS = [
    ("017493", "2022-12-31", "ANNUAL", "https://static.cninfo.com.cn/finalpage/2023-03-31/1216290698.PDF"),
    ("017493", "2023-06-30", "QUARTER", "https://static.cninfo.com.cn/finalpage/2023-07-21/1217355029.PDF"),
    ("017493", "2023-06-30", "HALF", "https://pdf.dfcfw.com/pdf/H2_AN202308311596889210_1.pdf"),
    ("160323", "2022-12-31", "ANNUAL", "https://www.chinaamc.com/c/2023-03-30/683748.shtml"),
    ("160323", "2023-06-30", "QUARTER", "https://www.chinaamc.com/c/2023-07-20/687684.shtml"),
    ("160323", "2023-09-30", "QUARTER", "https://accountquery.chinaamc.com/c/2023-10-24/677638.shtml"),
]
ALLOWED = {"static.cninfo.com.cn", "pdf.dfcfw.com", "www.chinaamc.com", "accountquery.chinaamc.com"}


def validate_pdf_identity(pages, code):
    """年报目录可能占四页，按主代码后的产品信息核份额，不硬截为前四页。"""
    if code not in {"017493", "160323"}:
        raise ValueError("PDF_FUND_SCOPE_INVALID")
    compact = re.sub(r"\s+", "", "\n".join(pages[:12]))
    name, master = ("东方红新动力", "000480") if code == "017493" else ("华夏磐泰", "160323")
    location = re.search(r"基金主代码[:：]?" + master, compact)
    if name not in re.sub(r"\s+", "", "\n".join(pages[:3])) or not location:
        raise ValueError("PDF_FUND_IDENTITY_MISMATCH")
    if code not in compact[location.start() : location.end() + 3500]:
        raise ValueError("PDF_SHARE_CLASS_NOT_IN_PRODUCT_SECTION")


def run():
    output = OUTPUT / "public-pdf-probe"
    output.mkdir(exist_ok=True)
    manifest_path = output / "results.json"
    if manifest_path.exists():
        result = read_json(manifest_path)
        for source in result["receipts"]:
            if file_hash(source["path"]) != source["sha256"]:
                raise ValueError("PUBLIC_SOURCE_RECEIPT_CHANGED")
        reviewed_path = output / "reviewed-results.json"
        if reviewed_path.exists():
            # 原首次尝试账本不覆盖；补充本地复核结论，避免把前四页的旧限制继续当作资料缺失。
            result = {**result, "local_review": read_json(reviewed_path)}
        return result
    save_once(
        output / "plan.json",
        {
            "targets": TARGETS,
            "maximum_documents": 6,
            "maximum_requests": 12,
            "maximum_bytes_per_response": 15000000,
            "new_fits": 0,
            "scope": "Historical full reports only; current receipt is not historical first-seen",
        },
    )
    receipts, findings = [], []
    requests = 0

    def get(client, url):
        nonlocal requests
        parsed_url = urlparse(url)
        if parsed_url.scheme != "https" or parsed_url.hostname not in ALLOWED or requests >= 12:
            raise ValueError("PUBLIC_SOURCE_SCOPE_OR_REQUEST_LIMIT")
        requests += 1
        with client.stream("GET", url) as response:
            response.raise_for_status()
            # 不自动跟随登录、挑战页或跳转到未检查来源。
            if response.status_code != 200:
                raise ValueError("UNEXPECTED_SOURCE_STATUS")
            data = bytearray()
            for part in response.iter_bytes():
                data.extend(part)
                if len(data) > 15000000:
                    raise ValueError("PUBLIC_DOCUMENT_TOO_LARGE")
            raw = bytes(data)
        sha = hashlib.sha256(raw).hexdigest()
        path = output / (sha + (".pdf" if raw.startswith(b"%PDF") else ".html"))
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(raw)
        elif file_hash(path) != sha:
            raise ValueError("PUBLIC_RAW_COLLISION")
        receipt = {
            "url": url,
            "path": str(path),
            "sha256": sha,
            "received_at": datetime.now().astimezone().isoformat(),
            "historical_first_seen_verified": False,
        }
        receipts.append(receipt)
        return raw, receipt

    sources = read_json(OUTPUT / "protocol.json")["sources"]
    with httpx.Client(timeout=httpx.Timeout(30, connect=5), follow_redirects=False) as client:
        for code, end, kind, url in TARGETS:
            item = {"fund_code": code, "report_end": end, "report_type": kind, "discovery_url": url}
            try:
                raw, receipt = get(client, url)
                if not raw.startswith(b"%PDF"):
                    if not url.endswith(".shtml"):
                        raise ValueError("PDF_SOURCE_RETURNED_NON_PDF")
                    soup = BeautifulSoup(raw, "html.parser")
                    links = [
                        a for a in soup.select("a[href]") if "磐泰" in a.get_text() and ".pdf" in a.get_text().lower()
                    ]
                    if len(links) != 1:
                        raise ValueError("OFFICIAL_ATTACHMENT_NOT_UNIQUE")
                    raw, receipt = get(client, urljoin(url, links[0]["href"]))
                if not raw.startswith(b"%PDF"):
                    raise ValueError("RESPONSE_NOT_PDF")
                reader = PdfReader(io.BytesIO(raw))
                if reader.is_encrypted or not 1 <= len(reader.pages) <= 180:
                    raise ValueError("PDF_ENCRYPTED_OR_PAGE_LIMIT")
                pages = [p.extract_text() or "" for p in reader.pages]
                entry = next(
                    e
                    for e in read_json(sources["manifest_" + code]["path"])["catalog"]
                    if e["report_end"] == end and e["report_type"] == kind
                )
                name = "东方红新动力" if code == "017493" else "华夏磐泰"
                validate_pdf_identity(pages, code)
                item.update(raw_receipt=receipt, pages=len(pages), text_sha256=digest(pages))
                parsed = parse_text(
                    pages,
                    entry["TITLE"],
                    fund_code=code,
                    fund_name=name,
                    master_code="000480" if code == "017493" else code,
                )
                if (parsed["report_end"], parsed["report_type"]) != (end, kind):
                    raise ValueError("PDF_PERIOD_MISMATCH")
                public_day = max(entry["published_date"], parsed["published_date"])
                parsed.update(
                    raw=receipt,
                    source=entry,
                    source_publication_date=public_day,
                    available_at=str(date.fromisoformat(public_day) + timedelta(days=1)) + "T08:00:00+08:00",
                    document_format="PUBLIC_ORIGINAL_PDF",
                    original_pdf_saved=True,
                    training_eligible=False,
                    historical_version_verified=False,
                    purpose="NEW_PUBLIC_SOURCE_PENDING_COMPLETENESS_AND_VERSION_REVIEW",
                )
                filename = f"{code}-{end}-{kind}-candidate.json"
                save_once(output / filename, parsed)
                item.update(
                    status="PARSED_CANDIDATE_NOT_ADOPTED",
                    holding_count=parsed["holding_count"],
                    parsed_file=str(output / filename),
                )
            except (httpx.HTTPError, ValueError) as exc:
                item.update(status="UNRESOLVED", reason=str(exc) if isinstance(exc, ValueError) else type(exc).__name__)
            findings.append(item)
            print(code, end, kind, item["status"], item.get("reason", ""), flush=True)
    result = {"requests": requests, "findings": findings, "receipts": receipts, "new_fits": 0}
    save_once(manifest_path, result)
    return result


if __name__ == "__main__":
    run()
