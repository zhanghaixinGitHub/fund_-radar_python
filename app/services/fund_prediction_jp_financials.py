"""日本披露证券的公开财报续查；只取发行人原文，不使用聚合页的自动摘要。"""

import hashlib
import re
import unicodedata
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import now, read
from app.services.fund_materials_store import versioned_save
from app.services.material_content import content_digest, document_pdfium, exact_source_text, source_bytes, write_source

PDF_HOST = "finance-frontend-pc-dist.west.edge.storage-yahoo.jp"


def reporting_identity(body: str, security: str, published: str):
    """核发行人原文代码和实际公布日，显式日期区间才作为报告期；不猜财年。

    日本财年标签可属于下一公历年。只有正文明确起止日、且期末不晚于公开日
    才返回报告期；缺失时保留全文交进一步解析，不把供应商摘要当原件。
    """
    if not re.fullmatch(r"[0-9A-Z]{4}\.JP", security):
        raise ValueError("JP_FINANCIAL_SECURITY_INVALID")
    plain = unicodedata.normalize("NFKC", body)
    if not re.search(r"コ\s*ー\s*ド\s*番\s*号\s*[:：]?\s*" + re.escape(security[:-3]) + r"(?![0-9A-Z])", plain[:6000]):
        raise ValueError("JP_FINANCIAL_ORIGINAL_IDENTITY")
    dates = list(re.finditer(r"(20\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日", plain))
    if not dates or str(date(*map(int, dates[0].groups()))) != published:
        raise ValueError("JP_FINANCIAL_PUBLICATION_DATE")
    periods = []
    for left, right in zip(dates, dates[1:], strict=False):
        between = plain[left.end():right.start()]
        # 同一张已核证券财报的「自…至…」财务报表栏可补偿封面叠印字形；
        # 不删重复字符或由财年名称猜期末，仍只接受正文明确的两个日期。
        if not re.fullmatch(r"\s*[～~〜－–—至-]\s*", between):
            continue
        start, end = (date(*map(int, m.groups())) for m in (left, right))
        if start <= end <= date.fromisoformat(published) and (end - start).days <= 370:
            periods.append((start, end))
    return max((str(end) for _, end in periods), default=None)


def financial_link(html: str, security: str, as_of: str, *, disclosure=False):
    """只接本证券页面明确引用的原财报PDF，不采摘要、付费报告或外部跳转。"""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text() if soup.title else ""
    if security[:-3] not in title:
        raise ValueError("JP_FINANCIAL_PAGE_IDENTITY")
    links = []
    for anchor in soup.find_all("a", href=True):
        label, url = anchor.get_text(" ", strip=True), anchor["href"]
        parsed = urlparse(url)
        path = re.fullmatch(r"/disclosure/(20\d{6})/\d+\.pdf", parsed.path)
        if "決算短信" not in label or parsed.scheme != "https" or parsed.hostname != PDF_HOST or not path:
            continue
        day = date(int(path[1][:4]), int(path[1][4:6]), int(path[1][6:])).isoformat()
        if day <= as_of:
            links.append({"url": url, "title": label, "published_date": day})
    links = list({row["url"]: row for row in links}.values())
    if disclosure and links:
        latest = max(row["published_date"] for row in links)
        links = [row for row in links if row["published_date"] == latest]
    if len(links) != 1:
        raise ValueError("JP_FINANCIAL_ORIGINAL_LINK_REQUIRED")
    return links[0]


def collect(security: str, output: Path, *, documents=(), client=None, retry_local_failure=False):
    """每天同一期有界续查一次；相同原件复用原取得时间，保留新增全文及第一失败。

    返回值是采集与解析成果，绝不是综合用途批准或方向。目录页只负责发现原文，
    新PDF仍须本聊天阅读；原文身份或日期冲突不自动覆盖既有公司资料。
    """
    if not re.fullmatch(r"[0-9A-Z]{4}\.JP", security):
        raise ValueError("JP_FINANCIAL_SECURITY_INVALID")
    output = Path(output)
    receipt = output / "latest.json"
    previous = {}
    if receipt.exists():
        previous = read(receipt)
        # 正常目录失败后的官网原件，允许通用续办消费本基金同证券的补查成果。
        # 原始失败和实际取得时间不改写；补查文件存在不构成用途批准。
        primary = output.parent / "primary-ir" / "financials.json"
        if previous.get("status") == "ACTION_REQUIRED" and primary.exists():
            fallback = read(primary)
            originals = fallback.get("originals", [])
            if fallback.get("security_code") != security:
                raise ValueError("JP_PRIMARY_FALLBACK_SECURITY_CONFLICT")
            if originals:
                selected = max(originals, key=lambda r: (r["published_date"], r["report_end"]))
                if (selected["security_code"] != security
                        or content_digest(source_bytes(Path(selected["raw_path"]))) != selected["raw_sha256"]
                        or selected["published_date"] > str(now().date())
                        or reporting_identity(selected["body"], security, selected["published_date"])
                        != selected["report_end"]):
                    raise ValueError("JP_PRIMARY_FALLBACK_ORIGINAL_CONFLICT")
                value = {**selected, "status": "ORIGINAL_PARSED_READING_REQUIRED",
                         "checked_at": now().isoformat(), "first_failure": previous.get("failure"),
                         "source_attempts": previous.get("source_attempts", []) + fallback.get("attempts", []),
                         "recovered_from": "ISSUER_PUBLIC_ORIGINAL"}
                versioned_save(receipt, value)
                return value
        # 仅明确的本机写入/解析修复可再次执行；第一次失败由版本化回执保留。
        if not (retry_local_failure and previous.get("status") in {
            "ACTION_REQUIRED", "ORIGINAL_PARSED_PERIOD_REVIEW_REQUIRED",
        }):
            return previous
    own = client is None
    client = client or httpx.Client(timeout=httpx.Timeout(25, connect=5), follow_redirects=True, trust_env=False)
    value = {"security_code": security, "checked_at": now().isoformat(), "source_attempts": []}

    def acquire(url, label):
        for archived in previous.get("source_attempts", []):
            if archived["url"] != url or archived["http_status"] != 200:
                continue
            stored = source_bytes(Path(archived["raw_path"]))
            if content_digest(stored) != archived["sha256"]:
                raise ValueError("JP_FINANCIAL_CACHED_SOURCE_CHANGED")
            value["source_attempts"].append(archived)
            # HTML定位需要原始链接表示；PDF则直接消费已经持久化的全文页。
            return (exact_source_text(stored).encode("utf-8") if label == "原件引用目录" else stored), archived
        response = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = response.content
        if len(raw) > 15_000_000:
            raise ValueError("JP_FINANCIAL_SOURCE_SIZE_BOUND")
        sha = hashlib.sha256(raw).hexdigest()
        suffix = ".pdf" if raw.startswith(b"%PDF") else ".html"
        path = output / "sources" / (sha + suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            write_source(path, raw)
        source = {"url": url, "final_url": str(response.url), "http_status": response.status_code,
                  "received_at": now().isoformat(), "raw_path": str(path.resolve()), "sha256": sha, "label": label}
        value["source_attempts"].append(source)
        response.raise_for_status()
        if urlparse(str(response.url)).hostname != urlparse(url).hostname:
            raise ValueError("JP_FINANCIAL_SOURCE_REDIRECT")
        return raw, source

    try:
        base_url = "https://finance.yahoo.co.jp/quote/" + security[:-3] + ".T/"
        prior_server_error = any(
            s["url"] == base_url + "financials" and s["http_status"] >= 500
            for s in previous.get("source_attempts", [])
        )
        try:
            if prior_server_error:
                raise ValueError("JP_USE_DISCLOSURE_AFTER_RECORDED_SERVER_FAILURE")
            page, _ = acquire(base_url + "financials", "原件引用目录")
            link = financial_link(page.decode("utf-8"), security, str(now().date()))
        except (httpx.HTTPError, ValueError):
            # 正常决算页失败只再查本证券适时披露目录一次；不无限重复请求，
            # 不以自动摘要代原件，同日多版本仍交人工而不猜适用版本。
            page, _ = acquire(base_url + "disclosure", "原件引用目录")
            link = financial_link(page.decode("utf-8"), security, str(now().date()), disclosure=True)
        value["link"] = link
        existing = next((d for d in documents if d.get("url") == link["url"]), None)
        if existing:
            if (
                content_digest(source_bytes(Path(existing["raw_path"]))) != existing["raw_sha256"]
                or digest(existing["body"]) != existing["body_hash"]
            ):
                raise ValueError("JP_FINANCIAL_EXISTING_ORIGINAL_CHANGED")
            value.update(status="UNCHANGED_ORIGINAL_REUSED", existing_document_id=existing["id"], link=link,
                         original_available_at=existing["available_at"])
        else:
            raw, source = acquire(link["url"], "发行人原财报")
            if not (raw.startswith(b"%PDF") or getattr(raw, "kind", None) == "PDF"):
                raise ValueError("JP_FINANCIAL_NOT_PDF")
            # 沿项目资料库已有PDFium页文本，兼容AES原件；无需安装加密依赖，
            # 不绕过库内原文或重新下载已取得PDF，也不改历史正文。
            from contextlib import closing

            pages = []
            with document_pdfium(Path(source["raw_path"])) as document:
                for page in document:
                    with closing(page), closing(page.get_textpage()) as text:
                        pages.append(text.get_text_range())
            body = "\n".join(pages)
            end = reporting_identity(body, security, link["published_date"])
            value.update(status="ORIGINAL_PARSED_READING_REQUIRED" if end else "ORIGINAL_PARSED_PERIOD_REVIEW_REQUIRED",
                         body=body, body_scope="FULL_TEXT", title=link["title"], published_date=link["published_date"],
                         report_end=end, url=link["url"], received_at=source["received_at"],
                         raw_path=source["raw_path"], raw_sha256=source["sha256"])
    except Exception as error:
        value.update(status="ACTION_REQUIRED", failure=type(error).__name__ + ":" + str(error)[:300])
    finally:
        if own:
            client.close()
    versioned_save(receipt, value)
    return value
