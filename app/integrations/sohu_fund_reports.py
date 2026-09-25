"""公开基金报告的第二转载来源；按基金、报告期、披露日期核对，绝不代替官方身份校验。"""

import hashlib
import re
import time
from datetime import datetime, timedelta
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup

from app.integrations.dbfund_reports import bounded_get
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import ROOT, blob, now, read
from app.services.fund_materials_store import versioned_save

BASE = "https://q.fund.sohu.com"


def page(client, code, url, *, body=False):
    """只读固定候选的正常公开页面，原文按内容留存，30 天复用；失败不覆盖成功文件。"""
    from app.integrations.public_fund_reports import PEERS, STORE

    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if (
        code not in PEERS
        or parsed.scheme != "https"
        or parsed.netloc != "q.fund.sohu.com"
        or parsed.path != ("/q/read.php" if body else "/q/notice.php")
        or query.get("code") != [code]
    ):
        raise ValueError("REPORT_REPRINT_SCOPE_INVALID")
    if body and not re.fullmatch(r"\d{1,10}", query.get("id", [""])[0]):
        raise ValueError("REPORT_REPRINT_ARTICLE_INVALID")
    path = STORE / "receipts" / (digest({"sohu": url}) + ".json")
    old = read(path) if path.exists() else None
    if old and now() - datetime.fromisoformat(old["checked_at"]) < timedelta(days=30):
        raw = (ROOT / old["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != old["sha256"]:
            raise ValueError("REPORT_CACHED_BODY_CHANGED")
        return raw, old
    if client.count >= 600:
        raise ValueError("REPORT_REQUEST_BUDGET")
    time.sleep(max(0, 0.4 - (time.monotonic() - client.last)))
    client.count += 1
    client.last = time.monotonic()
    raw = bounded_get(client.client, url, 8_000_000, headers={"Referer": BASE + "/"})
    soup = BeautifulSoup(raw, "html.parser")
    if code not in soup.get_text() or (body and not soup.select_one("#sohu_content")):
        raise ValueError("REPORT_REPRINT_EMPTY_OR_CHALLENGE")
    sha, filename = blob(raw, "html")
    receipt = {
        "url": url,
        "sha256": sha,
        "file": filename,
        "received_at": old["received_at"] if old and sha == old["sha256"] else now().isoformat(),
        "checked_at": now().isoformat(),
        "source": "SOHU_PUBLIC_ISSUER_REPORT_REPRINT",
        "format": "PUBLIC_REPRINT_TEXT",
        "historical_first_seen_verified": False,
        # 网页侧栏净值每天变化，正文修订由下面的内容摘要单独判断。
        "previous_sha256": old["sha256"] if old and sha != old["sha256"] else (old or {}).get("previous_sha256"),
    }
    versioned_save(path, receipt)
    return raw, receipt


def catalog(client, code):
    """沿页面提供的下一页链接翻页；不猜隐藏接口，不把提示公告当完整报告。"""
    from app.integrations.public_fund_reports import PEERS, report_period

    if not hasattr(client, "sohu_catalogs"):
        client.sohu_catalogs = {}
    if code in client.sohu_catalogs:
        return client.sohu_catalogs[code]
    url, seen, result = BASE + "/q/notice.php?code=" + code, set(), {}
    for _ in range(20):
        if url in seen:
            raise ValueError("REPORT_REPRINT_CATALOG_LOOP")
        seen.add(url)
        raw, receipt = page(client, code, url)
        soup = BeautifulSoup(raw, "html.parser")
        for link in soup.select('a[href*="read.php"]'):
            title = link.get_text(" ", strip=True)
            period = report_period(title)
            if not period or PEERS[code] not in re.sub(r"\s+", "", title):
                continue
            date_node = link.parent.select_one(".date")
            match = re.search(r"20\d{2}-\d{2}-\d{2}", date_node.text if date_node else "")
            if not match or not "2020-09-30" <= period[0] <= "2024-09-30":
                continue
            item = {
                "url": urljoin(BASE, link["href"]),
                "title": title,
                "published_date": match[0],
                "catalog_receipt": receipt,
            }
            previous = result.get(period)
            if previous and previous["url"] != item["url"]:
                raise ValueError("REPORT_REPRINT_PERIOD_AMBIGUOUS")
            result[period] = item
        next_link = next((a for a in soup.select('a[href*="notice.php"]') if "下一页" in a.text), None)
        if next_link is None:
            client.sohu_catalogs[code] = result
            return result
        url = urljoin(BASE, next_link["href"])
    raise ValueError("REPORT_REPRINT_CATALOG_LIMIT")


def report(client, code, entry):
    """目录匹配之外再次核正文标题、C 类代码和送出日期；页面日期参与可得时间。"""
    from app.integrations.public_fund_reports import PEERS, STORE, report_period

    item = catalog(client, code).get((entry["report_end"], entry["report_type"]))
    if not item:
        raise ValueError("REPORT_REPRINT_PERIOD_NOT_FOUND")
    raw, receipt = page(client, code, item["url"], body=True)
    soup = BeautifulSoup(raw, "html.parser")
    title = soup.select_one("h1")
    body = soup.select_one("#sohu_content")
    publication = soup.select_one(".article_info > .txt .c")
    title = title.get_text(" ", strip=True) if title else ""
    text = body.get_text("\n", strip=True) if body else ""
    published = re.search(r"20\d{2}-\d{2}-\d{2}", publication.text if publication else "")
    if (
        report_period(title) != (entry["report_end"], entry["report_type"])
        or PEERS[code] not in re.sub(r"\s+", "", title)
        or code not in text
        or not published
        or published[0] != item["published_date"]
    ):
        raise ValueError("REPORT_REPRINT_IDENTITY_OR_DATE_MISMATCH")
    content_hash = digest({"title": title, "text": text, "published": published[0]})
    path = STORE / "receipts" / (digest({"sohu_content": item["url"]}) + ".json")
    old = read(path) if path.exists() else None
    receipt = {
        **receipt,
        "content_hash": content_hash,
        "catalog_receipt": item["catalog_receipt"],
        "revised_after_receipt": bool(
            (old or {}).get("revised_after_receipt") or (old and old["content_hash"] != content_hash)
        ),
    }
    versioned_save(path, receipt)
    return {
        "data": {
            "art_code": entry["ID"],
            "security": [{"stock": code}],
            "notice_title": title,
            "notice_content": text,
            "notice_date": published[0],
            "attach_url": None,
        }
    }, receipt
