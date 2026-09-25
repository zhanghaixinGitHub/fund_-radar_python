"""德邦基金官网可公开访问的净值、公告及新闻，原始资料与研究输入分别留存。"""

import json
import re
import threading
import time
from datetime import date, datetime
from decimal import Decimal
from urllib.parse import urljoin, urlparse

import httpx
import pypdfium2 as pdfium
from bs4 import BeautifulSoup

from app.integrations.dbfund_reports import CATALOG, ORIGIN, PRODUCT, bounded_get
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import ROOT, blob, now, read, save
from app.services.fund_exposure_supplement import SUPPLEMENT, verified_bytes
from app.services.fund_materials_store import LIVE, source_path, versioned_save

NAV_URL = ORIGIN + "/common-web/chart/fundnettable/getFundNetTableJson"
NEWS_URL = ORIGIN + "/news/media/index.html"
PDF_LOCK = threading.Lock()


def client():
    return httpx.Client(
        timeout=httpx.Timeout(30, connect=5),
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": PRODUCT,
        },
    )


def fetch(c, url, suffix="json", params=None, *, reuse=True):
    """按实际访问地址保存证据；不跟随跳转到未知域名，不保存登录态或凭据。"""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "www.dbfund.com.cn":
        raise ValueError("EXPOSURE_PUBLIC_HOST_NOT_ALLOWED")
    request = {"url": url, "params": params}
    path = SUPPLEMENT / "public-receipts" / (digest(request) + ".json")
    if reuse and path.exists():
        receipt = read(path)
        return verified_bytes(receipt), receipt
    for attempt in range(2):
        try:
            raw = bounded_get(c, url, 20_000_000 if suffix == "pdf" else 4_000_000, params=params)
            break
        except (httpx.HTTPError, ValueError):
            if attempt:
                raise
            time.sleep(1)
    sha, filename = blob(raw, suffix)
    receipt = {
        **request,
        "sha256": sha,
        "file": filename,
        "received_at": now().isoformat(),
        "source_code": "DBFUND_OFFICIAL_PUBLIC",
        "historical_first_seen_verified": False,
    }
    version = SUPPLEMENT / "public-receipt-versions" / (digest(receipt) + ".json")
    save(version, receipt)
    save(path, receipt, replace=True)
    time.sleep(0.2)
    return raw, receipt


def parse_nav_page(value, page, cutoff):
    """核对单位净值与累计净值，两者不替代；官网日期不能越过本轮截止日。"""
    if value.get("currentPage") != page or not isinstance(value.get("dataList"), list):
        raise ValueError("EXPOSURE_OFFICIAL_NAV_PAGE_INVALID")
    rows = []
    for item in value["dataList"]:
        day = date.fromisoformat(item["date"])
        unit, cumulative = Decimal(str(item["netvalue"])), Decimal(str(item["totalnetvalue"]))
        if item.get("fundcode") != "002112" or day > cutoff:
            raise ValueError("EXPOSURE_OFFICIAL_NAV_SCOPE_INVALID")
        if not unit.is_finite() or not cumulative.is_finite() or unit <= 0 or cumulative <= 0:
            raise ValueError("EXPOSURE_OFFICIAL_NAV_VALUE_INVALID")
        rows.append(
            {
                "date": str(day),
                "unit_nav": str(unit),
                "accum_nav": str(cumulative),
                "day_change_pct": item.get("dayinc"),
                "status": item.get("todaystatus"),
            }
        )
    return rows


def acquire_nav(*, latest_only=False):
    """历史全页留存，日常只刷新第一页；各日保留首次取得时间而非假造历史采集时刻。"""
    output = SUPPLEMENT / "official-nav.json"
    previous = read(output) if output.exists() else {"rows": []}
    previous_rows = {row["date"]: row for row in previous["rows"]}
    rows, receipts = {}, []
    cutoff = now().date()
    with client() as c:
        page, expected, pages = 1, None, 1
        while page <= pages:
            params = {"fundcode": "002112", "from": "2015-11-16", "to": str(cutoff), "pages": f"{page}-100"}
            raw, receipt = fetch(c, NAV_URL, params=params, reuse=False)
            value = json.loads(raw)
            if expected is None:
                expected, pages = int(value["totalCount"]), int(value["totalPage"])
                if not 1 <= pages <= 100 or expected > 10000:
                    raise ValueError("EXPOSURE_OFFICIAL_NAV_LIMIT")
            if int(value["totalCount"]) != expected or int(value["totalPage"]) != pages:
                raise ValueError("EXPOSURE_OFFICIAL_NAV_COUNT_CHANGED")
            parsed = parse_nav_page(value, page, cutoff)
            for row in parsed:
                if row["date"] in rows:
                    raise ValueError("EXPOSURE_OFFICIAL_NAV_DUPLICATE")
                old = previous_rows.get(row["date"])
                first_seen = (
                    old["first_seen_at"]
                    if old and old["unit_nav"] == row["unit_nav"] and old["accum_nav"] == row["accum_nav"]
                    else receipt["received_at"]
                )
                rows[row["date"]] = {**row, "first_seen_at": first_seen, "receipt": receipt}
            receipts.append(receipt)
            if latest_only:
                break
            page += 1
    if not latest_only and len(rows) != expected:
        raise ValueError("EXPOSURE_OFFICIAL_NAV_INCOMPLETE")
    if latest_only:
        rows = {**previous_rows, **rows}
    value = {
        "at": now().isoformat(),
        "fund_code": "002112",
        "source": "DBFUND_OFFICIAL_PUBLIC",
        "rows": [rows[k] for k in sorted(rows)],
        "catalog_total": expected,
        "full_history_checked": not latest_only,
        "training_eligible": False,
        "receipts": receipts,
        "historical_publication_timestamps_verified": False,
    }
    # 变更前版本保留，官网同一 URL 日后修订也不会覆盖掉旧证据。
    archive = SUPPLEMENT / "official-nav-versions" / (digest(value) + ".json")
    save(archive, value)
    save(output, value, replace=True)
    return {"rows": len(rows), "latest": value["rows"][-1]["date"], "unit_nav": value["rows"][-1]["unit_nav"]}


def fill_live_nav_gap(nav, required, asof):
    """仅补输入留存中的缺失日：先逐日核对相邻来源一致性，绝不覆盖数据库净值或历史标签。"""
    if [r["nav_date"] for r in nav] == list(required):
        return nav
    output = SUPPLEMENT / "official-nav.json"
    if not output.exists():
        return nav
    official = {date.fromisoformat(r["date"]): r for r in read(output)["rows"]}
    existing = {r["nav_date"]: r for r in nav}
    overlap = set(required) & existing.keys() & official.keys()
    # 正常场景为 60 天已有、最后一天来源延迟。大量缺历史时不偷偷更换整段数据来源。
    if len(overlap) < len(required) - 1:
        return nav
    if any(Decimal(official[d]["unit_nav"]) != existing[d]["unit_nav"] for d in overlap):
        raise ValueError("EXPOSURE_OFFICIAL_NAV_CONFLICT")
    for day in required:
        if day in existing or day not in official:
            continue
        row = official[day]
        if datetime.fromisoformat(row["first_seen_at"]) > asof:
            continue
        verified_bytes(row["receipt"])
        existing[day] = {
            "nav_date": day,
            "unit_nav": Decimal(row["unit_nav"]),
            "ann_date": None,
            "content_hash": digest(row),
            "source_code": "DBFUND_OFFICIAL_PUBLIC",
            "received_at": row["first_seen_at"],
            "official_receipt": row["receipt"],
        }
    return [existing[d] for d in required if d in existing]


def catalog(c, category, *, refresh=False):
    """按官网实际分页读到末页，去重前后都核对总数，不能把第一页当完整目录。"""
    items, receipts, expected, pages, page = [], [], None, 1, 1
    while page <= pages:
        raw, receipt = fetch(
            c, CATALOG, params={"categoryId": category, "pageNumber": page, "pageSize": 15}, reuse=not refresh
        )
        value = json.loads(raw)
        if expected is None:
            expected, pages = int(value["totalCount"]), int(value["totalPage"])
            if pages > 100 or expected > 1500:
                raise ValueError("EXPOSURE_PUBLIC_CATALOG_LIMIT")
        if int(value["totalCount"]) != expected or int(value["totalPage"]) != pages:
            raise ValueError("EXPOSURE_PUBLIC_CATALOG_CHANGED")
        if not isinstance(value.get("contents"), list):
            raise ValueError("EXPOSURE_PUBLIC_CATALOG_SCHEMA")
        items.extend(value["contents"])
        receipts.append(receipt)
        page += 1
    if len(items) != expected or len({i["contentId"] for i in items}) != expected:
        raise ValueError("EXPOSURE_PUBLIC_CATALOG_INCOMPLETE")
    return items, receipts


def article_text(raw, pdf):
    """提取文字，图片型 PDF 明确标为需 OCR，不把空提取结果当完整正文。"""
    if pdf:
        if not raw.startswith(b"%PDF"):
            raise ValueError("EXPOSURE_PUBLIC_NOT_PDF")
        # PDFium 不支持并发调用。网络可多线程，文档解析用同一个锁，并显式释放页面资源。
        with PDF_LOCK, pdfium.PdfDocument(raw) as reader:
            if len(reader) > 2000:
                raise ValueError("EXPOSURE_PUBLIC_PDF_PAGE_LIMIT")
            pages = []
            for page in reader:
                textpage = page.get_textpage()
                try:
                    pages.append(textpage.get_text_bounded())
                finally:
                    textpage.close()
                    page.close()
        return pages, "TEXT_EXTRACTED" if all(len(p.strip()) > 10 for p in pages) else "PARTIAL_TEXT_CHECK_IMAGES"
    soup = BeautifulSoup(raw, "html.parser")
    body = soup.select_one(".ueditor_content_parse") or soup.select_one(".new-div-cont-news-detail")
    if body is None:
        raise ValueError("EXPOSURE_PUBLIC_ARTICLE_BODY_UNKNOWN")
    return [body.get_text("\n", strip=True)], "TEXT_EXTRACTED"


def same_document_catalog(old, new):
    """官网 indexTime 是搜索索引刷新时间，每天变化，不是文件修订依据。"""
    return all(old.get(key) == new.get(key) for key in ("contentId", "title", "url", "activationDate", "publishDate"))


def acquire_documents(*, live=False, progress=None):
    """产品页所有资料分类取并集，再采公司新闻；基金直接相关与公司背景资料分别标记。"""
    items, categories, catalog_receipts = {}, {}, []
    directory = LIVE / "supplement" if live else SUPPLEMENT
    with client() as c:
        raw, receipt = fetch(c, PRODUCT, "html", reuse=False)
        soup = BeautifulSoup(raw, "html.parser")
        for board in soup.select("[information_categoryid]"):
            category = board["information_categoryid"]
            categories[category] = "FUND_PRODUCT_DISCLOSURE"
        news, receipt = fetch(c, NEWS_URL, "html", reuse=False)
        match = re.search(r"var\s+categoryId\s*=\s*['\"]([^'\"]+)", news.decode("utf-8"))
        if not match or not categories:
            raise ValueError("EXPOSURE_PUBLIC_CATEGORY_MISSING")
        categories[match[1]] = "MANAGER_COMPANY_NEWS"
        category_counts = {}
        for category, scope in categories.items():
            entries, receipts = catalog(c, category, refresh=live)
            catalog_receipts.extend(receipts)
            category_counts[category] = len(entries)
            for item in entries:
                key = item["contentId"]
                entry = items.setdefault(key, {"catalog": item, "scopes": [], "categories": []})
                if scope not in entry["scopes"]:
                    entry["scopes"].append(scope)
                entry["categories"].append(category)
        versioned_save(
            directory / "public-catalog.json",
            {"at": now().isoformat(), "items": items, "category_counts": category_counts, "receipts": catalog_receipts},
        )
        result = {
            "at": now().isoformat(),
            "catalog_count": len(items),
            "category_counts": category_counts,
            "documents": [],
            "errors": [],
            "external_links": [],
            "missing": [],
            "created": 0,
            "updated": 0,
            "skipped": 0,
        }
        for index, (key, entry) in enumerate(items.items(), 1):
            item = entry["catalog"]
            url = urljoin(ORIGIN, item["url"])
            # 官网迁移后的部分 CMS 链接误带自身域名前缀，恢复其明示的外部原链接。
            if url.startswith(ORIGIN + "/https://mp.weixin.qq.com/"):
                url = url[len(ORIGIN) + 1 :]
            path = directory / "documents" / (digest(key) + ".json")
            previous = source_path(ROOT, "supplement/documents/" + digest(key) + ".json") if live else path
            if urlparse(url).netloc != "www.dbfund.com.cn":
                result["external_links"].append({"content_id": key, "title": item["title"], "url": url})
                # 微信公开正文不可读时，保留已独立核对的媒体转载，不能重跑目录后把它丢掉。
                if not previous.exists():
                    result["missing"].append({"content_id": key, "reason": "PUBLIC_BODY_NOT_AVAILABLE"})
                    # 历史已知缺口继续留证；新出现的不可读正文必须使本轮显示部分失败。
                    baseline = read(SUPPLEMENT / "public-catalog.json")["items"] if live else items
                    if key not in baseline:
                        result["errors"].append({"content_id": key, "reason": "PUBLIC_BODY_NOT_AVAILABLE"})
                    continue
            try:
                old = read(previous) if previous.exists() else None
                unchanged = old and same_document_catalog(old["catalog"], item)
                if unchanged:
                    document = old
                    result["skipped"] += 1
                    if not live:
                        verified_bytes(document["receipt"])
                else:
                    pdf = urlparse(url).path.lower().endswith(".pdf")
                    raw, receipt = fetch(c, url, "pdf" if pdf else "html", reuse=not (live and old))
                    pages, extraction = article_text(raw, pdf)
                    content = item["title"] + "\n" + "\n".join(pages)
                    related = any(word in content for word in ("002112", "德邦鑫星价值"))
                    document = {
                        **entry,
                        "url": url,
                        "receipt": receipt,
                        "pages": pages,
                        "text_status": extraction,
                        "page_count": len(pages),
                        "fund_name_or_code_mentioned": related,
                        "relevance": "FUND_EXPLICIT_MENTION" if related else "COMPANY_OR_PRODUCT_CATEGORY_CONTEXT",
                        "cms_dates_are_unverified_historical_availability": True,
                        "training_eligible": False,
                    }
                    versioned_save(path, document)
                    result["updated" if old else "created"] += 1
                result["documents"].append(
                    {
                        "content_id": key,
                        "title": item["title"],
                        "file": (previous if unchanged else path).relative_to(ROOT).as_posix(),
                        "sha256": document["receipt"]["sha256"],
                        "text_status": document["text_status"],
                        "relevance": document["relevance"],
                        "scopes": entry["scopes"],
                    }
                )
            except Exception as exc:
                # 只保存异常类型，避免对外 URL 或服务端报文携带不必要的信息。
                result["errors"].append(
                    {
                        "content_id": key,
                        "title": item["title"],
                        "url": url,
                        "reason": str(exc) if str(exc).startswith("EXPOSURE_") else type(exc).__name__,
                    }
                )
            if index % 20 == 0:
                print(
                    json.dumps(
                        {
                            "public_processed": index,
                            "catalog": len(items),
                            "saved": len(result["documents"]),
                            "errors": len(result["errors"]),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                save(directory / "public-result.json", result, replace=True)
                if progress:
                    progress(index, len(items), None, "正在检查基金公告与基金公司新闻")
        result["finished_at"] = now().isoformat()
        save(directory / "public-result.json", result, replace=True)
        return {
            "catalog": len(items),
            "saved": len(result["documents"]),
            "errors": result["errors"],
            "external_links": len(result["external_links"]),
            "missing": len(result["missing"]),
            "created": result["created"],
            "updated": result["updated"],
            "skipped": result["skipped"],
        }


def recover_public_documents():
    """修复三条失效官网链接；采用实际查到的销售机构转载，保留原链接与替代来源区别。"""
    mirrors = {
        "07c17ded1edb4ab5e065000000000001": "https://www.cgbchina.com.cn/noticeNewDetail.gsp?id=2638171",
        "07c17ded1ed54ab5e065000000000001": "https://www.cgbchina.com.cn/noticeNewDetail.gsp?id=2465965",
        "07c17ded1e5b4ab5e065000000000001": "https://www.cgbchina.com.cn/noticeNewDetail.gsp?id=1880455",
    }
    items = read(SUPPLEMENT / "public-catalog.json")["items"]
    result = {"at": now().isoformat(), "recovered": [], "external": [], "errors": []}
    with httpx.Client(timeout=httpx.Timeout(30, connect=5), headers={"User-Agent": "Mozilla/5.0"}) as c:
        for key, url in mirrors.items():
            try:
                raw = bounded_get(c, url, 4_000_000)
                soup = BeautifulSoup(raw, "html.parser")
                body = soup.select_one("#textContent")
                if body is None:
                    raise ValueError("EXPOSURE_MIRROR_BODY_MISSING")
                content = body.get_text("\n", strip=True)
                title = items[key]["catalog"]["title"]
                if re.sub(r"\s+", "", title) not in re.sub(r"\s+", "", content):
                    raise ValueError("EXPOSURE_MIRROR_TITLE_MISMATCH")
                sha, filename = blob(raw, "html")
                receipt = {
                    "url": url,
                    "file": filename,
                    "sha256": sha,
                    "received_at": now().isoformat(),
                    "source_code": "CGB_FUND_DISCLOSURE_REPUBLICATION",
                }
                value = {
                    **items[key],
                    "url": url,
                    "receipt": receipt,
                    "pages": [content],
                    "page_count": 1,
                    "text_status": "TEXT_EXTRACTED",
                    "relevance": "FUND_EXPLICIT_MENTION",
                    "original_official_url": items[key]["catalog"]["url"],
                    "is_official_site_copy": False,
                    "historical_first_seen_verified": False,
                    "training_eligible": False,
                }
                path = SUPPLEMENT / "documents" / (digest(key) + ".json")
                if not path.exists():
                    save(path, value)
                result["recovered"].append({"content_id": key, "url": url, "receipt": receipt})
            except Exception as exc:
                result["errors"].append({"content_id": key, "reason": type(exc).__name__})
        for entry in read(SUPPLEMENT / "public-result.json")["external_links"]:
            url = entry["url"]
            if urlparse(url).scheme != "https" or urlparse(url).netloc != "mp.weixin.qq.com":
                result["external"].append({**entry, "status": "UNSUPPORTED_EXTERNAL_HOST"})
                continue
            try:
                # 不跟随可能要求验证的跳转；仍保存实际返回，不能把微信限制当作正文。
                r = c.get(url)
                if len(r.content) > 4_000_000:
                    raise ValueError("EXPOSURE_EXTERNAL_RESPONSE_TOO_LARGE")
                sha, filename = blob(r.content, "html")
                body = BeautifulSoup(r.content, "html.parser").select_one("#js_content")
                result["external"].append(
                    {
                        **entry,
                        "http_status": r.status_code,
                        "status": "TEXT_AVAILABLE" if r.status_code == 200 and body else "PUBLIC_TEXT_UNAVAILABLE",
                        "pages": [body.get_text("\n", strip=True)] if r.status_code == 200 and body else [],
                        "receipt": {"sha256": sha, "file": filename, "url": url, "received_at": now().isoformat()},
                    }
                )
            except httpx.HTTPError as exc:
                result["external"].append({**entry, "status": "PUBLIC_TEXT_UNAVAILABLE", "reason": type(exc).__name__})
            time.sleep(0.3)
    save(SUPPLEMENT / "public-recovery.json", result, replace=True)
    return {
        "recovered": len(result["recovered"]),
        "errors": result["errors"],
        "external_available": sum(r["status"] == "TEXT_AVAILABLE" for r in result["external"]),
        "external_unavailable": sum(r["status"] != "TEXT_AVAILABLE" for r in result["external"]),
    }
