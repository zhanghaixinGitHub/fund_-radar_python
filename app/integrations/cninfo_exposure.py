"""按历史已披露持仓的关联期间读取巨潮公开公告；不使用账号、绕过登录或付费限制。"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from datetime import time as day_time
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from app.integrations.dbfund_supplement import article_text
from app.services.direction_1d_protocol import ZONE, digest
from app.services.fund_exposure_common import blob, now, read, save
from app.services.fund_exposure_features import select_report
from app.services.fund_exposure_quotes import reports, safe_error
from app.services.fund_exposure_supplement import SUPPLEMENT, plan, verified_bytes
from app.services.fund_materials_store import LIVE, merge_catalog, source_path, versioned_save

ORIGIN = "https://www.cninfo.com.cn"
QUERY = ORIGIN + "/new/hisAnnouncement/query"
REFERER = ORIGIN + "/new/commonUrl/pageOfSearch?url=disclosure/list/search"


def holding_intervals():
    """公告窗口按当时可见的报告取持仓，加前置 30 天以覆盖刚披露时的公司背景。"""
    path = SUPPLEMENT / "announcement-scope.json"
    if path.exists():
        return read(path)
    rs = reports()
    stop = datetime.strptime(plan()["end_date"], "%Y%m%d").date()
    day = date(2021, 1, 1)
    windows = {}
    while day <= stop:
        selected = select_report(rs, datetime.combine(day, day_time(23, 59), ZONE))
        if (day - date.fromisoformat(selected["report_end"])).days <= 210:
            for h in selected["holdings"]:
                code = h["stock_code"]
                periods = windows.setdefault(code, [])
                start = max(date(2020, 12, 2), day - timedelta(days=30))
                if periods and start <= date.fromisoformat(periods[-1][1]) + timedelta(days=1):
                    periods[-1][1] = str(day)
                else:
                    periods.append([str(start), str(day)])
        day += timedelta(days=1)
    result = {
        "created_at": now().isoformat(),
        "windows": windows,
        "meaning": "DISCLOSED_HOLDING_RELEVANCE_WINDOWS_PLUS_30_CALENDAR_DAYS_NOT_REALTIME_HOLDINGS",
        "report_hash": digest(rs),
        "cutoff": str(stop),
        "page_size": 30,
        "maximum_pages_per_window": 150,
        "maximum_new_requests": 40000,
        "maximum_pdf_bytes": 20_000_000,
        "minimum_interval_seconds": 0.6,
        "training_eligible": False,
    }
    save(path, result)
    return result


class PublicClient:
    """共享限速、有限重试与内容校验；公共搜索只请求正常页面使用的参数。"""

    def __init__(self, *, cache_period=None, request_limit=None):
        self.lock, self.last, self.count = threading.Lock(), 0.0, 0
        self.cache_period = cache_period
        self.request_limit = request_limit
        # HTTPX Client 可在线程间共享连接池；复用 TLS 连接，避免每个公开附件都重新握手。
        self.client = httpx.Client(
            timeout=httpx.Timeout(30, connect=5),
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=8),
            headers={"User-Agent": "Mozilla/5.0", "Referer": REFERER},
        )

    def close(self):
        """完成一轮后释放连接池；后台服务不留下闲置公开网站连接。"""
        self.client.close()

    def fetch(self, url, params=None, *, reuse=True):
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.netloc not in {"www.cninfo.com.cn", "static.cninfo.com.cn"}:
            raise ValueError("EXPOSURE_CNINFO_HOST_NOT_ALLOWED")
        request = {"url": url, "params": params}
        if self.cache_period and not parsed.path.lower().endswith(".pdf"):
            request["checked_period"] = self.cache_period
        path = SUPPLEMENT / "cninfo-receipts" / (digest(request) + ".json")
        if reuse and path.exists():
            receipt = read(path)
            return verified_bytes(receipt), receipt
        for attempt in range(2):
            try:
                with self.lock:
                    if self.count >= (self.request_limit or holding_intervals()["maximum_new_requests"]):
                        raise ValueError("EXPOSURE_CNINFO_REQUEST_BUDGET")
                    # 搜索接口保持低频；公开附件 CDN 采用有限并发，避免大批 PDF 阻塞整个采集。
                    interval = 0.15 if parsed.netloc == "static.cninfo.com.cn" else 0.6
                    time.sleep(max(0, interval - (time.monotonic() - self.last)))
                    self.last = time.monotonic()
                    self.count += 1
                with self.client.stream("POST" if params is not None else "GET", url, data=params) as response:
                    response.raise_for_status()
                    raw = bytearray()
                    for part in response.iter_bytes():
                        raw.extend(part)
                        if len(raw) > (100_000_000 if parsed.path.lower().endswith(".pdf") else 4_000_000):
                            raise ValueError("EXPOSURE_CNINFO_RESPONSE_TOO_LARGE")
                suffix = "pdf" if parsed.path.lower().endswith(".pdf") else "json"
                sha, filename = blob(bytes(raw), suffix)
                receipt = {
                    **request,
                    "sha256": sha,
                    "file": filename,
                    "received_at": now().isoformat(),
                    "source": "CNINFO_PUBLIC_DISCLOSURE",
                    "training_eligible": False,
                }
                versioned_save(path, receipt)
                return bytes(raw), receipt
            except httpx.HTTPError:
                if attempt:
                    raise
                time.sleep(2)


def acquire_announcements(*, live_scope=None, progress=None):
    """每个关联期间读到最后一页；保留完整目录、标题、发布时间和原文地址。"""
    scope = live_scope or holding_intervals()
    client = PublicClient(
        cache_period=scope.get("check_id", scope["cutoff"]) if live_scope else None,
        request_limit=scope.get("maximum_new_requests"),
    )
    directory = LIVE / "supplement" if live_scope else SUPPLEMENT
    raw, receipt = client.fetch(ORIGIN + "/new/data/szse_stock.json")
    stocks = {x["code"]: x for x in json.loads(raw)["stockList"]}
    versioned_save(directory / "cninfo-stock-map.json", {"rows": stocks, "receipt": receipt})

    def one(code):
        output = directory / "company-announcements" / (code + ".json")
        if output.exists() and (not live_scope or read(output).get("scope_hash") == digest(scope)):
            value = read(output)
            return value
        stock = stocks.get(code.split(".")[0])
        if stock is None:
            return {"code": code, "status": "MISSING_PUBLIC_STOCK_ID", "rows": []}
        rows, receipts = {}, []
        for start, end in scope["windows"][code]:
            page, pages, expected, returned = 1, 1, None, 0
            window_ids = set()
            while page <= pages:
                params = {
                    "pageNum": page,
                    "pageSize": 30,
                    "column": "szse",
                    "tabName": "fulltext",
                    "plate": "",
                    "stock": stock["code"] + "," + stock["orgId"],
                    "searchkey": "",
                    "secid": "",
                    "category": "",
                    "trade": "",
                    "seDate": start + "~" + end,
                    "sortName": "",
                    "sortType": "",
                    "isHLtitle": "true",
                }
                raw, receipt = client.fetch(QUERY, params)
                value = json.loads(raw)
                if expected is None:
                    expected = int(value["totalAnnouncement"])
                    # 官网 totalpages 在实测中始终为 1，实际下一页由 hasMore 表示；按总数核算页数。
                    pages = max(1, (expected + 29) // 30)
                    if pages > scope["maximum_pages_per_window"]:
                        raise ValueError("EXPOSURE_CNINFO_PAGINATION_LIMIT")
                if int(value["totalAnnouncement"]) != expected:
                    raise ValueError("EXPOSURE_CNINFO_CATALOG_CHANGED")
                items = value.get("announcements") or []
                returned += len(items)
                for item in items:
                    if item["announcementId"] in window_ids:
                        raise ValueError("EXPOSURE_CNINFO_DUPLICATE_PAGE")
                    window_ids.add(item["announcementId"])
                    if code.split(".")[0] not in item["secCode"].split(","):
                        raise ValueError("EXPOSURE_CNINFO_STOCK_MISMATCH")
                    announced = datetime.fromtimestamp(item["announcementTime"] / 1000, ZONE)
                    if not start <= str(announced.date()) <= end:
                        raise ValueError("EXPOSURE_CNINFO_DATE_MISMATCH")
                    item = {
                        **item,
                        "title_plain": BeautifulSoup(item["announcementTitle"], "html.parser").get_text(),
                        "announced_at_source": announced.isoformat(),
                        "receipt": receipt,
                        "source_time_meaning": "SOURCE_DISPLAY_TIME_NOT_INDEPENDENTLY_PROVEN_FIRST_RELEASE",
                    }
                    rows[item["announcementId"]] = item
                receipts.append(receipt)
                page += 1
            if returned != expected:
                raise ValueError("EXPOSURE_CNINFO_COUNT_INCOMPLETE")
        if live_scope:
            old = source_path(SUPPLEMENT.parent, "supplement/company-announcements/" + code + ".json")
            rows = {
                r["announcementId"]: r
                for r in merge_catalog(
                    read(old)["rows"] if old.exists() else [],
                    list(rows.values()),
                    "announcementId",
                    "announced_at_source",
                    scope["windows"][code],
                )
            }
        value = {
            "code": code,
            "status": "AVAILABLE" if rows else "SOURCE_RETURNED_EMPTY",
            "rows": list(rows.values()),
            "receipts": receipts,
            "scope_hash": digest(scope),
            "first_seen_at": now().isoformat(),
            "training_eligible": False,
        }
        versioned_save(output, value)
        return value

    result = {"started_at": now().isoformat(), "companies": len(scope["windows"]), "rows": 0, "errors": []}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(one, code): code for code in scope["windows"]}
        for done, f in enumerate(as_completed(futures), 1):
            try:
                value = f.result()
                result["rows"] += len(value["rows"])
                if value["status"] == "MISSING_PUBLIC_STOCK_ID":
                    result["errors"].append({"code": futures[f], "reason": value["status"]})
            except Exception as exc:
                result["errors"].append({"code": futures[f], "reason": safe_error(exc)})
            result["completed_companies"] = done
            save(directory / "announcement-result.json", result, replace=True)
            if progress:
                progress(done, len(futures), futures[f], "正在更新公司公告目录")
            if done % 20 == 0 or done == len(futures):
                print(
                    json.dumps(
                        {"announcement_companies": done, "rows": result["rows"], "errors": len(result["errors"])}
                    ),
                    flush=True,
                )
    result["finished_at"] = now().isoformat()
    save(directory / "announcement-result.json", result, replace=True)
    client.close()
    return result


def acquire_attachments(*, live=False, progress=None):
    """下载已核对公司与日期的公告原文；先保存原文件，再提取正文，逐份记录缺口。"""
    limits_path = SUPPLEMENT / "attachment-limits.json"
    if not limits_path.exists():
        save(
            limits_path,
            {
                "created_at": now().isoformat(),
                "maximum_pdf_bytes": 100_000_000,
                "maximum_pdf_pages": 2000,
                "reason": "NORMAL_PUBLIC_LONG_REPORT_EXCEEDED_INITIAL_20MB",
                "new_purchase_cny": 0,
                "parser": "pypdfium2_5.13.0_with_mutex",
            },
        )
    client, entries = PublicClient(), {}
    directory = LIVE / "supplement" if live else SUPPLEMENT
    for path in (directory / "company-announcements").glob("*.json"):
        for item in read(path)["rows"]:
            entries[item["announcementId"]] = item

    def one(item):
        key = item["announcementId"]
        url = urljoin("https://static.cninfo.com.cn/", item["adjunctUrl"])
        relative = "supplement/company-documents/" + digest(key) + ".json"
        output = directory / "company-documents" / (digest(key) + ".json")
        previous = source_path(SUPPLEMENT.parent, relative) if live else output
        metadata_hash = digest(
            {k: item.get(k) for k in ("adjunctUrl", "adjunctSize", "announcementTitle", "announcementTime")}
        )
        if previous.exists():
            value = read(previous)
            unchanged = value.get("catalog_hash") == metadata_hash or (
                not value.get("catalog_hash") and not item.get("source_changed")
            )
            if unchanged and value["receipt"]["url"] == url and value["title"] == item["title_plain"]:
                # 已核验的历史 PDF 只复用索引，避免每次同步重读或下载两万份原文。
                if not live:
                    verified_bytes(value["receipt"])
                elif not (SUPPLEMENT.parent / value["receipt"]["file"]).exists():
                    raise ValueError("EXPOSURE_RAW_FILE_MISSING")
                return value["text_status"], "skipped"
        if not urlparse(url).path.lower().endswith(".pdf"):
            raise ValueError("EXPOSURE_CNINFO_ATTACHMENT_NOT_PDF")
        raw, receipt = client.fetch(url, reuse=not previous.exists())
        pages, status = article_text(raw, True)
        versioned_save(
            output,
            {
                "announcement_id": key,
                "stock_code": item["secCode"],
                "title": item["title_plain"],
                "announced_at_source": item["announced_at_source"],
                "receipt": receipt,
                "pages": pages,
                "text_status": status,
                "training_eligible": False,
                "catalog_hash": metadata_hash,
            },
        )
        return status, "updated" if previous.exists() else "created"

    result = {
        "started_at": now().isoformat(),
        "expected": len(entries),
        "saved": 0,
        "partial_text": 0,
        "errors": [],
        "created": 0,
        "updated": 0,
        "skipped": 0,
    }
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(one, item): key for key, item in entries.items()}
        for done, f in enumerate(as_completed(futures), 1):
            try:
                status, outcome = f.result()
                result[outcome] += 1
                result["saved"] += 1
                result["partial_text"] += status != "TEXT_EXTRACTED"
            except Exception as exc:
                result["errors"].append({"announcement_id": futures[f], "reason": safe_error(exc)})
            if done % 50 == 0 or done == len(futures):
                save(directory / "attachment-result.json", result, replace=True)
                if progress:
                    progress(done, len(futures), None, "正在保存新增公司公告原文")
                print(
                    json.dumps(
                        {
                            "company_pdfs": done,
                            "expected": len(entries),
                            "saved": result["saved"],
                            "errors": len(result["errors"]),
                        }
                    ),
                    flush=True,
                )
    result["finished_at"] = now().isoformat()
    save(directory / "attachment-result.json", result, replace=True)
    client.close()
    return result
