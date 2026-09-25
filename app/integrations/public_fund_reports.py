"""固定参照基金的公开报告目录及转载正文；复用既有原文留存和持仓表校验。

仅读取正常公开页面使用的接口，不访问账户，不绕过 PDF 访问校验。
转载正文保留完整响应、来源和实际接收时间，不能冒充管理人 PDF 原文。
"""

import hashlib
import io
import json
import re
import time
from datetime import datetime, timedelta

import httpx
from pypdf import PdfReader

from app.integrations.dbfund_reports import bounded_get, parse_text
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import ROOT, blob, now, read
from app.services.fund_materials_store import versioned_save

CATALOG_URL = "https://api.fund.eastmoney.com/f10/JJGG"
BODY_URL = "https://np-cnotice-fund.eastmoney.com/api/content/ann"
PARSER = "PUBLIC_REPRINT_HOLDINGS_V1"
PEERS = {
    "002170": "东吴移动互联",
    "004237": "中欧新蓝筹",
    "004605": "富国新活力",
    "005187": "长安鑫兴",
    "005312": "万家经济新动能",
    "006038": "大成景恒",
    "007509": "华商润丰",
    "008960": "长信国防军工量化",
    "017493": "东方红新动力",
    "160323": "华夏磐泰",
}
STORE = ROOT / "peer-materials"
# 已从管理人公开披露页核对的原文补充，精确绑定基金和公告；不接受任意远程地址。
ISSUER_PDFS = {
    (
        "002170",
        "AN202403291629027319",
    ): "https://www.scfund.com.cn/upload2010/2024/03/29/020616521_402_0d84ac97-bb0a-3674-bae5-9df49f4fcc43.pdf",
    (
        "004237",
        "AN202403291629028802",
    ): "https://www.zofund.com/tempdir/minisite/20240328/f15bb8fd-9f07-46b9-864e-d6c571eb716f_1711644217312.pdf",
    (
        "004237",
        "AN202410251640474220",
    ): "https://www.zofund.com/tempdir/minisite/20241024/96c9275d-60ad-4029-860d-d6284ea0d406_1729767448933.pdf",
}


class ReportClient:
    """单轮有界请求，正文按版本复用；目录每个任务重新核对，重试沿用同一检查标识。"""

    def __init__(self, check_id: str):
        self.check_id, self.count, self.last = check_id, 0, 0.0
        self.failed_requests = 0
        self.client = httpx.Client(
            timeout=httpx.Timeout(30, connect=5),
            follow_redirects=False,
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://fundf10.eastmoney.com/"},
        )

    def close(self):
        self.client.close()

    def get(self, url, params, *, catalog=False):
        if url not in (CATALOG_URL, BODY_URL):
            raise ValueError("REPORT_HOST_NOT_ALLOWED")
        # 基金代码和公告标识必须显式提供，禁止空参数扩大为全市场。
        if catalog and params.get("fundcode") not in PEERS:
            raise ValueError("REPORT_PEER_SCOPE_INVALID")
        if not catalog and not re.fullmatch(r"AN\d{18}", str(params.get("art_code", ""))):
            raise ValueError("REPORT_ARTICLE_ID_INVALID")
        path = STORE / "receipts" / (digest({"url": url, "params": params}) + ".json")
        old = read(path) if path.exists() else None
        if old:
            due = now() - datetime.fromisoformat(old["checked_at"]) >= timedelta(days=30)
            reuse = old.get("check_id") == self.check_id if catalog else not due
            if reuse and not old.get("invalid"):
                raw = (ROOT / old["file"]).read_bytes()
                if hashlib.sha256(raw).hexdigest() != old["sha256"]:
                    raise ValueError("REPORT_CACHED_BODY_CHANGED")
                return json.loads(raw), old
        # 连续网络失败后停止本轮外呼，后续仍可复用成功缓存；避免来源故障时逐项轰击。
        if not catalog and self.failed_requests >= 4:
            raise ValueError("REPORT_SOURCE_TEMPORARILY_UNAVAILABLE")
        for attempt in range(2):
            if self.count >= 600:
                raise ValueError("REPORT_REQUEST_BUDGET")
            time.sleep(max(0, 0.6 - (time.monotonic() - self.last)))
            self.count += 1
            self.last = time.monotonic()
            try:
                raw = bounded_get(self.client, url, 8_000_000, params=params)
                value = json.loads(raw)
                if (catalog and value.get("ErrCode") != 0) or (not catalog and value.get("success") != 1):
                    raise ValueError("REPORT_PUBLIC_SOURCE_REJECTED")
                if not catalog:
                    self.failed_requests = 0
                break
            except (httpx.HTTPError, json.JSONDecodeError):
                if attempt:
                    if not catalog:
                        self.failed_requests += 1
                    raise
                time.sleep(2)
        sha, filename = blob(raw, "json")
        receipt = {
            "url": url,
            "params": params,
            "sha256": sha,
            "file": filename,
            "received_at": old["received_at"] if old and sha == old["sha256"] else now().isoformat(),
            "checked_at": now().isoformat(),
            "check_id": self.check_id,
            "source": "EASTMONEY_PUBLIC_ISSUER_REPORT_REPRINT",
            "format": "PUBLIC_REPRINT_TEXT",
            "historical_first_seen_verified": False,
            # 同一公告正文变化时保留旧摘要，新版本最早只能从实际发现时起使用。
            "previous_sha256": old["sha256"] if old and sha != old["sha256"] else (old or {}).get("previous_sha256"),
            "revised_after_receipt": bool(
                (old or {}).get("revised_after_receipt") or (old and sha != old["sha256"] and not old.get("invalid"))
            ),
            "invalid": not bool(value.get("Data"))
            if catalog
            else not bool((value.get("data") or {}).get("notice_content")),
        }
        versioned_save(path, receipt)
        return value, receipt

    def invalidate_body(self, article_id):
        """失败正文仍保留原文和版本；下一轮允许重新询问该项，不冻结一次空值或截断。"""
        params = {"client_source": "web_fund", "show_all": 1, "art_code": article_id}
        path = STORE / "receipts" / (digest({"url": BODY_URL, "params": params}) + ".json")
        if path.exists():
            value = read(path)
            versioned_save(path, {**value, "invalid": True})

    def issuer_pdf(self, code, article_id, *, url=None):
        """仅取已核对的管理人原文，成功后按哈希复用；不会打开转载站的访问校验。"""
        if url is None:
            url = ISSUER_PDFS[(code, article_id)]
        elif code != "006038" or not re.fullmatch(
            r"https://www\.dcfund\.com\.cn/home/working/download/\d{6}/[a-zA-Z0-9]+\.pdf", url
        ):
            raise ValueError("REPORT_HOST_NOT_ALLOWED")
        path = STORE / "receipts" / (digest({"issuer_pdf": url}) + ".json")
        old = read(path) if path.exists() else None
        if old and now() - datetime.fromisoformat(old["checked_at"]) < timedelta(days=30):
            raw = (ROOT / old["file"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != old["sha256"]:
                raise ValueError("REPORT_CACHED_BODY_CHANGED")
            return raw, old
        if self.count >= 600:
            raise ValueError("REPORT_REQUEST_BUDGET")
        self.count += 1
        raw = bounded_get(self.client, url, 20_000_000)
        if not raw.startswith(b"%PDF-"):
            raise ValueError("REPORT_RESPONSE_NOT_PDF")
        sha, filename = blob(raw, "pdf")
        receipt = {
            "url": url,
            "sha256": sha,
            "file": filename,
            "received_at": old["received_at"] if old and sha == old["sha256"] else now().isoformat(),
            "checked_at": now().isoformat(),
            "source": "ISSUER_PUBLIC_REPORT",
            "format": "ISSUER_PDF",
            "historical_first_seen_verified": False,
            "previous_sha256": old["sha256"] if old and sha != old["sha256"] else (old or {}).get("previous_sha256"),
            "revised_after_receipt": bool((old or {}).get("revised_after_receipt") or (old and sha != old["sha256"])),
        }
        versioned_save(path, receipt)
        return raw, receipt


def report_period(title: str):
    """识别报告期，而非把网页日期误当持仓日期；摘要和提示性公告不作为持仓报告。"""
    compact = re.sub(r"\s+", "", title)
    translation = str.maketrans("〇零一二三四五六七八九", "00123456789")
    compact = re.sub(r"[〇零0一二三四五六七八九]{4}(?=年)", lambda m: m[0].translate(translation), compact)
    year = re.search(r"(20\d{2})年", compact)
    if not year or any(word in compact for word in ("摘要", "提示", "更正公告", "修订公告")):
        return None
    quarter = re.search(r"第?([1234一二三四])季度报告", compact)
    if quarter:
        q = int(quarter[1]) if quarter[1].isdigit() else "一二三四".index(quarter[1]) + 1
        return f"{year[1]}-{q * 3:02d}-{'30' if q in (2, 3) else '31'}", "QUARTER"
    if "中期报告" in compact or "半年度报告" in compact:
        return year[1] + "-06-30", "HALF"
    if "年度报告" in compact:
        return year[1] + "-12-31", "ANNUAL"
    return None


def catalog(client: ReportClient, code: str) -> tuple[list, list]:
    """完整翻页并检查总数、重复页和基金归属，再限制为开发历史需要的报告。"""
    if code not in PEERS:
        raise ValueError("REPORT_PEER_SCOPE_INVALID")
    rows, receipts, seen, expected = [], [], set(), None
    for page in range(1, 21):
        value, receipt = client.get(
            CATALOG_URL,
            {"fundcode": code, "pageIndex": page, "pageSize": 20, "type": 3},
            catalog=True,
        )
        receipts.append(receipt)
        total = value.get("TotalCount")
        items = value.get("Data")
        if not isinstance(total, int) or not 0 <= total <= 400 or not isinstance(items, list):
            raise ValueError("REPORT_CATALOG_SCHEMA_OR_LIMIT")
        if expected is None:
            expected = total
        if total != expected or value.get("PageIndex") != page or len(items) > 20:
            raise ValueError("REPORT_CATALOG_COUNT_CHANGED")
        for item in items:
            if item.get("FUNDCODE") != code or item["ID"] in seen:
                raise ValueError("REPORT_CATALOG_IDENTITY_OR_DUPLICATE")
            seen.add(item["ID"])
            period = report_period(item["TITLE"])
            published = str(item.get("PUBLISHDATEDesc", ""))[:10]
            if period and "2020-09-30" <= period[0] <= "2024-09-30" and published <= "2024-12-31":
                if PEERS[code] not in re.sub(r"\s+", "", item["TITLE"]):
                    raise ValueError("REPORT_CATALOG_FUND_MISMATCH")
                rows.append({**item, "report_end": period[0], "report_type": period[1], "published_date": published})
        if len(seen) == expected:
            break
        if not items or page == 20:
            raise ValueError("REPORT_CATALOG_INCOMPLETE")
    if not rows:
        raise ValueError("REPORT_CATALOG_EMPTY")
    return rows, receipts


def acquire_one(client: ReportClient, code: str, entry: dict):
    """主要来源不可用时核对第二公开转载；缓存损坏、身份错误等不能降级掩盖。"""
    if code not in PEERS or entry.get("FUNDCODE") != code:
        raise ValueError("REPORT_PEER_SCOPE_INVALID")
    manifest = STORE / code / "manifest.json"
    previous = read(manifest).get("documents", {}).get(entry["ID"]) if manifest.exists() else None
    if previous:
        path = STORE / previous["file"]
        parsed = read(path)
        receipt = parsed["raw"]
        unchanged = all(
            parsed.get("source", {}).get(k) == entry.get(k)
            for k in ("TITLE", "report_end", "report_type", "published_date")
        )
        if (
            unchanged
            and parsed.get("parser_version") == PARSER
            and now() - datetime.fromisoformat(receipt["checked_at"]) < timedelta(days=30)
        ):
            if hashlib.sha256((ROOT / receipt["file"]).read_bytes()).hexdigest() != receipt["sha256"]:
                raise ValueError("REPORT_CACHED_BODY_CHANGED")
            return parsed, path, False
    try:
        return _acquire_one(client, code, entry)
    except (httpx.HTTPError, ValueError) as exc:
        allowed = {
            "REPORT_SOURCE_TEMPORARILY_UNAVAILABLE",
            "REPORT_BODY_EMPTY_OR_TRUNCATED",
            "REPORT_PUBLIC_SOURCE_REJECTED",
            "EXPOSURE_EQUITY_TOTAL_MISMATCH",
        }
        if isinstance(exc, ValueError) and str(exc) not in allowed:
            raise
        return _acquire_one(client, code, entry, fallback=True)


def _acquire_one(client: ReportClient, code: str, entry: dict, *, fallback=False):
    """核对目录、转载正文身份和日期；使用共享解析器核验金额、序号、行业合计。"""
    if code not in PEERS or entry.get("FUNDCODE") != code:
        raise ValueError("REPORT_PEER_SCOPE_INVALID")
    issuer = ((code, entry["ID"]) in ISSUER_PDFS and not fallback) or (fallback and code == "006038")
    if fallback and not issuer:
        from app.integrations.sohu_fund_reports import report

        value, receipt = report(client, code, entry)
    elif issuer:
        if fallback:
            from app.integrations.dcfund_reports import pdf as dc_pdf

            raw, receipt, official_entry = dc_pdf(client, code, entry)
        else:
            raw, receipt = client.issuer_pdf(code, entry["ID"])
            official_entry = {"title": entry["TITLE"], "published_date": entry["published_date"]}
        pdf = PdfReader(io.BytesIO(raw))
        if len(pdf.pages) > 180:
            raise ValueError("REPORT_PAGE_LIMIT")
        pages = [page.extract_text() for page in pdf.pages]
        value = {
            "data": {
                "art_code": entry["ID"],
                "security": [{"stock": code}],
                "notice_title": official_entry["title"],
                "notice_content": "\n".join(pages),
                "notice_date": official_entry["published_date"],
                "attach_url": receipt["url"],
            }
        }
    else:
        value, receipt = client.get(BODY_URL, {"client_source": "web_fund", "show_all": 1, "art_code": entry["ID"]})
    data = value.get("data") or {}
    text = data.get("notice_content")
    if data.get("art_code") != entry["ID"] or code not in {s.get("stock") for s in data.get("security", [])}:
        raise ValueError("REPORT_FUND_IDENTITY_MISMATCH")
    if not isinstance(text, str) or len(text) < 1500:
        raise ValueError("REPORT_BODY_EMPTY_OR_TRUNCATED")
    if PEERS[code] not in data.get("notice_title", "") or report_period(data["notice_title"]) != (
        entry["report_end"],
        entry["report_type"],
    ):
        raise ValueError("REPORT_BODY_PERIOD_MISMATCH")
    version = "-" + digest(receipt["received_at"])[:12] if receipt.get("revised_after_receipt") else ""
    target = STORE / code / "reports" / (receipt["sha256"] + version + "-" + PARSER + ".json")
    if target.exists():
        return read(target), target, False
    master = re.search(r"基金主代码\s*[:：]?\s*(\d{6})", text)
    parsed = parse_text(
        pages if issuer else [text],
        data["notice_title"],
        fund_code=code,
        fund_name=PEERS[code],
        master_code=master[1] if master else code,
    )
    # 历史转载可能比报告送出时间更晚。两者取较晚日期，不能用早一天的报告日期倒推可得性。
    public_day = max(parsed["published_date"], entry["published_date"], str(data.get("notice_date", ""))[:10])
    from datetime import date
    from datetime import time as day_time

    from app.services.direction_1d_protocol import ZONE

    public_date = date.fromisoformat(public_day)
    if public_date < date.fromisoformat(parsed["report_end"]) or public_date > date(2024, 12, 31):
        raise ValueError("REPORT_PUBLICATION_DATE_INVALID")
    available_at = datetime.combine(public_date + timedelta(days=1), day_time(8), ZONE).isoformat()
    if receipt.get("revised_after_receipt"):
        available_at = max(available_at, receipt["received_at"])
    parsed.update(
        {
            "available_at": available_at,
            "source_publication_date": public_day,
            "raw": receipt,
            "source": entry,
            "parsed_at": now().isoformat(),
            "parser_version": PARSER,
            "document_format": "ISSUER_PDF" if issuer else "PUBLIC_REPRINT_TEXT",
            "original_pdf_saved": issuer,
            "original_pdf_url": data.get("attach_url"),
            "training_eligible": False,
            "availability_basis": "LATER_DECLARED_AND_REPRINT_DATE_NEXT_DAY_0800_RECONSTRUCTION",
            "product_description_excerpt": text[:18000],
        }
    )
    versioned_save(target, parsed)
    return parsed, target, True
