"""基金市场公开报告目录与原文；仅正常公开读取，缓存正文、留存失败和来源版本。"""

import hashlib
import json
import re
import time
from datetime import date
from pathlib import Path

import httpx

from app.integrations.dbfund_reports import bounded_get
from app.integrations.public_fund_reports import BODY_URL, CATALOG_URL, report_period
from app.services.fund_exposure_common import _publish, now
from app.services.fund_materials_store import versioned_save

TABLE_URL = "https://fundf10.eastmoney.com/FundArchivesDatas.aspx"


def period(title: str):
    """完整报告的更正版可采集；只有更正说明、提示或摘要的公告不能冒充完整报告。"""
    return report_period(title)


class MarketReportClient:
    """每批最多500次请求、串行限速；连续失败熔断。原文成功后按哈希复用，不定期重下。"""

    def __init__(self, root: Path):
        self.root, self.count, self.last, self.failures = root, 0, 0.0, 0
        self.body_downloads = 0
        self.client = httpx.Client(
            timeout=httpx.Timeout(25, connect=5),
            follow_redirects=False,
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://fundf10.eastmoney.com/"},
        )

    def close(self):
        self.client.close()

    def _get(self, url: str, params: dict) -> bytes:
        if url not in (CATALOG_URL, BODY_URL, TABLE_URL):
            raise ValueError("MARKET_REPORT_HOST")
        if self.count >= 500 or self.failures >= 5:
            raise ValueError("MARKET_REPORT_REQUEST_LIMIT_OR_SOURCE_UNAVAILABLE")
        time.sleep(max(0, 0.35 - (time.monotonic() - self.last)))
        self.count += 1
        self.last = time.monotonic()
        try:
            raw = bounded_get(self.client, url, 20_000_000, params=params)
            self.failures = 0
            return raw
        except httpx.HTTPError:
            self.failures += 1
            raise

    def _archive(self, raw: bytes, suffix: str, url: str, params: dict) -> dict:
        sha = hashlib.sha256(raw).hexdigest()
        p = self.root / "sources" / (sha + suffix)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            if hashlib.sha256(p.read_bytes()).hexdigest() != sha:
                raise ValueError("MARKET_REPORT_ARCHIVE_HASH")
        else:
            # 来源正文不可覆盖；并发入口由业务锁串行执行。
            _publish(p, raw)
        return {
            "file": p.relative_to(self.root).as_posix(),
            "sha256": sha,
            "url": url,
            "params": params,
            "received_at": now().isoformat(),
        }

    def catalog(self, code: str) -> list[dict]:
        """完整翻页校验总数、身份与重复项；目录空值不能引起历史删除。"""
        if not re.fullmatch(r"[0-9]{6}", code):
            raise ValueError("MARKET_REPORT_FUND_CODE")
        entries, seen, expected = [], set(), None
        for page in range(1, 21):
            params = {"fundcode": code, "pageIndex": page, "pageSize": 100, "type": 3}
            raw = self._get(CATALOG_URL, params)
            value = json.loads(raw)
            rows, total = value.get("Data"), value.get("TotalCount")
            if (
                value.get("ErrCode") != 0
                or not isinstance(rows, list)
                or type(total) is not int
                or not 0 <= total <= 2000
            ):
                raise ValueError("MARKET_REPORT_CATALOG_SCHEMA")
            expected = total if expected is None else expected
            if total != expected or value.get("PageIndex") != page or len(rows) > 100:
                raise ValueError("MARKET_REPORT_CATALOG_CHANGED")
            self._archive(raw, ".json", CATALOG_URL, params)
            for row in rows:
                article = row.get("ID", "")
                if row.get("FUNDCODE") != code or not re.fullmatch(r"AN\d{18}", article) or article in seen:
                    raise ValueError("MARKET_REPORT_CATALOG_IDENTITY")
                seen.add(article)
                target = period(row.get("TITLE", ""))
                if target:
                    published = date.fromisoformat(str(row["PUBLISHDATEDesc"])[:10]).isoformat()
                    if not target[0] <= published <= str(now().date()):
                        raise ValueError("MARKET_REPORT_CATALOG_DATE")
                    entries.append(
                        {
                            "id": article,
                            "title": row["TITLE"],
                            "end": target[0],
                            "kind": target[1],
                            "published": published,
                            "attachment_type": row.get("ATTACHTYPE"),
                        }
                    )
            if len(seen) == total:
                return entries
            if not rows:
                break
        raise ValueError("MARKET_REPORT_CATALOG_INCOMPLETE")

    def body(self, code: str, entry: dict, previous: dict | None = None) -> dict:
        """同一目录版本只下载一次；解析失败可重用原文，空正文仍留回执并允许下次重试。"""
        if previous and previous.get("source"):
            ref = previous["source"]
            p = (self.root / ref["file"]).resolve()
            if not p.is_relative_to(self.root.resolve()) or hashlib.sha256(p.read_bytes()).hexdigest() != ref["sha256"]:
                raise ValueError("MARKET_REPORT_CACHED_HASH")
            return ref
        params = {"client_source": "web_fund", "show_all": 1, "art_code": entry["id"]}
        raw = self._get(BODY_URL, params)
        self.body_downloads += 1
        value = json.loads(raw)
        data = value.get("data") or {}
        receipt = self._archive(raw, ".json", BODY_URL, params)
        receipt.update(article_id=entry["id"], title=entry["title"], published_on=entry["published"])
        versioned_save(self.root / "receipts" / code / (entry["id"] + ".json"), receipt)
        if (
            value.get("success") != 1
            or data.get("art_code") != entry["id"]
            or code not in {s.get("stock") for s in data.get("security", [])}
            or len(data.get("notice_content", "")) < 1500
        ):
            raise ValueError("MARKET_REPORT_BODY_UNAVAILABLE")
        return receipt

    def table(self, code: str, end: str, kind: str) -> dict:
        """仅新报告解析需要时读取同报告期的公开持仓表；不在日常目录检查中读取行情。"""
        params = {
            "type": "jjcc" if kind == "stocks" else "zqcc",
            "code": code,
            "topline": 1000,
            "year": int(end[:4]),
            "month": int(end[5:7]),
        }
        raw = self._get(TABLE_URL, params)
        ref = self._archive(raw, ".txt", TABLE_URL, params)
        return {**ref, "fund_code": code, "kind": kind}


def read_source(root: Path, reference: dict) -> bytes:
    path = (root / reference["file"]).resolve()
    if not path.is_relative_to(root.resolve()) or path.stat().st_size > 20_000_000:
        raise ValueError("MARKET_REPORT_SOURCE_PATH")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("MARKET_REPORT_SOURCE_HASH")
    return raw
