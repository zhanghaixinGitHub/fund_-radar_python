"""全市场基金公告采集；公开转载与管理人原件区分，失败不伪装为无公告。

复用已有公开来源、限速和内容寻址归档。这里仅提供原文与候选事项，
不会根据公告标题直接判断利好利空、整改完成或改变正式评级。
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import text

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import get_engine
from app.integrations.market_fund_reports import MarketReportClient, read_source
from app.integrations.public_fund_reports import BODY_URL, CATALOG_URL
from app.services.fund_exposure_common import now, read
from app.services.fund_materials_store import versioned_save

_EVENT = re.compile("基金经理|投资策略|基金合同|费率|管理费|托管费|销售服务费|监管|处罚|警示|暂停|赎回|清算|清盘|合并")
logger = get_logger(__name__)


def directory() -> Path:
    return Path(get_settings().fund_insight_directory) / "market-workflow" / "news"


class MarketNewsClient(MarketReportClient):
    """全部公告目录使用type=0，不能把仅含定期报告的type=3当作完整近期消息。"""

    def news_catalog(self, code: str, lower: str) -> list[dict]:
        if not re.fullmatch(r"[0-9]{6}", code):
            raise ValueError("MARKET_NEWS_FUND_CODE")
        entries, seen, expected = [], set(), None
        for page in range(1, 21):
            params = {"fundcode": code, "pageIndex": page, "pageSize": 100, "type": 0}
            raw = self._get(CATALOG_URL, params)
            value = json.loads(raw)
            rows, total = value.get("Data"), value.get("TotalCount")
            if value.get("ErrCode") != 0 or not isinstance(rows, list) or type(total) is not int:
                raise ValueError("MARKET_NEWS_CATALOG_SCHEMA")
            expected = total if expected is None else expected
            if total != expected or value.get("PageIndex") != page or not 0 <= total <= 2000 or len(rows) > 100:
                raise ValueError("MARKET_NEWS_CATALOG_CHANGED_OR_LIMIT")
            self._archive(raw, ".json", CATALOG_URL, params)
            for row in rows:
                article = row.get("ID", "")
                published = str(row.get("PUBLISHDATEDesc", ""))[:10]
                # 栏目公开日期独立保存；同日只有日期精度，后续预测仍须验证盘中可用时间。
                datetime.strptime(published, "%Y-%m-%d")
                if (
                    row.get("FUNDCODE") != code
                    or not re.fullmatch(r"AN\d{18}", article)
                    or article in seen
                    or published > now().date().isoformat()
                ):
                    raise ValueError("MARKET_NEWS_CATALOG_IDENTITY_DATE")
                seen.add(article)
                if published >= lower:
                    entries.append({"id": article, "title": row["TITLE"], "published": published})
            if len(seen) == total:
                return entries
            if not rows:
                raise ValueError("MARKET_NEWS_CATALOG_INCOMPLETE")
        raise ValueError("MARKET_NEWS_CATALOG_LIMIT")

    def news_body(self, code: str, entry: dict, old: dict | None = None) -> dict:
        """已取得的同一公告版本复用原文；目录标题或日期变化时重新取得并保留旧版。"""
        # 正文可能在同一公告编号下修订，每日重新确认一次，旧版本仍由内容摘要保存。
        if (
            old
            and old.get("entry") == entry
            and old.get("source")
            and now() - datetime.fromisoformat(old["source"]["received_at"]) < timedelta(hours=24)
        ):
            raw = read_source(self.root, old["source"])
            received = old["source"]["received_at"]
            source = old["source"]
        else:
            params = {"client_source": "web_fund", "show_all": 1, "art_code": entry["id"]}
            raw = self._get(BODY_URL, params)
            source = self._archive(raw, ".json", BODY_URL, params)
            received = source["received_at"]
        value = json.loads(raw)
        data = value.get("data") or {}
        body = data.get("notice_content", "")
        if (
            value.get("success") != 1
            or data.get("art_code") != entry["id"]
            or code not in {s.get("stock") for s in data.get("security", [])}
            or not isinstance(body, str)
            or len(body.strip()) < 50
        ):
            raise ValueError("MARKET_NEWS_BODY_UNVERIFIED")
        return {
            "entry": entry,
            "source": source,
            "body": body,
            "received_at": received,
            "source_kind": "PUBLIC_REPRINT",
            "publication_precision": "DAY",
            "requires_review": True,
            "candidate_material_event": bool(_EVENT.search(entry["title"])),
        }


def collect(code: str, *, client=None, root: Path | None = None, progress=lambda *args: None) -> dict:
    """采集一只基金的30日公告，目录完整才可记录检查成功；既有失败和旧公告留档。"""
    if not re.fullmatch(r"[0-9]{6}", code):
        raise ValueError("MARKET_NEWS_FUND_CODE")
    root = root or directory()
    state_path = root / "funds" / (code + ".json")
    old = read(state_path) if state_path.exists() else {}
    current = now()
    if old.get("status") == "SUCCEEDED" and old.get("finished_at"):
        if (current - datetime.fromisoformat(old["finished_at"])).total_seconds() < 600:
            return {**old, "reused": True}
    owned = client is None
    client = client or MarketNewsClient(root)
    result = {
        "fund_code": code,
        "started_at": current.isoformat(),
        "status": "RUNNING",
        "items": old.get("items", {}),
        "errors": [],
        # 候选事项必须一直等待有原文依据的审阅，不能因第二次下载内容相同就自动消失。
        "candidate_events": list(old.get("candidate_events", [])),
        "catalog_complete": False,
    }
    try:
        lower = (current.date() - timedelta(days=30)).isoformat()
        entries = client.news_catalog(code, lower)
        result["catalog_complete"] = True
        result["coverage"] = {"start": lower, "end": current.date().isoformat()}
        for i, entry in enumerate(entries, 1):
            progress(i - 1, len(entries), code, "正在核验基金公告原文")
            try:
                item = client.news_body(code, entry, result["items"].get(entry["id"]))
                previous = result["items"].get(entry["id"])
                result["items"][entry["id"]] = item
                if item["candidate_material_event"] and (
                    not previous or previous["source"]["sha256"] != item["source"]["sha256"]
                ):
                    event = {"fund_code": code, "article_id": entry["id"], "source_hash": item["source"]["sha256"]}
                    if event not in result["candidate_events"]:
                        result["candidate_events"].append(event)
            except (ValueError, OSError, KeyError, TypeError) as error:
                logger.exception(
                    "fund_market_news.collect >>> 公告原文核验未完成, fund_code=%s, article_id=%s", code, entry["id"]
                )
                result["errors"].append({"article_id": entry["id"], "reason": type(error).__name__})
        result["status"] = "PARTIAL_SUCCESS" if result["errors"] else "SUCCEEDED"
    except Exception as error:
        logger.exception("fund_market_news.collect >>> 公告目录或来源访问失败, fund_code=%s", code)
        result["status"] = "FAILED"
        result["errors"].append({"reason": type(error).__name__})
    finally:
        if owned:
            client.close()
    result["finished_at"] = now().isoformat()
    versioned_save(state_path, result)
    return result


def synchronize(code: str, *, progress=lambda *args: None, client=None, root=None) -> dict:
    """独立和批量入口共用跨进程锁；同一基金不会由两个任务同时改写公告游标。"""
    if not re.fullmatch(r"[0-9]{6}", code):
        raise ValueError("MARKET_NEWS_FUND_CODE")
    key = "market-fund-news:" + code
    with get_engine().connect() as connection:
        if not connection.scalar(text("SELECT pg_try_advisory_lock(hashtextextended(:key,0))"), {"key": key}):
            raise ValueError("MARKET_NEWS_BUSY")
        try:
            return collect(code, progress=progress, client=client, root=root)
        finally:
            connection.execute(text("SELECT pg_advisory_unlock(hashtextextended(:key,0))"), {"key": key})
