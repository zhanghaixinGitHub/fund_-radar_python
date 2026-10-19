"""由Codex定时任务调用的独立市场持仓检查；无需启动Vue、Java或FastAPI网页服务。"""

import re
from datetime import date, timedelta
from pathlib import Path
from uuid import uuid4

import httpx
from sqlalchemy import text

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import get_engine
from app.integrations.market_fund_reports import MarketReportClient, period
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import now, read, save
from app.services.fund_exposure_runtime import execution_lock
from app.services.fund_holdings_publication import publish_report
from app.services.fund_materials_store import versioned_save
from app.services.market_holdings_parse import parse_report

logger = get_logger(__name__)
DISCLOSURE_MONTHS = frozenset({1, 2, 3, 4, 7, 8, 10})


def due(day: date, last_checked_on: str | None) -> bool:
    """年报1—3月、中报7—8月及季报月份每天查；其余月份每周查，漏跑可在本周补查。"""
    last = date.fromisoformat(last_checked_on) if last_checked_on else date.min
    start = day if day.month in DISCLOSURE_MONTHS else day - timedelta(days=day.weekday())
    return last < start


def market_scope() -> list[dict]:
    """每轮重新读取市场有效目录，新增基金自动纳入；不读取用户关注或个人持仓。"""
    with get_engine().connect() as connection:
        connection.execute(text("SET TRANSACTION READ ONLY"))
        connection.execute(text("SET LOCAL statement_timeout='15000'"))
        rows = connection.execute(
            text(
                "SELECT fund_code,fund_name FROM fund_share_class "
                "WHERE source_code='TUSHARE_PRO_FUND' AND status='ACTIVE' ORDER BY fund_code"
            )
        )
        values = [dict(row) for row in rows.mappings()]
    if not values or len(values) > 500 or any(not re.fullmatch(r"[0-9]{6}", row["fund_code"]) for row in values):
        raise ValueError("MARKET_HOLDINGS_SCOPE_EMPTY_OR_LIMIT")
    return values


class MarketHoldingsMaintenance:
    """持久断点、逐基金失败隔离；只有成功正文可复用，失败从不删除历史报告。"""

    def __init__(self, root: Path | None = None, directory: Path | None = None):
        self.directory = (directory or Path(get_settings().fund_material_directory)).resolve()
        self.root = (root or self.directory / "market-maintenance").resolve()

    def check(self, *, force: bool = False, funds: list[dict] | None = None, client=None) -> dict:
        path = self.root / "state.json"
        previous = read(path) if path.exists() else {}
        if not force and not due(now().date(), previous.get("last_checked_on")):
            return {"status": "NOT_DUE", "last_checked_on": previous["last_checked_on"], "downloads": 0}
        # 与现有002112资料维护共用数据库锁；定时任务与手工任务不能并发覆盖持仓快照。
        with execution_lock() as locked:
            if not locked:
                return {"status": "BUSY", "downloads": 0}
            previous = read(path) if path.exists() else {}
            if not force and not due(now().date(), previous.get("last_checked_on")):
                return {"status": "NOT_DUE", "last_checked_on": previous["last_checked_on"], "downloads": 0}
            scope = funds if funds is not None else market_scope()
            own_client = client is None
            client = client or MarketReportClient(self.root)
            result = {
                "run_id": str(uuid4()),
                "started_at": now().isoformat(),
                "status": "RUNNING",
                "scope_count": len(scope),
                "funds": [],
                "downloads": 0,
                "published": 0,
                "last_checked_on": previous.get("last_checked_on"),
            }
            save(path, result, replace=True)
            logger.info(
                "market_holdings_maintenance.check >>> 开始检查市场持仓, run_id=%s, funds=%s",
                result["run_id"],
                len(scope),
            )
            try:
                for fund in scope:
                    try:
                        item = self._fund(client, fund)
                    except (
                        httpx.HTTPError,
                        ValueError,
                        OSError,
                        KeyError,
                        ArithmeticError,
                        TypeError,
                        IndexError,
                    ) as exc:
                        logger.exception(
                            "market_holdings_maintenance.check >>> 基金检查失败, run_id=%s, fund=%s",
                            result["run_id"],
                            fund["fund_code"],
                        )
                        item = {"fund_code": fund["fund_code"], "status": "FAILED", "errors": [type(exc).__name__]}
                    result["funds"].append(item)
                    result["published"] += item.get("published", 0)
                    save(path, result, replace=True)
            finally:
                result["downloads"] = client.body_downloads
                result["requests"] = client.count
                if own_client:
                    client.close()
            result["status"] = (
                "PARTIAL_SUCCESS" if any(f["status"] != "SUCCEEDED" for f in result["funds"]) else "SUCCEEDED"
            )
            result["finished_at"] = now().isoformat()
            result["last_checked_on"] = now().date().isoformat()
            versioned_save(path, result)
            save(self.root / "runs" / (result["run_id"] + ".json"), result)
            logger.info(
                "market_holdings_maintenance.check >>> 检查结束, run_id=%s, status=%s, downloads=%s, published=%s",
                result["run_id"],
                result["status"],
                result["downloads"],
                result["published"],
            )
            return result

    def _fund(self, client, fund: dict) -> dict:
        code = fund["fund_code"]
        snapshot = self.directory / (code + ".json")
        reports = read(snapshot)["reports"] if snapshot.exists() else []
        checkpoint = self.root / "funds" / (code + ".json")
        state = read(checkpoint) if checkpoint.exists() else {"fund_code": code, "entries": {}}
        entries = state["entries"]
        catalog = client.catalog(code)
        if not catalog:
            raise ValueError("MARKET_HOLDINGS_CATALOG_EMPTY")
        start = max((r["endDate"] for r in reports), default=str(now().date() - timedelta(days=366)))
        published_start = state.setdefault(
            "initial_published_through",
            max((r["publishedDate"] for r in reports), default=str(now().date() - timedelta(days=366))),
        )
        known = {(r["endDate"], r["publishedDate"], (period(r["title"]) or (None, None))[1]) for r in reports}
        errors, published, unchanged = [], 0, 0
        for entry in sorted(catalog, key=lambda e: (e["published"], e["id"])):
            old = entries.get(entry["id"])
            fingerprint = digest(entry)
            if old and old["fingerprint"] == fingerprint and old["status"] in {"EXISTING", "PUBLISHED"}:
                unchanged += 1
                continue
            if (
                old is None
                and not state.get("catalog_initialized")
                and (entry["end"], entry["published"], entry["kind"]) in known
                and not any(word in entry["title"] for word in ("更正", "修订", "更新"))
            ):
                entries[entry["id"]] = {"fingerprint": fingerprint, "entry": entry, "status": "EXISTING"}
                unchanged += 1
                continue
            # 老报告期的新更正版仍需收录；不能只按期末日期把新公告过滤掉。
            if old is None and entry["end"] < start and entry["published"] < published_start:
                continue
            current = old if old and old["fingerprint"] == fingerprint else {}
            record = {**current, "fingerprint": fingerprint, "entry": entry, "status": "DOWNLOADING"}
            entries[entry["id"]] = record
            try:
                source = client.body(code, entry, current)
                record.update(source=source, status="DOWNLOADED")
                versioned_save(checkpoint, state)
                report, issues = parse_report(client, self.root, fund, entry, source)
                before = next((r for r in reports if r["id"] == report["id"]), None)
                value = publish_report(code, report, directory=self.directory)
                reports = value["reports"]
                record.update(status="PARTIAL" if issues else "PUBLISHED", issues=issues, report_id=report["id"])
                published += int(before != next(r for r in reports if r["id"] == report["id"]))
                errors.extend({"article_id": entry["id"], **i} for i in issues)
            except (httpx.HTTPError, ValueError, OSError, KeyError, ArithmeticError, TypeError, IndexError) as exc:
                logger.exception(
                    "market_holdings_maintenance._fund >>> 新报告待重试, fund=%s, article=%s", code, entry["id"]
                )
                record.update(status="FAILED", reason=str(exc) if isinstance(exc, ValueError) else type(exc).__name__)
                errors.append({"article_id": entry["id"], "reason": record["reason"]})
            versioned_save(checkpoint, state)
        state["catalog_initialized"] = True
        state["checked_at"] = now().isoformat()
        versioned_save(checkpoint, state)
        return {
            "fund_code": code,
            "status": "PARTIAL_SUCCESS" if errors else "SUCCEEDED",
            "published": published,
            "unchanged": unchanged,
            "catalog_count": len(catalog),
            "errors": errors,
        }
