"""Tushare 公募基金目录与日净值同步编排服务。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from typing import Protocol
from uuid import UUID

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import get_engine
from app.integrations.tushare import (
    TushareFundBasic,
    TushareFundClient,
    TushareFundCompany,
    TushareFundNav,
    TushareIntegrationError,
)
from app.repositories.fund_sync import (
    TUSHARE_SOURCE_CODE,
    FundCatalogUpsert,
    NavDailyUpsert,
    WriteStats,
    complete_sync_run,
    create_sync_run,
    ensure_tushare_source,
    fail_sync_run,
    upsert_fund_catalog_batch,
    upsert_nav_daily_batch,
)

logger = get_logger(__name__)


class TushareFundProvider(Protocol):
    """同步服务依赖的最小外部数据源契约，便于以假实现覆盖自动化测试。"""

    def list_fund_companies(self) -> tuple[TushareFundCompany, ...]:
        """返回基金公司名称映射数据。"""

    def list_fund_basics(self) -> tuple[TushareFundBasic, ...]:
        """返回全市场基金目录分片合并后的数据。"""

    def list_nav_daily(self, nav_date: date) -> tuple[TushareFundNav, ...]:
        """返回指定净值日期的批量净值。"""


@dataclass(frozen=True)
class SyncOutcome:
    """单次同步的脱敏结果摘要，可安全写入 Celery 返回值或命令行。"""

    sync_run_id: UUID
    sync_type: str
    requested_nav_date: date | None
    fetched_count: int
    created_count: int
    updated_count: int
    skipped_count: int

    def to_payload(self) -> dict[str, str | int | None]:
        """转换为 JSON 可序列化的任务结果，不携带外部原始数据。"""
        return {
            "sync_run_id": str(self.sync_run_id),
            "sync_type": self.sync_type,
            "requested_nav_date": self.requested_nav_date.isoformat() if self.requested_nav_date else None,
            "fetched_count": self.fetched_count,
            "created_count": self.created_count,
            "updated_count": self.updated_count,
            "skipped_count": self.skipped_count,
        }


class TushareFundSyncService:
    """将已授权 Tushare 基金数据批量规范化并幂等落库。

    目录同步从不删除历史行；净值同步必须指定交易日，按日期一次请求并
    只写入目录中存在的基金份额。读取接口不调用本服务。
    """

    def __init__(
        self,
        *,
        provider: TushareFundProvider | None = None,
        engine: Engine | None = None,
        batch_size: int | None = None,
    ) -> None:
        settings = get_settings()
        self._engine = engine or get_engine()
        self._batch_size = batch_size or settings.tushare_sync_batch_size
        if self._batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self._owns_provider = provider is None
        self._provider = provider or TushareFundClient(
            token=settings.tushare_token.get_secret_value(),
            api_url=settings.tushare_api_url,
            connect_timeout_seconds=settings.tushare_connect_timeout_seconds,
            read_timeout_seconds=settings.tushare_read_timeout_seconds,
            max_retries=settings.tushare_max_retries,
            catalog_max_rows_per_query=settings.tushare_catalog_max_rows_per_query,
        )

    def close(self) -> None:
        """关闭本服务创建的外部 HTTP 客户端。"""
        if self._owns_provider and isinstance(self._provider, TushareFundClient):
            self._provider.close()

    def sync_catalog(self) -> SyncOutcome:
        """同步基金公司映射和基金目录，异常时记录失败运行并向调用方抛出。

        Returns:
            成功运行的新增、更新、跳过数量。

        Raises:
            TushareIntegrationError: 外部接口、字段或数据完整性校验失败。
        """
        source_id, sync_run_id = self._start_run(sync_type="CATALOG", requested_nav_date=None)
        try:
            companies = self._provider.list_fund_companies()
            basics = self._provider.list_fund_basics()
            records, invalid_count = _normalize_catalog_records(companies, basics)
            write_stats = WriteStats(skipped_count=invalid_count)
            for batch in _chunked(records, self._batch_size):
                with Session(self._engine) as session, session.begin():
                    write_stats = write_stats.combine(upsert_fund_catalog_batch(session, batch))
            outcome = SyncOutcome(
                sync_run_id=sync_run_id,
                sync_type="CATALOG",
                requested_nav_date=None,
                fetched_count=len(basics),
                created_count=write_stats.created_count,
                updated_count=write_stats.updated_count,
                skipped_count=write_stats.skipped_count,
            )
            self._complete_run(source_id, outcome, write_stats)
            logger.info(
                "tushare_fund_sync.sync_catalog >>> completed run_id=%s fetched=%s created=%s updated=%s skipped=%s",
                sync_run_id,
                outcome.fetched_count,
                outcome.created_count,
                outcome.updated_count,
                outcome.skipped_count,
            )
            return outcome
        except Exception as error:
            self._record_failure(source_id, sync_run_id, error)
            logger.exception("tushare_fund_sync.sync_catalog >>> failed run_id=%s", sync_run_id)
            raise

    def sync_nav_daily(self, nav_date: date) -> SyncOutcome:
        """同步指定日期的批量净值，不对每只基金发起单独远程请求。

        Args:
            nav_date: 需要同步的基金净值日期。

        Returns:
            成功运行的新增、更新、跳过数量。

        Raises:
            TushareIntegrationError: 外部接口、字段或数据完整性校验失败。
        """
        source_id, sync_run_id = self._start_run(sync_type="NAV_DAILY", requested_nav_date=nav_date)
        try:
            navs = self._provider.list_nav_daily(nav_date)
            records, invalid_count = _normalize_nav_records(navs, nav_date)
            if not records:
                raise TushareIntegrationError(
                    "fund_nav", f"no valid NAV records returned for nav_date={nav_date.isoformat()}"
                )
            write_stats = WriteStats(skipped_count=invalid_count)
            for batch in _chunked(records, self._batch_size):
                with Session(self._engine) as session, session.begin():
                    write_stats = write_stats.combine(
                        upsert_nav_daily_batch(session, source_id=source_id, records=batch)
                    )
            outcome = SyncOutcome(
                sync_run_id=sync_run_id,
                sync_type="NAV_DAILY",
                requested_nav_date=nav_date,
                fetched_count=len(navs),
                created_count=write_stats.created_count,
                updated_count=write_stats.updated_count,
                skipped_count=write_stats.skipped_count,
            )
            self._complete_run(source_id, outcome, write_stats)
            logger.info(
                "tushare_fund_sync.sync_nav_daily >>> completed run_id=%s nav_date=%s "
                "fetched=%s created=%s updated=%s skipped=%s",
                sync_run_id,
                nav_date.isoformat(),
                outcome.fetched_count,
                outcome.created_count,
                outcome.updated_count,
                outcome.skipped_count,
            )
            return outcome
        except Exception as error:
            self._record_failure(source_id, sync_run_id, error)
            logger.exception(
                "tushare_fund_sync.sync_nav_daily >>> failed run_id=%s nav_date=%s",
                sync_run_id,
                nav_date.isoformat(),
            )
            raise

    def _start_run(self, *, sync_type: str, requested_nav_date: date | None) -> tuple[UUID, UUID]:
        with Session(self._engine) as session, session.begin():
            source = ensure_tushare_source(session)
            run = create_sync_run(
                session,
                source_id=source.source_id,
                sync_type=sync_type,
                requested_nav_date=requested_nav_date,
            )
            return source.source_id, run.sync_run_id

    def _complete_run(self, source_id: UUID, outcome: SyncOutcome, write_stats: WriteStats) -> None:
        with Session(self._engine) as session, session.begin():
            complete_sync_run(
                session,
                source_id=source_id,
                sync_run_id=outcome.sync_run_id,
                fetched_count=outcome.fetched_count,
                write_stats=write_stats,
            )

    def _record_failure(self, source_id: UUID, sync_run_id: UUID, error: Exception) -> None:
        try:
            with Session(self._engine) as session, session.begin():
                fail_sync_run(
                    session,
                    source_id=source_id,
                    sync_run_id=sync_run_id,
                    error_summary=_safe_error_summary(error),
                )
        except Exception:
            logger.exception("tushare_fund_sync._record_failure >>> unable to persist failed run_id=%s", sync_run_id)


def _normalize_catalog_records(
    companies: tuple[TushareFundCompany, ...], basics: tuple[TushareFundBasic, ...]
) -> tuple[tuple[FundCatalogUpsert, ...], int]:
    """规范化管理人、代码、分类和状态；无关键字段的目录记录只计数跳过。"""
    company_name_by_short_name = _build_company_name_mapping(companies)
    records: list[FundCatalogUpsert] = []
    source_ts_code_by_fund_code: dict[str, str] = {}
    invalid_count = 0
    for basic in basics:
        fund_code = _normalize_fund_code(basic.ts_code)
        manager_name = _normalize_manager_name(basic.management, company_name_by_short_name)
        if fund_code is None or manager_name is None or not basic.name.strip():
            invalid_count += 1
            continue
        previous_ts_code = source_ts_code_by_fund_code.get(fund_code)
        if previous_ts_code is not None and previous_ts_code != basic.ts_code:
            raise TushareIntegrationError(
                "fund_basic", f"normalized fund_code collision for {fund_code}: {previous_ts_code} vs {basic.ts_code}"
            )
        source_ts_code_by_fund_code[fund_code] = basic.ts_code
        records.append(
            FundCatalogUpsert(
                fund_code=fund_code,
                manager_name=manager_name,
                # Tushare fund_basic 没有稳定的主产品键；以份额简称作为新主实体的保守回退。
                master_name=basic.name.strip(),
                fund_name=basic.name.strip(),
                fund_type=_normalize_fund_type(basic.fund_type),
                status=_normalize_fund_status(basic.status),
                share_class=_derive_share_class(basic.name),
                established_date=basic.found_date,
            )
        )
    return tuple(records), invalid_count


def _normalize_nav_records(
    navs: tuple[TushareFundNav, ...], requested_nav_date: date
) -> tuple[tuple[NavDailyUpsert, ...], int]:
    """校验请求日期与返回日期一致，并计算不含审计字段的净值内容哈希。"""
    by_key: dict[tuple[str, date], NavDailyUpsert] = {}
    invalid_count = 0
    for nav in navs:
        fund_code = _normalize_fund_code(nav.ts_code)
        if fund_code is None or nav.nav_date != requested_nav_date:
            invalid_count += 1
            continue
        record = NavDailyUpsert(
            fund_code=fund_code,
            nav_date=nav.nav_date,
            unit_nav=nav.unit_nav,
            accumulated_nav=nav.accumulated_nav,
            content_hash=_nav_content_hash(fund_code, nav),
        )
        key = (record.fund_code, record.nav_date)
        existing = by_key.get(key)
        if existing is not None and existing.content_hash != record.content_hash:
            raise TushareIntegrationError("fund_nav", f"conflicting duplicate NAV for fund_code={fund_code}")
        by_key[key] = record
    return tuple(by_key[key] for key in sorted(by_key)), invalid_count


def _build_company_name_mapping(companies: tuple[TushareFundCompany, ...]) -> dict[str, str]:
    """仅保留简称唯一对应的公司全称，歧义简称回退基金列表原值。"""
    mapping: dict[str, str] = {}
    ambiguous_short_names: set[str] = set()
    for company in companies:
        if company.short_name is None:
            continue
        short_name = company.short_name.strip()
        full_name = company.name.strip()
        if not short_name or not full_name or short_name in ambiguous_short_names:
            continue
        existing = mapping.get(short_name)
        if existing is None:
            mapping[short_name] = full_name
        elif existing != full_name:
            mapping.pop(short_name)
            ambiguous_short_names.add(short_name)
    return mapping


def _normalize_manager_name(manager_name: str | None, company_name_by_short_name: dict[str, str]) -> str | None:
    if manager_name is None:
        return None
    normalized = manager_name.strip()
    if not normalized:
        return None
    return company_name_by_short_name.get(normalized, normalized)


def _normalize_fund_code(ts_code: str) -> str | None:
    """将 Tushare 的 `000001.OF`/`510050.SH` 统一为项目现有的代码口径。"""
    fund_code = ts_code.split(".", maxsplit=1)[0].strip()
    if not fund_code or len(fund_code) > 32:
        return None
    return fund_code


def _normalize_fund_type(source_fund_type: str | None) -> str:
    source_type = (source_fund_type or "").upper()
    if "QDII" in source_type:
        return "QDII"
    if "FOF" in source_type:
        return "FOF"
    if "货币" in source_type:
        return "MONEY"
    if "债" in source_type:
        return "BOND"
    if "指数" in source_type:
        return "INDEX"
    if "混合" in source_type:
        return "MIXED"
    if "股票" in source_type:
        return "STOCK"
    return "OTHER"


def _normalize_fund_status(source_status: str | None) -> str:
    return {"L": "ACTIVE", "D": "DELISTED", "I": "ISSUING"}.get((source_status or "").upper(), "UNKNOWN")


def _derive_share_class(fund_name: str) -> str:
    """只识别常见的末尾份额字母，其余保持未指定，避免将 ETF 等简称误判。"""
    suffix = fund_name.strip()[-1:].upper()
    return suffix if suffix in {"A", "C", "E", "H", "R", "Y"} else "UNSPECIFIED"


def _nav_content_hash(fund_code: str, nav: TushareFundNav) -> str:
    payload = {
        "fund_code": fund_code,
        "nav_date": nav.nav_date.isoformat(),
        "unit_nav": format(nav.unit_nav, "f"),
        "accumulated_nav": format(nav.accumulated_nav, "f") if nav.accumulated_nav is not None else None,
        "source_code": TUSHARE_SOURCE_CODE,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _chunked(records: tuple[FundCatalogUpsert, ...] | tuple[NavDailyUpsert, ...], size: int) -> Iterator:
    for start in range(0, len(records), size):
        yield records[start : start + size]


def _safe_error_summary(error: Exception) -> str:
    """生成不含请求体、Token 和堆栈的持久化错误摘要。"""
    return f"{type(error).__name__}: {str(error).replace(chr(10), ' ').replace(chr(13), ' ')[:450]}"
