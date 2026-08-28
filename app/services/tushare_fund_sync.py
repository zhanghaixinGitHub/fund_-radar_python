"""Tushare 公募基金目录与日净值同步编排服务。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol
from uuid import UUID

from sqlalchemy import Engine, func, select
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
    MarketSyncTarget,
    NavDailyUpsert,
    WriteStats,
    assign_source_fund_codes,
    complete_sync_run,
    create_sync_run,
    ensure_tushare_source,
    fail_sync_run,
    get_latest_nav_dates,
    list_active_market_sync_targets,
    upsert_fund_catalog_batch,
    upsert_nav_daily_batch,
)

logger = get_logger(__name__)

_MARKET_NAV_INCREMENTAL_LOCK_KEY = 7_089_123_006


class TushareFundProvider(Protocol):
    """同步服务依赖的最小外部数据源契约，便于以假实现覆盖自动化测试。"""

    def list_fund_companies(self) -> tuple[TushareFundCompany, ...]:
        """返回基金公司名称映射数据。"""

    def list_fund_basics(self) -> tuple[TushareFundBasic, ...]:
        """返回全市场基金目录分片合并后的数据。"""

    def list_fund_basics_by_ts_codes(self, ts_codes: tuple[str, ...]) -> tuple[TushareFundBasic, ...]:
        """返回指定完整 Tushare 代码的基金目录。"""

    def resolve_fund_basics_by_fund_codes(self, fund_codes: tuple[str, ...]) -> tuple[TushareFundBasic, ...]:
        """返回指定六位展示代码经来源验证后的唯一完整 Tushare 目录。"""

    def list_nav_daily(self, nav_date: date) -> tuple[TushareFundNav, ...]:
        """返回指定净值日期的批量净值。"""

    def list_nav_history(
        self, ts_code: str, *, start_date: date | None = None, end_date: date | None = None
    ) -> tuple[TushareFundNav, ...]:
        """返回指定基金份额的历史净值。"""


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


class MarketNavIncrementalPreconditionError(ValueError):
    """基金市场范围未完成同源基线或来源代码映射时拒绝启动日常增量同步。"""


class MarketNavIncrementalInProgressError(RuntimeError):
    """同一环境已有基金市场增量同步运行时拒绝重复启动。"""


MarketNavProgressReporter = Callable[[int, int, str | None, str], None]


@dataclass(frozen=True)
class MarketNavIncrementalWindow:
    """一只基金市场份额本轮需从本地水位后补齐的净值日期窗口。"""

    ts_code: str
    fund_code: str
    start_date: date
    end_date: date


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
            nav_max_rows_per_query=settings.tushare_market_nav_max_rows_per_query,
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

    def sync_market_catalog(self, ts_codes: tuple[str, ...]) -> SyncOutcome:
        """将指定的来源基金目录纳入基金市场。

        Args:
            ts_codes: 带交易标识的完整 Tushare 基金代码，必须全部唯一且可查询。

        Raises:
            TushareIntegrationError: 任一指定基金缺失、错配或目录字段无效时抛出。
        """
        source_id, sync_run_id = self._start_run(sync_type="MARKET_CATALOG", requested_nav_date=None)
        try:
            companies = self._provider.list_fund_companies()
            basics = self._provider.list_fund_basics_by_ts_codes(ts_codes)
            _ensure_market_catalog_complete(ts_codes, basics)
            records, invalid_count = _normalize_catalog_records(companies, basics)
            if invalid_count or len(records) != len(ts_codes):
                raise TushareIntegrationError("fund_basic", "market catalog contains invalid records")
            write_stats = WriteStats()
            for batch in _chunked(records, self._batch_size):
                with Session(self._engine) as session, session.begin():
                    write_stats = write_stats.combine(upsert_fund_catalog_batch(session, batch))
            outcome = SyncOutcome(
                sync_run_id=sync_run_id,
                sync_type="MARKET_CATALOG",
                requested_nav_date=None,
                fetched_count=len(basics),
                created_count=write_stats.created_count,
                updated_count=write_stats.updated_count,
                skipped_count=write_stats.skipped_count,
            )
            self._complete_run(source_id, outcome, write_stats)
            logger.info(
                "tushare_fund_sync.sync_market_catalog >>> completed run_id=%s fund_count=%s created=%s updated=%s",
                sync_run_id,
                len(ts_codes),
                outcome.created_count,
                outcome.updated_count,
            )
            return outcome
        except Exception as error:
            self._record_failure(source_id, sync_run_id, error)
            logger.exception("tushare_fund_sync.sync_market_catalog >>> failed run_id=%s", sync_run_id)
            raise

    def sync_market_nav_history(
        self, ts_codes: tuple[str, ...], *, start_date: date | None = None, end_date: date | None = None
    ) -> SyncOutcome:
        """回填已纳入基金市场份额的完整历史净值；响应校验通过后才开始写库。"""
        if start_date and end_date and start_date > end_date:
            raise ValueError("start_date must not be after end_date.")
        source_id, sync_run_id = self._start_run(sync_type="MARKET_NAV_HISTORY", requested_nav_date=None)
        try:
            navs = tuple(
                nav
                for ts_code in ts_codes
                for nav in self._provider.list_nav_history(ts_code, start_date=start_date, end_date=end_date)
            )
            records, invalid_count = _normalize_market_nav_history_records(
                navs,
                ts_codes=ts_codes,
                start_date=start_date,
                end_date=end_date,
            )
            if invalid_count or not records:
                raise TushareIntegrationError("fund_nav", "market NAV history contains invalid or empty records")
            _ensure_market_nav_history_complete(ts_codes, records)
            write_stats = WriteStats()
            for batch in _chunked(records, self._batch_size):
                with Session(self._engine) as session, session.begin():
                    write_stats = write_stats.combine(
                        upsert_nav_daily_batch(session, source_id=source_id, records=batch)
                    )
            outcome = SyncOutcome(
                sync_run_id=sync_run_id,
                sync_type="MARKET_NAV_HISTORY",
                requested_nav_date=None,
                fetched_count=len(navs),
                created_count=write_stats.created_count,
                updated_count=write_stats.updated_count,
                skipped_count=write_stats.skipped_count,
            )
            self._complete_run(source_id, outcome, write_stats)
            logger.info(
                "tushare_fund_sync.sync_market_nav_history >>> completed run_id=%s "
                "fund_count=%s fetched=%s created=%s updated=%s",
                sync_run_id,
                len(ts_codes),
                outcome.fetched_count,
                outcome.created_count,
                outcome.updated_count,
            )
            return outcome
        except Exception as error:
            self._record_failure(source_id, sync_run_id, error)
            logger.exception("tushare_fund_sync.sync_market_nav_history >>> failed run_id=%s", sync_run_id)
            raise

    def sync_market_nav_incremental(
        self,
        *,
        as_of_date: date | None = None,
        progress_reporter: MarketNavProgressReporter | None = None,
    ) -> SyncOutcome:
        """补齐基金市场中所有启用份额的缺失净值日期。

        同步范围只由数据库中的基金市场决定，与任何用户的关注列表无关。每只
        基金都必须具有可校验的精确 Tushare 代码和同源历史基线，避免猜测交易所后缀。
        """
        target_date = as_of_date or date.today()
        with _market_nav_incremental_lock(self._engine):
            return self._run_market_nav_incremental(target_date=target_date, progress_reporter=progress_reporter)

    def _run_market_nav_incremental(
        self,
        *,
        target_date: date,
        progress_reporter: MarketNavProgressReporter | None,
    ) -> SyncOutcome:
        """在已取得跨进程互斥锁后执行基金市场的实际增量同步。"""
        source_id, sync_run_id = self._start_run(
            sync_type="MARKET_NAV_INCREMENTAL", requested_nav_date=target_date
        )
        try:
            _report_market_nav_progress(progress_reporter, 0, 0, None, "正在读取基金市场同步范围")
            with Session(self._engine) as session:
                targets = list_active_market_sync_targets(session)
            ts_codes = self._resolve_market_source_fund_codes(source_id, targets)
            market_fund_codes = tuple(_require_normalized_fund_code(ts_code) for ts_code in ts_codes)
            total_steps = len(market_fund_codes) + 1
            _report_market_nav_progress(progress_reporter, 0, total_steps, None, "正在读取本地同步水位")
            with Session(self._engine) as session:
                latest_nav_dates = get_latest_nav_dates(
                    session, source_id=source_id, fund_codes=market_fund_codes
                )
            windows = _build_market_nav_incremental_windows(
                ts_codes=ts_codes,
                latest_nav_dates=latest_nav_dates,
                as_of_date=target_date,
            )
            windows_by_ts_code = {window.ts_code: window for window in windows}
            navs: list[TushareFundNav] = []
            for completed_count, ts_code in enumerate(ts_codes, start=1):
                window = windows_by_ts_code.get(ts_code)
                if window is None:
                    _report_market_nav_progress(
                        progress_reporter,
                        completed_count,
                        total_steps,
                        ts_code,
                        f"{ts_code} 已是最新，无需请求外部数据",
                    )
                    continue
                navs.extend(
                    self._provider.list_nav_history(
                        window.ts_code, start_date=window.start_date, end_date=window.end_date
                    )
                )
                _report_market_nav_progress(
                    progress_reporter,
                    completed_count,
                    total_steps,
                    ts_code,
                    f"已读取 {ts_code} 的待补齐净值",
                )
            _report_market_nav_progress(
                progress_reporter, len(market_fund_codes), total_steps, None, "正在校验并写入净值数据"
            )
            records, invalid_count = _normalize_market_nav_incremental_records(tuple(navs), windows=windows)
            if invalid_count:
                raise TushareIntegrationError("fund_nav", "market incremental NAV contains invalid records")
            write_stats = WriteStats()
            for batch in _chunked(records, self._batch_size):
                with Session(self._engine) as session, session.begin():
                    write_stats = write_stats.combine(
                        upsert_nav_daily_batch(session, source_id=source_id, records=batch)
                    )
            outcome = SyncOutcome(
                sync_run_id=sync_run_id,
                sync_type="MARKET_NAV_INCREMENTAL",
                requested_nav_date=target_date,
                fetched_count=len(navs),
                created_count=write_stats.created_count,
                updated_count=write_stats.updated_count,
                skipped_count=write_stats.skipped_count,
            )
            self._complete_run(source_id, outcome, write_stats)
            _report_market_nav_progress(progress_reporter, total_steps, total_steps, None, "同步完成")
            logger.info(
                "tushare_fund_sync.sync_market_nav_incremental >>> completed run_id=%s target_date=%s "
                "fund_count=%s windows=%s fetched=%s created=%s updated=%s skipped=%s",
                sync_run_id,
                target_date,
                len(market_fund_codes),
                len(windows),
                outcome.fetched_count,
                outcome.created_count,
                outcome.updated_count,
                outcome.skipped_count,
            )
            return outcome
        except Exception as error:
            self._record_failure(source_id, sync_run_id, error)
            logger.exception(
                "tushare_fund_sync.sync_market_nav_incremental >>> failed run_id=%s target_date=%s",
                sync_run_id,
                target_date,
            )
            raise

    def _resolve_market_source_fund_codes(
        self, source_id: UUID, targets: tuple[MarketSyncTarget, ...]
    ) -> tuple[str, ...]:
        """返回基金市场的精确来源代码，必要时从已同步日期批量反查并原子补全。"""
        if not targets:
            raise MarketNavIncrementalPreconditionError("fund market has no active Tushare fund shares")
        target_by_fund_code = {target.fund_code: target for target in targets}
        if len(target_by_fund_code) != len(targets):
            raise MarketNavIncrementalPreconditionError("fund market contains duplicate active fund codes")
        source_code_by_fund_code = {
            target.fund_code: target.source_fund_code
            for target in targets
            if target.source_fund_code is not None
        }
        missing_fund_codes = tuple(
            target.fund_code for target in targets if target.source_fund_code is None
        )
        if missing_fund_codes:
            with Session(self._engine) as session:
                latest_nav_dates = get_latest_nav_dates(
                    session, source_id=source_id, fund_codes=missing_fund_codes
                )
            no_baseline = sorted(set(missing_fund_codes) - set(latest_nav_dates))
            if no_baseline:
                raise MarketNavIncrementalPreconditionError(
                    "market NAV baseline is missing for fund_code=" + ",".join(no_baseline)
                )
            for nav_date in sorted(set(latest_nav_dates.values())):
                for nav in self._provider.list_nav_daily(nav_date):
                    fund_code = _normalize_fund_code(nav.ts_code)
                    if fund_code not in target_by_fund_code or fund_code not in missing_fund_codes:
                        continue
                    existing = source_code_by_fund_code.get(fund_code)
                    if existing is not None and existing != nav.ts_code:
                        raise TushareIntegrationError(
                            "fund_nav", f"conflicting source code for fund_code={fund_code}"
                        )
                    source_code_by_fund_code[fund_code] = nav.ts_code
            unresolved = sorted(set(missing_fund_codes) - set(source_code_by_fund_code))
            if unresolved:
                basics = self._provider.resolve_fund_basics_by_fund_codes(tuple(unresolved))
                for basic in basics:
                    fund_code = _require_normalized_fund_code(basic.ts_code)
                    if fund_code not in unresolved:
                        raise TushareIntegrationError(
                            "fund_basic", f"resolved source code is outside market scope: {basic.ts_code}"
                        )
                    existing = source_code_by_fund_code.get(fund_code)
                    if existing is not None and existing != basic.ts_code:
                        raise TushareIntegrationError(
                            "fund_basic", f"conflicting source code for fund_code={fund_code}"
                        )
                    source_code_by_fund_code[fund_code] = basic.ts_code
                unresolved = sorted(set(missing_fund_codes) - set(source_code_by_fund_code))
                if unresolved:
                    raise MarketNavIncrementalPreconditionError(
                        "could not resolve exact Tushare source code for fund_code=" + ",".join(unresolved)
                    )
            with Session(self._engine) as session, session.begin():
                assign_source_fund_codes(
                    session,
                    {fund_code: source_code_by_fund_code[fund_code] for fund_code in missing_fund_codes},
                )
        ts_codes = tuple(source_code_by_fund_code[target.fund_code] for target in targets)
        if len(set(ts_codes)) != len(ts_codes):
            raise MarketNavIncrementalPreconditionError("fund market source code mapping is not unique")
        return ts_codes

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


@contextmanager
def _market_nav_incremental_lock(engine: Engine) -> Iterator[None]:
    """使用 PostgreSQL 会话级咨询锁串行化手动和 Celery 增量同步。"""
    with engine.connect() as connection:
        acquired = bool(connection.scalar(select(func.pg_try_advisory_lock(_MARKET_NAV_INCREMENTAL_LOCK_KEY))))
        if not acquired:
            raise MarketNavIncrementalInProgressError("market NAV incremental sync is already running")
        try:
            yield
        finally:
            released = bool(connection.scalar(select(func.pg_advisory_unlock(_MARKET_NAV_INCREMENTAL_LOCK_KEY))))
            if not released:
                logger.error("tushare_fund_sync._market_nav_incremental_lock >>> advisory lock release failed")


def _report_market_nav_progress(
    reporter: MarketNavProgressReporter | None,
    completed_count: int,
    total_count: int,
    current_fund_code: str | None,
    message: str,
) -> None:
    """安全上报同步进度；展示层故障不能影响真实数据写入。"""
    if reporter is None:
        return
    try:
        reporter(completed_count, total_count, current_fund_code, message)
    except Exception:
        logger.warning("tushare_fund_sync._report_market_nav_progress >>> progress callback failed", exc_info=True)


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
                source_fund_code=basic.ts_code,
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


def _normalize_market_nav_history_records(
    navs: tuple[TushareFundNav, ...],
    *,
    ts_codes: tuple[str, ...],
    start_date: date | None,
    end_date: date | None,
) -> tuple[tuple[NavDailyUpsert, ...], int]:
    """规范化指定基金的历史净值，拒绝窗口外记录和互相冲突的重复值。"""
    expected_fund_codes = {_normalize_fund_code(ts_code) for ts_code in ts_codes}
    if None in expected_fund_codes:
        raise ValueError("ts_codes must contain normalizable fund codes.")
    by_key: dict[tuple[str, date], NavDailyUpsert] = {}
    invalid_count = 0
    for nav in navs:
        fund_code = _normalize_fund_code(nav.ts_code)
        if (
            fund_code not in expected_fund_codes
            or (start_date is not None and nav.nav_date < start_date)
            or (end_date is not None and nav.nav_date > end_date)
        ):
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


def _build_market_nav_incremental_windows(
    *,
    ts_codes: tuple[str, ...],
    latest_nav_dates: Mapping[str, date],
    as_of_date: date,
) -> tuple[MarketNavIncrementalWindow, ...]:
    """根据同源水位生成每只基金市场份额的最小补数窗口。

    Raises:
        MarketNavIncrementalPreconditionError: 任一基金市场份额没有完整历史基线时抛出。
    """
    if not ts_codes or len(set(ts_codes)) != len(ts_codes):
        raise MarketNavIncrementalPreconditionError("market fund code list must be non-empty and unique")
    fund_code_by_ts_code = {ts_code: _require_normalized_fund_code(ts_code) for ts_code in ts_codes}
    if len(set(fund_code_by_ts_code.values())) != len(fund_code_by_ts_code):
        raise MarketNavIncrementalPreconditionError("market fund code list must normalize to unique fund codes")
    missing_fund_codes = sorted(
        fund_code for fund_code in fund_code_by_ts_code.values() if fund_code not in latest_nav_dates
    )
    if missing_fund_codes:
        raise MarketNavIncrementalPreconditionError(
            "market NAV baseline is missing for fund_code=" + ",".join(missing_fund_codes)
        )
    return tuple(
        MarketNavIncrementalWindow(
            ts_code=ts_code,
            fund_code=fund_code,
            start_date=latest_nav_dates[fund_code] + timedelta(days=1),
            end_date=as_of_date,
        )
        for ts_code, fund_code in fund_code_by_ts_code.items()
        if latest_nav_dates[fund_code] < as_of_date
    )


def _normalize_market_nav_incremental_records(
    navs: tuple[TushareFundNav, ...], *, windows: tuple[MarketNavIncrementalWindow, ...]
) -> tuple[tuple[NavDailyUpsert, ...], int]:
    """只接受每只基金自身增量窗口内的净值，冲突重复值拒绝写入。"""
    window_by_fund_code = {window.fund_code: window for window in windows}
    by_key: dict[tuple[str, date], NavDailyUpsert] = {}
    invalid_count = 0
    for nav in navs:
        fund_code = _normalize_fund_code(nav.ts_code)
        window = window_by_fund_code.get(fund_code) if fund_code is not None else None
        if window is None or nav.nav_date < window.start_date or nav.nav_date > window.end_date:
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


def _ensure_market_catalog_complete(
    requested_ts_codes: tuple[str, ...], basics: tuple[TushareFundBasic, ...]
) -> None:
    """确保每个请求的完整 Tushare 代码都恰好对应一条目录记录。"""
    requested = set(requested_ts_codes)
    returned = {basic.ts_code for basic in basics}
    if len(requested) != len(requested_ts_codes) or requested != returned:
        raise TushareIntegrationError("fund_basic", "market catalog response does not match requested codes")


def _ensure_market_nav_history_complete(ts_codes: tuple[str, ...], records: tuple[NavDailyUpsert, ...]) -> None:
    """确保每只基金市场份额都至少取得一条可写入的历史净值。"""
    expected = {_normalize_fund_code(ts_code) for ts_code in ts_codes}
    returned = {record.fund_code for record in records}
    if expected != returned:
        raise TushareIntegrationError("fund_nav", "market NAV history is missing at least one requested fund")


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


def _require_normalized_fund_code(ts_code: str) -> str:
    """返回可写入的基金代码；配置代码无法规范化时失败关闭。"""
    fund_code = _normalize_fund_code(ts_code)
    if fund_code is None:
        raise MarketNavIncrementalPreconditionError(f"invalid market ts_code={ts_code}")
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
