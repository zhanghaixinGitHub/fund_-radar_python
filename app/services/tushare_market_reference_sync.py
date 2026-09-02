"""免费场内基金和市场参考指数的受控同步服务。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import get_engine
from app.integrations.tushare import TushareIntegrationError
from app.integrations.tushare_market_reference import (
    TushareFundDaily,
    TushareIndexBasic,
    TushareIndexClassification,
    TushareIndexWeight,
    TushareMarketReferenceClient,
)
from app.repositories.benchmark_series import BenchmarkNavPoint, upsert_benchmark_nav_points
from app.repositories.fund_sync import (
    WriteStats,
    complete_sync_run,
    create_sync_run,
    fail_sync_run,
)
from app.repositories.market_reference_sync import (
    FundExchangeDailyUpsert,
    IndexWeightSnapshotUpsert,
    MarketIndexCatalogUpsert,
    MarketIndexClassificationUpsert,
    list_active_exchange_targets,
    list_reference_benchmark_codes,
    mark_cursor_failure,
    mark_cursor_success,
    require_tushare_source_capabilities,
    upsert_fund_exchange_daily_batch,
    upsert_index_weight_snapshots_batch,
    upsert_market_index_catalog_batch,
    upsert_market_index_classifications_batch,
)
from app.services.tushare_fund_sync import SyncOutcome

logger = get_logger(__name__)

MARKET_FREE_EXCHANGE_DAILY_SYNC_TYPE = "MARKET_FREE_EXCHANGE_DAILY"
MARKET_FREE_INDEX_CATALOG_SYNC_TYPE = "MARKET_FREE_INDEX_CATALOG"
MARKET_FREE_INDEX_CLASSIFY_SYNC_TYPE = "MARKET_FREE_INDEX_CLASSIFY"
MARKET_FREE_INDEX_DAILY_SYNC_TYPE = "MARKET_FREE_INDEX_DAILY"
MARKET_FREE_INDEX_WEIGHT_SYNC_TYPE = "MARKET_FREE_INDEX_WEIGHT"
_MARKET_REFERENCE_API_NAMES = (
    "fund_daily",
    "index_basic",
    "index_classify",
    "index_daily",
    "index_weight",
)
_HISTORY_START_DATE = date(1990, 1, 1)
_MARKET_REFERENCE_LOCK_KEY = 731_220_002

ProgressReporter = Callable[[int, int, str | None, str], None]


class MarketReferenceSyncInProgressError(RuntimeError):
    """已有进程正在同步场内基金或市场参考数据时拒绝第二次外部拉取。"""


@dataclass(frozen=True)
class MarketReferenceSyncResult:
    """市场参考同步的各阶段来源运行结果。"""

    outcomes: tuple[SyncOutcome, ...]

    @property
    def write_stats(self) -> WriteStats:
        """汇总所有阶段的本地新增、更新和跳过统计。"""
        stats = WriteStats()
        for outcome in self.outcomes:
            stats = stats.combine(
                WriteStats(
                    created_count=outcome.created_count,
                    updated_count=outcome.updated_count,
                    skipped_count=outcome.skipped_count,
                )
            )
        return stats

    @property
    def fetched_count(self) -> int:
        """返回各阶段来源返回记录数之和。"""
        return sum(outcome.fetched_count for outcome in self.outcomes)


class MarketReferenceSyncService:
    """按数据域同步已授权场内基金和市场参考指数数据。

    该服务只接收已有父运行标识，不创建预测或模型结果。每个实体请求和
    写入独立失败隔离，成功后才更新其水位；任一实体失败会保留其他成功数据
    并抛出异常交由父任务标为部分成功。
    """

    def __init__(
        self,
        *,
        client: TushareMarketReferenceClient | None = None,
    ) -> None:
        settings = get_settings()
        self._engine = get_engine()
        self._owns_client = client is None
        self._client = client or TushareMarketReferenceClient(
            token=settings.tushare_token.get_secret_value(),
            api_url=settings.tushare_api_url,
            connect_timeout_seconds=settings.tushare_connect_timeout_seconds,
            read_timeout_seconds=settings.tushare_read_timeout_seconds,
            max_retries=settings.tushare_max_retries,
            catalog_max_rows_per_query=settings.tushare_market_reference_max_rows_per_query,
            max_rows_per_query=settings.tushare_market_reference_max_rows_per_query,
        )
        self._index_catalog_markets = _parse_csv_setting(settings.tushare_index_catalog_markets)

    def close(self) -> None:
        """关闭由本服务创建的 HTTP 客户端。"""
        if self._owns_client:
            self._client.close()

    def sync_all(
        self,
        *,
        parent_sync_run_id: UUID,
        as_of_date: date,
        progress_reporter: ProgressReporter | None = None,
    ) -> MarketReferenceSyncResult:
        """执行场内基金、指数目录、分类、日线和权重的受控补齐。"""
        with _market_reference_sync_lock(self._engine):
            return self._sync_all_unlocked(
                parent_sync_run_id=parent_sync_run_id,
                as_of_date=as_of_date,
                progress_reporter=progress_reporter,
            )

    def _sync_all_unlocked(
        self,
        *,
        parent_sync_run_id: UUID,
        as_of_date: date,
        progress_reporter: ProgressReporter | None,
    ) -> MarketReferenceSyncResult:
        """在已取得 PostgreSQL 咨询锁后执行各个市场参考数据域。"""
        with Session(self._engine) as session:
            source = require_tushare_source_capabilities(session, _MARKET_REFERENCE_API_NAMES)
            source_id = source.source_id

        outcomes: list[SyncOutcome] = []
        outcomes.append(
            self._sync_exchange_daily(
                source_id=source_id,
                parent_sync_run_id=parent_sync_run_id,
                as_of_date=as_of_date,
                progress_reporter=progress_reporter,
            )
        )
        outcomes.append(
            self._sync_index_catalog(
                source_id=source_id,
                parent_sync_run_id=parent_sync_run_id,
                progress_reporter=progress_reporter,
            )
        )
        outcomes.append(
            self._sync_index_classification(
                source_id=source_id,
                parent_sync_run_id=parent_sync_run_id,
                progress_reporter=progress_reporter,
            )
        )
        daily_outcome, weight_outcome = self._sync_registered_reference_indices(
            source_id=source_id,
            parent_sync_run_id=parent_sync_run_id,
            as_of_date=as_of_date,
            progress_reporter=progress_reporter,
        )
        outcomes.extend((daily_outcome, weight_outcome))
        return MarketReferenceSyncResult(outcomes=tuple(outcomes))

    def _sync_exchange_daily(
        self,
        *,
        source_id: UUID,
        parent_sync_run_id: UUID,
        as_of_date: date,
        progress_reporter: ProgressReporter | None,
    ) -> SyncOutcome:
        run_id = self._start_stage(
            source_id=source_id,
            parent_sync_run_id=parent_sync_run_id,
            sync_type=MARKET_FREE_EXCHANGE_DAILY_SYNC_TYPE,
            start_date=_HISTORY_START_DATE,
            end_date=as_of_date,
        )
        with Session(self._engine) as session:
            targets = list_active_exchange_targets(session)
        stats = WriteStats()
        fetched_count = 0
        failures: list[str] = []
        total = len(targets)
        for current, target in enumerate(targets, start=1):
            _report(progress_reporter, current, total, target.fund_code, "正在同步场内基金日线")
            try:
                source_rows = self._client.list_fund_exchange_daily(
                    target.source_fund_code,
                    start_date=_HISTORY_START_DATE,
                    end_date=as_of_date,
                )
                records = tuple(_to_exchange_daily_upsert(target.fund_code, row) for row in source_rows)
                with Session(self._engine) as session, session.begin():
                    write_stats = upsert_fund_exchange_daily_batch(session, source_id=source_id, records=records)
                    if source_rows:
                        mark_cursor_success(
                            session,
                            source_id=source_id,
                            dataset_code="FUND_EXCHANGE_DAILY",
                            entity_key=target.source_fund_code,
                            last_successful_data_date=max(row.trade_date for row in source_rows),
                            last_sync_run_id=run_id,
                        )
                stats = stats.combine(write_stats)
                fetched_count += len(source_rows)
            except (TushareIntegrationError, ValueError) as error:
                failures.append(target.source_fund_code)
                self._record_entity_failure(
                    source_id=source_id,
                    dataset_code="FUND_EXCHANGE_DAILY",
                    entity_key=target.source_fund_code,
                    error=error,
                )
        return self._finish_stage(
            source_id=source_id,
            sync_run_id=run_id,
            sync_type=MARKET_FREE_EXCHANGE_DAILY_SYNC_TYPE,
            requested_date=as_of_date,
            fetched_count=fetched_count,
            write_stats=stats,
            failures=failures,
        )

    def _sync_index_catalog(
        self,
        *,
        source_id: UUID,
        parent_sync_run_id: UUID,
        progress_reporter: ProgressReporter | None,
    ) -> SyncOutcome:
        run_id = self._start_stage(
            source_id=source_id,
            parent_sync_run_id=parent_sync_run_id,
            sync_type=MARKET_FREE_INDEX_CATALOG_SYNC_TYPE,
            start_date=None,
            end_date=None,
        )
        _report(progress_reporter, 0, 1, None, "正在同步指数目录")
        try:
            source_rows = self._client.list_index_basics(self._index_catalog_markets)
            records = tuple(_to_index_catalog_upsert(row) for row in source_rows)
            with Session(self._engine) as session, session.begin():
                stats = upsert_market_index_catalog_batch(session, source_id=source_id, records=records)
                mark_cursor_success(
                    session,
                    source_id=source_id,
                    dataset_code="INDEX_CATALOG",
                    entity_key="GLOBAL",
                    last_successful_data_date=None,
                    last_sync_run_id=run_id,
                )
        except (TushareIntegrationError, ValueError) as error:
            self._record_entity_failure(
                source_id=source_id,
                dataset_code="INDEX_CATALOG",
                entity_key="GLOBAL",
                error=error,
            )
            return self._finish_stage(
                source_id=source_id,
                sync_run_id=run_id,
                sync_type=MARKET_FREE_INDEX_CATALOG_SYNC_TYPE,
                requested_date=None,
                fetched_count=0,
                write_stats=WriteStats(),
                failures=("GLOBAL",),
            )
        _report(progress_reporter, 1, 1, None, "指数目录同步完成")
        return self._finish_stage(
            source_id=source_id,
            sync_run_id=run_id,
            sync_type=MARKET_FREE_INDEX_CATALOG_SYNC_TYPE,
            requested_date=None,
            fetched_count=len(source_rows),
            write_stats=stats,
            failures=(),
        )

    def _sync_index_classification(
        self,
        *,
        source_id: UUID,
        parent_sync_run_id: UUID,
        progress_reporter: ProgressReporter | None,
    ) -> SyncOutcome:
        run_id = self._start_stage(
            source_id=source_id,
            parent_sync_run_id=parent_sync_run_id,
            sync_type=MARKET_FREE_INDEX_CLASSIFY_SYNC_TYPE,
            start_date=None,
            end_date=None,
        )
        _report(progress_reporter, 0, 1, None, "正在同步指数分类")
        try:
            source_rows = self._client.list_index_classifications()
            records = tuple(_to_index_classification_upsert(row) for row in source_rows)
            with Session(self._engine) as session, session.begin():
                stats = upsert_market_index_classifications_batch(session, source_id=source_id, records=records)
                mark_cursor_success(
                    session,
                    source_id=source_id,
                    dataset_code="INDEX_CLASSIFY",
                    entity_key="GLOBAL",
                    last_successful_data_date=None,
                    last_sync_run_id=run_id,
                )
        except (TushareIntegrationError, ValueError) as error:
            self._record_entity_failure(
                source_id=source_id,
                dataset_code="INDEX_CLASSIFY",
                entity_key="GLOBAL",
                error=error,
            )
            return self._finish_stage(
                source_id=source_id,
                sync_run_id=run_id,
                sync_type=MARKET_FREE_INDEX_CLASSIFY_SYNC_TYPE,
                requested_date=None,
                fetched_count=0,
                write_stats=WriteStats(),
                failures=("GLOBAL",),
            )
        _report(progress_reporter, 1, 1, None, "指数分类同步完成")
        return self._finish_stage(
            source_id=source_id,
            sync_run_id=run_id,
            sync_type=MARKET_FREE_INDEX_CLASSIFY_SYNC_TYPE,
            requested_date=None,
            fetched_count=len(source_rows),
            write_stats=stats,
            failures=(),
        )

    def _sync_registered_reference_indices(
        self,
        *,
        source_id: UUID,
        parent_sync_run_id: UUID,
        as_of_date: date,
        progress_reporter: ProgressReporter | None,
    ) -> tuple[SyncOutcome, SyncOutcome]:
        with Session(self._engine) as session:
            index_codes = list_reference_benchmark_codes(session, source_id=source_id)
        daily_run_id = self._start_stage(
            source_id=source_id,
            parent_sync_run_id=parent_sync_run_id,
            sync_type=MARKET_FREE_INDEX_DAILY_SYNC_TYPE,
            start_date=_HISTORY_START_DATE,
            end_date=as_of_date,
        )
        weight_run_id = self._start_stage(
            source_id=source_id,
            parent_sync_run_id=parent_sync_run_id,
            sync_type=MARKET_FREE_INDEX_WEIGHT_SYNC_TYPE,
            start_date=_HISTORY_START_DATE,
            end_date=as_of_date,
        )
        daily_stats = WriteStats()
        weight_stats = WriteStats()
        daily_fetched_count = 0
        weight_fetched_count = 0
        daily_failures: list[str] = []
        weight_failures: list[str] = []
        for current, index_code in enumerate(index_codes, start=1):
            _report(progress_reporter, current, len(index_codes), index_code, "正在同步市场参考指数")
            try:
                daily_rows = self._client.list_index_daily(
                    index_code, start_date=_HISTORY_START_DATE, end_date=as_of_date
                )
                with Session(self._engine) as session, session.begin():
                    changed_count = upsert_benchmark_nav_points(
                        session,
                        benchmark_code=index_code,
                        points=tuple(
                            BenchmarkNavPoint(
                                nav_date=row.trade_date,
                                closing_value=row.close_price,
                                source_published_at=None,
                                row_hash=_stable_hash(
                                    {
                                        "index_code": row.index_code,
                                        "trade_date": row.trade_date,
                                        "close_price": row.close_price,
                                    }
                                ),
                            )
                            for row in daily_rows
                        ),
                    )
                    if daily_rows:
                        mark_cursor_success(
                            session,
                            source_id=source_id,
                            dataset_code="INDEX_DAILY",
                            entity_key=index_code,
                            last_successful_data_date=max(row.trade_date for row in daily_rows),
                            last_sync_run_id=daily_run_id,
                        )
                daily_stats = daily_stats.combine(WriteStats(updated_count=changed_count))
                daily_fetched_count += len(daily_rows)
            except (TushareIntegrationError, ValueError) as error:
                daily_failures.append(index_code)
                self._record_entity_failure(
                    source_id=source_id, dataset_code="INDEX_DAILY", entity_key=index_code, error=error
                )
            try:
                weight_rows = self._client.list_index_weights(
                    index_code, start_date=_HISTORY_START_DATE, end_date=as_of_date
                )
                records = tuple(_to_index_weight_upsert(row) for row in weight_rows)
                with Session(self._engine) as session, session.begin():
                    write_stats = upsert_index_weight_snapshots_batch(session, source_id=source_id, records=records)
                    if weight_rows:
                        mark_cursor_success(
                            session,
                            source_id=source_id,
                            dataset_code="INDEX_WEIGHT",
                            entity_key=index_code,
                            last_successful_data_date=max(row.trade_date for row in weight_rows),
                            last_sync_run_id=weight_run_id,
                        )
                weight_stats = weight_stats.combine(write_stats)
                weight_fetched_count += len(weight_rows)
            except (TushareIntegrationError, ValueError) as error:
                weight_failures.append(index_code)
                self._record_entity_failure(
                    source_id=source_id, dataset_code="INDEX_WEIGHT", entity_key=index_code, error=error
                )
        return (
            self._finish_stage(
                source_id=source_id,
                sync_run_id=daily_run_id,
                sync_type=MARKET_FREE_INDEX_DAILY_SYNC_TYPE,
                requested_date=as_of_date,
                fetched_count=daily_fetched_count,
                write_stats=daily_stats,
                failures=daily_failures,
            ),
            self._finish_stage(
                source_id=source_id,
                sync_run_id=weight_run_id,
                sync_type=MARKET_FREE_INDEX_WEIGHT_SYNC_TYPE,
                requested_date=as_of_date,
                fetched_count=weight_fetched_count,
                write_stats=weight_stats,
                failures=weight_failures,
            ),
        )

    def _start_stage(
        self,
        *,
        source_id: UUID,
        parent_sync_run_id: UUID,
        sync_type: str,
        start_date: date | None,
        end_date: date | None,
    ) -> UUID:
        with Session(self._engine) as session, session.begin():
            run = create_sync_run(
                session,
                source_id=source_id,
                sync_type=sync_type,
                requested_nav_date=end_date,
                parent_sync_run_id=parent_sync_run_id,
                requested_window_start=start_date,
                requested_window_end=end_date,
                data_as_of_date=end_date,
            )
            return run.sync_run_id

    def _finish_stage(
        self,
        *,
        source_id: UUID,
        sync_run_id: UUID,
        sync_type: str,
        requested_date: date | None,
        fetched_count: int,
        write_stats: WriteStats,
        failures: tuple[str, ...] | list[str],
    ) -> SyncOutcome:
        if failures:
            error_summary = f"failed entities={','.join(failures[:10])}; count={len(failures)}"
            with Session(self._engine) as session, session.begin():
                fail_sync_run(session, source_id=source_id, sync_run_id=sync_run_id, error_summary=error_summary)
            raise TushareIntegrationError(sync_type, error_summary)
        with Session(self._engine) as session, session.begin():
            complete_sync_run(
                session,
                source_id=source_id,
                sync_run_id=sync_run_id,
                fetched_count=fetched_count,
                write_stats=write_stats,
            )
        logger.info(
            "tushare_market_reference_sync._finish_stage >>> completed sync_type=%s run_id=%s fetched=%s "
            "created=%s updated=%s skipped=%s",
            sync_type,
            sync_run_id,
            fetched_count,
            write_stats.created_count,
            write_stats.updated_count,
            write_stats.skipped_count,
        )
        return SyncOutcome(
            sync_run_id=sync_run_id,
            sync_type=sync_type,
            requested_nav_date=requested_date,
            fetched_count=fetched_count,
            created_count=write_stats.created_count,
            updated_count=write_stats.updated_count,
            skipped_count=write_stats.skipped_count,
        )

    def _record_entity_failure(
        self, *, source_id: UUID, dataset_code: str, entity_key: str, error: Exception
    ) -> None:
        try:
            with Session(self._engine) as session, session.begin():
                mark_cursor_failure(
                    session,
                    source_id=source_id,
                    dataset_code=dataset_code,
                    entity_key=entity_key,
                    error_summary=f"{type(error).__name__}: {str(error)}",
                )
        except Exception:
            logger.exception(
                "tushare_market_reference_sync._record_entity_failure >>> failed to persist cursor, "
                "dataset_code=%s entity_key=%s",
                dataset_code,
                entity_key,
            )


def _parse_csv_setting(value: str) -> tuple[str, ...]:
    """将逗号分隔配置转换为去重、稳定排序的来源市场代码。"""
    parts = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    if not parts or len(set(parts)) != len(parts):
        raise ValueError("tushare_index_catalog_markets must contain unique values")
    return parts


def _to_exchange_daily_upsert(fund_code: str, row: TushareFundDaily) -> FundExchangeDailyUpsert:
    return FundExchangeDailyUpsert(
        fund_code=fund_code,
        trade_date=row.trade_date,
        open_price=row.open_price,
        high_price=row.high_price,
        low_price=row.low_price,
        close_price=row.close_price,
        previous_close_price=row.previous_close_price,
        change_value=row.change_value,
        change_percent=row.change_percent,
        volume=row.volume,
        amount=row.amount,
        content_hash=_stable_hash(
            {
                "fund_code": fund_code,
                "trade_date": row.trade_date,
                "previous_close_price": row.previous_close_price,
                "open_price": row.open_price,
                "high_price": row.high_price,
                "low_price": row.low_price,
                "close_price": row.close_price,
                "change_value": row.change_value,
                "change_percent": row.change_percent,
                "volume": row.volume,
                "amount": row.amount,
            }
        ),
    )


def _to_index_catalog_upsert(row: TushareIndexBasic) -> MarketIndexCatalogUpsert:
    return MarketIndexCatalogUpsert(
        index_code=row.index_code,
        display_name=row.display_name,
        market=row.market,
        publisher=row.publisher,
        category=row.category,
        base_date=row.base_date,
        list_date=row.list_date,
        expiry_date=row.expiry_date,
        row_hash=_stable_hash(row.__dict__),
    )


def _to_index_classification_upsert(row: TushareIndexClassification) -> MarketIndexClassificationUpsert:
    return MarketIndexClassificationUpsert(
        classification_code=row.classification_code,
        classification_name=row.classification_name,
        parent_classification_code=row.parent_classification_code,
        hierarchy_level=row.hierarchy_level,
        source_name=row.source_name,
        row_hash=_stable_hash(row.__dict__),
    )


def _to_index_weight_upsert(row: TushareIndexWeight) -> IndexWeightSnapshotUpsert:
    return IndexWeightSnapshotUpsert(
        index_code=row.index_code,
        weight_date=row.trade_date,
        constituent_code=row.constituent_code,
        weight=row.weight,
        row_hash=_stable_hash(row.__dict__),
    )


def _stable_hash(payload: object) -> str:
    """生成只包含业务字段的稳定 SHA-256 摘要。"""
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            default=_json_default,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _json_default(value: object) -> str:
    if isinstance(value, (date, Decimal)):
        return str(value)
    raise TypeError(f"unsupported hash value type={type(value)!r}")


def _report(
    progress_reporter: ProgressReporter | None,
    current: int,
    total: int,
    entity_key: str | None,
    message: str,
) -> None:
    if progress_reporter is not None:
        progress_reporter(current, total, entity_key, message)


@contextmanager
def _market_reference_sync_lock(engine: Engine) -> Iterator[None]:
    """使用 PostgreSQL 会话级咨询锁串行化管理端和维护命令的市场参考同步。"""
    with engine.connect() as connection:
        acquired = bool(connection.scalar(select(func.pg_try_advisory_lock(_MARKET_REFERENCE_LOCK_KEY))))
        if not acquired:
            raise MarketReferenceSyncInProgressError("market reference sync is already running")
        try:
            yield
        finally:
            released = bool(connection.scalar(select(func.pg_advisory_unlock(_MARKET_REFERENCE_LOCK_KEY))))
            if not released:
                logger.error(
                    "tushare_market_reference_sync._market_reference_sync_lock >>> advisory lock release failed"
                )
