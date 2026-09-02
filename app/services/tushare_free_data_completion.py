"""当前 2000 积分免费数据补齐的总编排服务。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.session import get_engine
from app.repositories.fund_sync import (
    TUSHARE_AUTHORIZED_API_NAMES,
    WriteStats,
    complete_sync_run,
    create_sync_run,
    ensure_tushare_source,
    fail_sync_run,
    link_sync_run_to_parent,
)
from app.repositories.market_reference_sync import require_tushare_source_capabilities
from app.services.tushare_fund_sync import MarketDetailSyncResult, SyncOutcome, TushareFundSyncService
from app.services.tushare_market_reference_sync import MarketReferenceSyncService

logger = get_logger(__name__)

MARKET_FREE_DATA_COMPLETION_SYNC_TYPE = "MARKET_FREE_DATA_COMPLETION"
ProgressReporter = Callable[[int, int, str | None, str], None]
FundServiceFactory = Callable[[], TushareFundSyncService]
MarketReferenceServiceFactory = Callable[[], MarketReferenceSyncService]


@dataclass(frozen=True)
class FreeDataCompletionResult:
    """一次免费数据补齐的父运行与全部子阶段统计。"""

    overall_outcome: SyncOutcome
    outcomes: tuple[SyncOutcome, ...]


class TushareFreeDataCompletionService:
    """编排已有基金详情同步和新增场内/指数同步的单次管理员任务。

    所有真实调用由同步中心显式发起；该服务不被详情页或预测读取链路调用。
    """

    def __init__(
        self,
        *,
        fund_service_factory: FundServiceFactory = TushareFundSyncService,
        market_reference_service_factory: MarketReferenceServiceFactory = MarketReferenceSyncService,
    ) -> None:
        self._engine = get_engine()
        self._fund_service_factory = fund_service_factory
        self._market_reference_service_factory = market_reference_service_factory

    def close(self) -> None:
        """总编排服务不持有长连接；子服务在每次同步结束时自行关闭。"""
        return None

    def sync(
        self, *, as_of_date: date | None = None, progress_reporter: ProgressReporter | None = None
    ) -> FreeDataCompletionResult:
        """执行手动全量补齐并以父子运行记录所有成功或失败阶段。

        Raises:
            TushareIntegrationError: 外部来源、字段、一致性或某一阶段失败。
            ValueError: 来源未启用、能力未登记或本地配置无效。
        """
        target_date = as_of_date or date.today()
        with Session(self._engine) as session, session.begin():
            # 当前能力仅在最小验权成功后由实现方登记；来源仍可在后续被管理员禁用。
            source = ensure_tushare_source(session)
            require_tushare_source_capabilities(session, TUSHARE_AUTHORIZED_API_NAMES)
            parent_run = create_sync_run(
                session,
                source_id=source.source_id,
                sync_type=MARKET_FREE_DATA_COMPLETION_SYNC_TYPE,
                requested_nav_date=target_date,
                data_as_of_date=target_date,
            )
            parent_sync_run_id = parent_run.sync_run_id
            source_id = source.source_id

        fund_service: TushareFundSyncService | None = None
        market_reference_service: MarketReferenceSyncService | None = None
        try:
            _report(progress_reporter, 0, 2, None, "正在补齐基金档案、净值、经理、份额和分红")
            fund_service = self._fund_service_factory()
            fund_result = fund_service.sync_market_details(
                history_end_date=target_date,
                progress_reporter=progress_reporter,
            )
            self._link_market_detail_runs(parent_sync_run_id, fund_result)

            _report(progress_reporter, 1, 2, None, "正在补齐场内基金与市场参考指数数据")
            market_reference_service = self._market_reference_service_factory()
            market_result = market_reference_service.sync_all(
                parent_sync_run_id=parent_sync_run_id,
                as_of_date=target_date,
                progress_reporter=progress_reporter,
            )
            all_outcomes = fund_result.outcomes + market_result.outcomes
            all_stats = _combine_write_stats(all_outcomes)
            overall_outcome = SyncOutcome(
                sync_run_id=parent_sync_run_id,
                sync_type=MARKET_FREE_DATA_COMPLETION_SYNC_TYPE,
                requested_nav_date=target_date,
                fetched_count=sum(outcome.fetched_count for outcome in all_outcomes),
                created_count=all_stats.created_count,
                updated_count=all_stats.updated_count,
                skipped_count=all_stats.skipped_count,
            )
            with Session(self._engine) as session, session.begin():
                complete_sync_run(
                    session,
                    source_id=source_id,
                    sync_run_id=parent_sync_run_id,
                    fetched_count=overall_outcome.fetched_count,
                    write_stats=all_stats,
                )
            _report(progress_reporter, 2, 2, None, "当前免费数据补齐完成")
            logger.info(
                "tushare_free_data_completion.sync >>> completed parent_run_id=%s fetched=%s created=%s "
                "updated=%s skipped=%s",
                parent_sync_run_id,
                overall_outcome.fetched_count,
                overall_outcome.created_count,
                overall_outcome.updated_count,
                overall_outcome.skipped_count,
            )
            return FreeDataCompletionResult(overall_outcome=overall_outcome, outcomes=all_outcomes)
        except Exception as error:
            with Session(self._engine) as session, session.begin():
                fail_sync_run(
                    session,
                    source_id=source_id,
                    sync_run_id=parent_sync_run_id,
                    error_summary=f"{type(error).__name__}: {str(error)}",
                )
            logger.exception(
                "tushare_free_data_completion.sync >>> failed parent_run_id=%s", parent_sync_run_id
            )
            raise
        finally:
            if fund_service is not None:
                fund_service.close()
            if market_reference_service is not None:
                market_reference_service.close()

    def _link_market_detail_runs(self, parent_sync_run_id: UUID, result: MarketDetailSyncResult) -> None:
        """将既有完整资料父运行及五个阶段补挂到本次总补齐父运行。"""
        with Session(self._engine) as session, session.begin():
            link_sync_run_to_parent(
                session,
                sync_run_id=result.overall_outcome.sync_run_id,
                parent_sync_run_id=parent_sync_run_id,
            )
            for outcome in result.outcomes:
                link_sync_run_to_parent(
                    session,
                    sync_run_id=outcome.sync_run_id,
                    parent_sync_run_id=result.overall_outcome.sync_run_id,
                )


def _combine_write_stats(outcomes: tuple[SyncOutcome, ...]) -> WriteStats:
    """将来源阶段统计汇总成父运行的写入统计。"""
    stats = WriteStats()
    for outcome in outcomes:
        stats = stats.combine(
            WriteStats(
                created_count=outcome.created_count,
                updated_count=outcome.updated_count,
                skipped_count=outcome.skipped_count,
            )
        )
    return stats


def _report(
    progress_reporter: ProgressReporter | None,
    current: int,
    total: int,
    entity_key: str | None,
    message: str,
) -> None:
    if progress_reporter is not None:
        progress_reporter(current, total, entity_key, message)
