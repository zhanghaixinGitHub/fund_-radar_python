"""本机数据同步任务中心。

手动同步不依赖 Celery Worker：该模块只维护一个受控的后台线程，并向 Java 提供
安全的任务进度摘要。真实净值拉取和写库仍由 ``TushareFundSyncService`` 完成。
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from threading import Lock
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.session import get_engine
from app.integrations.tushare import TushareIntegrationError
from app.repositories.fund_sync import get_latest_successful_sync_time
from app.services.tushare_fund_sync import (
    MarketDetailSyncResult,
    MarketNavIncrementalInProgressError,
    MarketNavIncrementalPreconditionError,
    SyncOutcome,
    TushareFundSyncService,
)

logger = get_logger(__name__)

MARKET_NAV_INCREMENTAL_JOB_TYPE = "MARKET_NAV_INCREMENTAL"
MARKET_DETAIL_JOB_TYPE = "MARKET_DETAIL"
_ACTIVE_STATUSES = frozenset({"QUEUED", "RUNNING"})
_SYNC_TYPES_BY_JOB_TYPE = {
    MARKET_NAV_INCREMENTAL_JOB_TYPE: ("MARKET_NAV_INCREMENTAL",),
    MARKET_DETAIL_JOB_TYPE: ("MARKET_DETAIL",),
}


class SyncJobInProgressError(RuntimeError):
    """同步中心中已有运行中的任务时拒绝重复提交。"""


@dataclass(frozen=True)
class SyncJobSnapshot:
    """可安全返回给 Java 的本机同步任务摘要，不保存外部原始响应。"""

    job_id: UUID
    job_type: str
    status: str
    requested_nav_date: date
    fund_codes: tuple[str, ...]
    progress_current: int
    progress_total: int
    current_fund_code: str | None
    progress_message: str
    sync_run_id: UUID | None
    fetched_count: int
    created_count: int
    updated_count: int
    skipped_count: int
    error_code: str | None
    error_message: str | None
    started_at: datetime | None
    finished_at: datetime | None


SyncServiceFactory = Callable[[], TushareFundSyncService]


class LocalSyncJobManager:
    """单进程、单并发的同步任务管理器，适用于本机部署的手动任务。"""

    def __init__(self, service_factory: SyncServiceFactory = TushareFundSyncService) -> None:
        self._service_factory = service_factory
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fund-sync-job")
        self._jobs: dict[UUID, SyncJobSnapshot] = {}
        self._latest_job_ids: dict[str, UUID] = {}
        self._active_job_id: UUID | None = None
        self._lock = Lock()
        self._closed = False

    def start_market_nav_incremental(self) -> SyncJobSnapshot:
        """创建并提交基金市场增量任务；已有活动任务时返回受控冲突。"""
        with self._lock:
            if self._closed:
                raise RuntimeError("sync job manager is stopped")
            if self._active_job_id is not None:
                active = self._jobs.get(self._active_job_id)
                if active is not None and active.status in _ACTIVE_STATUSES:
                    raise SyncJobInProgressError("a local sync job is already running")
            snapshot = SyncJobSnapshot(
                job_id=uuid4(),
                job_type=MARKET_NAV_INCREMENTAL_JOB_TYPE,
                status="QUEUED",
                requested_nav_date=date.today(),
                fund_codes=(),
                progress_current=0,
                progress_total=0,
                current_fund_code=None,
                progress_message="任务已创建，等待读取基金市场范围",
                sync_run_id=None,
                fetched_count=0,
                created_count=0,
                updated_count=0,
                skipped_count=0,
                error_code=None,
                error_message=None,
                started_at=None,
                finished_at=None,
            )
            self._jobs[snapshot.job_id] = snapshot
            self._latest_job_ids[MARKET_NAV_INCREMENTAL_JOB_TYPE] = snapshot.job_id
            self._active_job_id = snapshot.job_id
            self._executor.submit(self._run_market_nav_incremental, snapshot.job_id)
            return snapshot

    def start_market_details(self) -> SyncJobSnapshot:
        """创建完整资料基线同步任务；与净值任务共用单并发保护。"""
        with self._lock:
            if self._closed:
                raise RuntimeError("sync job manager is stopped")
            if self._active_job_id is not None:
                active = self._jobs.get(self._active_job_id)
                if active is not None and active.status in _ACTIVE_STATUSES:
                    raise SyncJobInProgressError("a local sync job is already running")
            snapshot = SyncJobSnapshot(
                job_id=uuid4(),
                job_type=MARKET_DETAIL_JOB_TYPE,
                status="QUEUED",
                requested_nav_date=date.today(),
                fund_codes=(),
                progress_current=0,
                progress_total=0,
                current_fund_code=None,
                progress_message="任务已创建，等待读取基金市场范围",
                sync_run_id=None,
                fetched_count=0,
                created_count=0,
                updated_count=0,
                skipped_count=0,
                error_code=None,
                error_message=None,
                started_at=None,
                finished_at=None,
            )
            self._jobs[snapshot.job_id] = snapshot
            self._latest_job_ids[MARKET_DETAIL_JOB_TYPE] = snapshot.job_id
            self._active_job_id = snapshot.job_id
            self._executor.submit(self._run_market_details, snapshot.job_id)
            return snapshot

    def get_job(self, job_id: UUID) -> SyncJobSnapshot | None:
        """按任务标识读取最新进度。"""
        with self._lock:
            return self._jobs.get(job_id)

    def get_latest_job(self, job_type: str = MARKET_NAV_INCREMENTAL_JOB_TYPE) -> SyncJobSnapshot | None:
        """读取当前进程中指定类型最近一次创建的同步任务。"""
        with self._lock:
            job_id = self._latest_job_ids.get(job_type)
            return self._jobs.get(job_id) if job_id else None

    def close(self) -> None:
        """停止接受新任务；不阻断进行中的同步写库。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=False)

    def get_last_successful_time(self, job_type: str) -> datetime | None:
        """从持久化运行记录读取指定任务最近一次完整成功时间。"""
        sync_types = _SYNC_TYPES_BY_JOB_TYPE.get(job_type)
        if sync_types is None:
            raise ValueError("unsupported sync job type")
        with Session(get_engine()) as session:
            return get_latest_successful_sync_time(session, sync_types=sync_types)

    def _run_market_nav_incremental(self, job_id: UUID) -> None:
        service: TushareFundSyncService | None = None
        self._replace_job(job_id, status="RUNNING", started_at=datetime.now(UTC), progress_message="正在准备同步")
        try:
            service = self._service_factory()
            outcome = service.sync_market_nav_incremental(
                progress_reporter=lambda current, total, fund_code, message: self._update_progress(
                    job_id, current, total, fund_code, message
                ),
            )
            self._complete_job(job_id, outcome)
            logger.info(
                "sync_jobs._run_market_nav_incremental >>> completed job_id=%s, sync_run_id=%s, "
                "fetched=%s, created=%s, updated=%s",
                job_id,
                outcome.sync_run_id,
                outcome.fetched_count,
                outcome.created_count,
                outcome.updated_count,
            )
        except MarketNavIncrementalPreconditionError:
            self._fail_job(job_id, "MARKET_SYNC_BASELINE_MISSING", "请先完成基金市场历史净值回填或来源代码校验。")
        except MarketNavIncrementalInProgressError:
            self._fail_job(job_id, "MARKET_SYNC_IN_PROGRESS", "已有基金市场同步正在执行，请稍后重试。")
        except TushareIntegrationError:
            self._fail_job(job_id, "MARKET_SYNC_FAILED", "基金市场净值同步未完成，请稍后重试。")
        except ValueError:
            self._fail_job(job_id, "MARKET_SYNC_UNAVAILABLE", "基金市场同步服务尚未完成配置。")
        except Exception:
            logger.exception("sync_jobs._run_market_nav_incremental >>> unexpected task failure, job_id=%s", job_id)
            self._fail_job(job_id, "MARKET_SYNC_FAILED", "基金市场净值同步未完成，请稍后重试。")
        finally:
            if service is not None:
                service.close()
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None

    def _run_market_details(self, job_id: UUID) -> None:
        """执行完整资料同步并将五类来源进度映射为安全任务快照。"""
        service: TushareFundSyncService | None = None
        self._replace_job(job_id, status="RUNNING", started_at=datetime.now(UTC), progress_message="正在准备同步")
        try:
            service = self._service_factory()
            result: MarketDetailSyncResult = service.sync_market_details(
                progress_reporter=lambda current, total, fund_code, message: self._update_progress(
                    job_id, current, total, fund_code, message
                )
            )
            self._complete_job(job_id, result.overall_outcome)
            logger.info(
                "sync_jobs._run_market_details >>> completed job_id=%s, sync_run_id=%s, "
                "fetched=%s, created=%s, updated=%s",
                job_id,
                result.overall_outcome.sync_run_id,
                result.overall_outcome.fetched_count,
                result.overall_outcome.created_count,
                result.overall_outcome.updated_count,
            )
        except MarketNavIncrementalPreconditionError:
            self._fail_job(job_id, "MARKET_DETAIL_BASELINE_MISSING", "请先完成基金市场目录和历史净值回填。")
        except MarketNavIncrementalInProgressError:
            self._fail_job(job_id, "MARKET_SYNC_IN_PROGRESS", "已有基金市场同步正在执行，请稍后重试。")
        except TushareIntegrationError:
            self._fail_job(job_id, "MARKET_DETAIL_SYNC_FAILED", "基金完整资料同步未完成，请稍后重试。")
        except ValueError:
            self._fail_job(job_id, "MARKET_DETAIL_SYNC_UNAVAILABLE", "基金完整资料同步服务尚未完成配置。")
        except Exception:
            logger.exception("sync_jobs._run_market_details >>> unexpected task failure, job_id=%s", job_id)
            self._fail_job(job_id, "MARKET_DETAIL_SYNC_FAILED", "基金完整资料同步未完成，请稍后重试。")
        finally:
            if service is not None:
                service.close()
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None

    def _update_progress(
        self, job_id: UUID, current: int, total: int, fund_code: str | None, message: str
    ) -> None:
        self._replace_job(
            job_id,
            progress_current=current,
            progress_total=total,
            current_fund_code=fund_code,
            progress_message=message,
        )

    def _complete_job(self, job_id: UUID, outcome: SyncOutcome) -> None:
        snapshot = self._required_job(job_id)
        self._replace_job(
            job_id,
            status="SUCCEEDED",
            progress_current=snapshot.progress_total,
            current_fund_code=None,
            progress_message="同步完成",
            sync_run_id=outcome.sync_run_id,
            fetched_count=outcome.fetched_count,
            created_count=outcome.created_count,
            updated_count=outcome.updated_count,
            skipped_count=outcome.skipped_count,
            finished_at=datetime.now(UTC),
        )

    def _fail_job(self, job_id: UUID, error_code: str, error_message: str) -> None:
        self._replace_job(
            job_id,
            status="FAILED",
            current_fund_code=None,
            progress_message="同步未完成",
            error_code=error_code,
            error_message=error_message,
            finished_at=datetime.now(UTC),
        )
        logger.warning("sync_jobs._fail_job >>> job_id=%s, code=%s", job_id, error_code)

    def _required_job(self, job_id: UUID) -> SyncJobSnapshot:
        snapshot = self.get_job(job_id)
        if snapshot is None:
            raise LookupError(f"sync job does not exist: {job_id}")
        return snapshot

    def _replace_job(self, job_id: UUID, **changes: object) -> None:
        with self._lock:
            snapshot = self._jobs.get(job_id)
            if snapshot is not None:
                self._jobs[job_id] = replace(snapshot, **changes)


_manager_lock = Lock()
_manager: LocalSyncJobManager | None = None


def get_sync_job_manager() -> LocalSyncJobManager:
    """返回当前 FastAPI 进程的同步任务中心单例。"""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = LocalSyncJobManager()
        return _manager


def close_sync_job_manager() -> None:
    """应用关闭时释放后台线程资源。"""
    global _manager
    with _manager_lock:
        manager = _manager
        _manager = None
    if manager is not None:
        manager.close()
