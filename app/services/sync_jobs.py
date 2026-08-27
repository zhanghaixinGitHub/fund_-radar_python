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

from app.core.logging import get_logger
from app.integrations.tushare import TushareIntegrationError
from app.services.tushare_fund_sync import (
    FocusedNavIncrementalInProgressError,
    FocusedNavIncrementalPreconditionError,
    SyncOutcome,
    TushareFundSyncService,
)

logger = get_logger(__name__)

FOCUSED_NAV_INCREMENTAL_JOB_TYPE = "FOCUSED_NAV_INCREMENTAL"
_ACTIVE_STATUSES = frozenset({"QUEUED", "RUNNING"})


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
        self._latest_job_id: UUID | None = None
        self._active_job_id: UUID | None = None
        self._lock = Lock()
        self._closed = False

    def start_focused_nav_incremental(self, ts_codes: tuple[str, ...]) -> SyncJobSnapshot:
        """创建并提交重点基金增量任务；已有活动任务时返回受控冲突。"""
        if not ts_codes:
            raise ValueError("focused fund codes must not be empty")
        with self._lock:
            if self._closed:
                raise RuntimeError("sync job manager is stopped")
            if self._active_job_id is not None:
                active = self._jobs.get(self._active_job_id)
                if active is not None and active.status in _ACTIVE_STATUSES:
                    raise SyncJobInProgressError("a local sync job is already running")
            snapshot = SyncJobSnapshot(
                job_id=uuid4(),
                job_type=FOCUSED_NAV_INCREMENTAL_JOB_TYPE,
                status="QUEUED",
                requested_nav_date=date.today(),
                fund_codes=tuple(ts_codes),
                progress_current=0,
                progress_total=len(ts_codes) + 1,
                current_fund_code=None,
                progress_message="任务已创建，等待后台执行",
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
            self._latest_job_id = snapshot.job_id
            self._active_job_id = snapshot.job_id
            self._executor.submit(self._run_focused_nav_incremental, snapshot.job_id)
            return snapshot

    def get_job(self, job_id: UUID) -> SyncJobSnapshot | None:
        """按任务标识读取最新进度。"""
        with self._lock:
            return self._jobs.get(job_id)

    def get_latest_job(self) -> SyncJobSnapshot | None:
        """读取当前进程中最近一次创建的同步任务。"""
        with self._lock:
            return self._jobs.get(self._latest_job_id) if self._latest_job_id else None

    def close(self) -> None:
        """停止接受新任务；不阻断进行中的同步写库。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=False)

    def _run_focused_nav_incremental(self, job_id: UUID) -> None:
        service: TushareFundSyncService | None = None
        self._replace_job(job_id, status="RUNNING", started_at=datetime.now(UTC), progress_message="正在准备同步")
        try:
            snapshot = self._required_job(job_id)
            service = self._service_factory()
            outcome = service.sync_focused_nav_incremental(
                snapshot.fund_codes,
                progress_reporter=lambda current, total, fund_code, message: self._update_progress(
                    job_id, current, total, fund_code, message
                ),
            )
            self._complete_job(job_id, outcome)
            logger.info(
                "sync_jobs._run_focused_nav_incremental >>> completed job_id=%s, sync_run_id=%s, "
                "fetched=%s, created=%s, updated=%s",
                job_id,
                outcome.sync_run_id,
                outcome.fetched_count,
                outcome.created_count,
                outcome.updated_count,
            )
        except FocusedNavIncrementalPreconditionError:
            self._fail_job(job_id, "FOCUSED_SYNC_BASELINE_MISSING", "请先完成重点基金历史净值回填。")
        except FocusedNavIncrementalInProgressError:
            self._fail_job(job_id, "FOCUSED_SYNC_IN_PROGRESS", "已有重点基金同步正在执行，请稍后重试。")
        except TushareIntegrationError:
            self._fail_job(job_id, "FOCUSED_SYNC_FAILED", "净值同步未完成，请稍后重试。")
        except ValueError:
            self._fail_job(job_id, "FOCUSED_SYNC_UNAVAILABLE", "净值同步服务尚未完成配置。")
        except Exception:
            logger.exception("sync_jobs._run_focused_nav_incremental >>> unexpected task failure, job_id=%s", job_id)
            self._fail_job(job_id, "FOCUSED_SYNC_FAILED", "净值同步未完成，请稍后重试。")
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
