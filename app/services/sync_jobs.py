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
from app.repositories.market_reference_sync import SourceCapabilityError
from app.services.direction_1d_spx_manual import synchronize as synchronize_spx
from app.services.direction_1d_sync import Direction1dSyncService
from app.services.multi_prediction_sync import MultiPredictionSyncService
from app.services.simulation_fee_sync import SimulationFeeSyncService
from app.services.stock_feature_snapshot import (
    FeatureSnapshotBuildInProgressError,
    StockFeatureBuildSummary,
    StockFeatureSnapshotService,
)
from app.services.tushare_free_data_completion import (
    FreeDataCompletionResult,
    TushareFreeDataCompletionService,
)
from app.services.tushare_fund_sync import (
    MarketDetailSyncResult,
    MarketNavIncrementalInProgressError,
    MarketNavIncrementalPreconditionError,
    SyncOutcome,
    TushareFundSyncService,
)
from app.services.tushare_market_reference_sync import MarketReferenceSyncInProgressError

logger = get_logger(__name__)

MARKET_ALL_JOB_TYPE = "MARKET_ALL"
MARKET_NAV_INCREMENTAL_JOB_TYPE = "MARKET_NAV_INCREMENTAL"
MARKET_DETAIL_JOB_TYPE = "MARKET_DETAIL"
STOCK_FEATURE_SNAPSHOT_JOB_TYPE = "STOCK_FEATURE_SNAPSHOT"
MARKET_FREE_DATA_COMPLETION_JOB_TYPE = "MARKET_FREE_DATA_COMPLETION"
SPX_MANUAL_JOB_TYPE = "SPX_MANUAL"
SIMULATION_FEE_JOB_TYPE = "SIMULATION_FEES"
MULTI_PREDICTION_JOB_TYPE = "MULTI_PREDICTIONS"
DIRECTION_1D_JOB_TYPE = "DIRECTION_1D_PREDICTIONS"
_ALL_JOB_STAGES = (
    # SPX有早上08:00的观测边界，先取这一小份数据，避免被较长的全市场同步拖到截止后。
    (SPX_MANUAL_JOB_TYPE, "标普500"),
    # 资料更新内部已执行完整资料同步，批次不再单独重复抓取同一批资料。
    (MARKET_FREE_DATA_COMPLETION_JOB_TYPE, "基金资料与市场数据更新"),
    (MARKET_NAV_INCREMENTAL_JOB_TYPE, "净值增量"),
    (STOCK_FEATURE_SNAPSHOT_JOB_TYPE, "历史指标计算"),
    # 预测使用前面已同步净值；独立校验输入完整性，来源失败时不能默认算作预测成功。
    (SIMULATION_FEE_JOB_TYPE, "模拟费率"),
    (MULTI_PREDICTION_JOB_TYPE, "全部关注多周期预测、综合建议与到期核验"),
)
_ACTIVE_STATUSES = frozenset({"QUEUED", "RUNNING"})
_SYNC_TYPES_BY_JOB_TYPE = {
    MARKET_NAV_INCREMENTAL_JOB_TYPE: ("MARKET_NAV_INCREMENTAL",),
    MARKET_DETAIL_JOB_TYPE: ("MARKET_DETAIL",),
    MARKET_FREE_DATA_COMPLETION_JOB_TYPE: ("MARKET_FREE_DATA_COMPLETION",),
}


def _feature_completion_message(summary: StockFeatureBuildSummary) -> str:
    """返回可展示的特征构建结果摘要，不混淆为预测或回测结论。"""
    return (
        "历史指标计算完成："
        f"处理 {summary.attempted_fund_count} 只，新建 {summary.created_count}，"
        f"更新 {summary.updated_count}，未变化 {summary.skipped_count}"
    )


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
FeatureServiceFactory = Callable[[], StockFeatureSnapshotService]
FreeDataCompletionServiceFactory = Callable[[], TushareFreeDataCompletionService]


class LocalSyncJobManager:
    """单进程、单并发的同步任务管理器，适用于本机部署的手动任务。"""

    def __init__(
        self,
        service_factory: SyncServiceFactory = TushareFundSyncService,
        feature_service_factory: FeatureServiceFactory = StockFeatureSnapshotService,
        free_data_completion_service_factory: FreeDataCompletionServiceFactory = TushareFreeDataCompletionService,
        spx_synchronizer: Callable[[], dict] = synchronize_spx,
        fee_service_factory: Callable[[], SimulationFeeSyncService] = SimulationFeeSyncService,
        prediction_service_factory: Callable[[], Direction1dSyncService] = Direction1dSyncService,
        multi_prediction_service_factory=MultiPredictionSyncService,
    ) -> None:
        self._service_factory = service_factory
        self._feature_service_factory = feature_service_factory
        self._free_data_completion_service_factory = free_data_completion_service_factory
        self._spx_synchronizer = spx_synchronizer
        self._fee_service_factory = fee_service_factory
        self._prediction_service_factory = prediction_service_factory
        self._multi_prediction_service_factory = multi_prediction_service_factory
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fund-sync-job")
        self._jobs: dict[UUID, SyncJobSnapshot] = {}
        self._latest_job_ids: dict[str, UUID] = {}
        self._batch_child_ids: dict[UUID, tuple[UUID, ...]] = {}
        self._active_job_id: UUID | None = None
        self._lock = Lock()
        self._closed = False

    def start_market_nav_incremental(self) -> SyncJobSnapshot:
        """创建净值任务，单独执行时保留原有自动特征阶段。"""
        return self._start_job(MARKET_NAV_INCREMENTAL_JOB_TYPE, self._run_market_nav_incremental)

    def start_market_details(self) -> SyncJobSnapshot:
        """创建完整资料任务，与其他同步共用互斥保护。"""
        return self._start_job(MARKET_DETAIL_JOB_TYPE, self._run_market_details)

    def start_stock_feature_snapshots(self) -> SyncJobSnapshot:
        """创建已落库净值的独立特征快照任务。"""
        return self._start_job(STOCK_FEATURE_SNAPSHOT_JOB_TYPE, self._run_stock_feature_snapshots)

    def start_market_free_data_completion(self) -> SyncJobSnapshot:
        """创建管理员显式提交的已授权免费数据补齐任务。"""
        return self._start_job(MARKET_FREE_DATA_COMPLETION_JOB_TYPE, self._run_market_free_data_completion)

    def start_simulation_fees(self, fund_code: str | None = None) -> SyncJobSnapshot:
        """单只刷新和全量初始化都使用同一任务互斥与进度机制。"""
        return self._start_job(SIMULATION_FEE_JOB_TYPE, lambda job_id: self._run_simulation_fees(job_id, fund_code))

    def start_direction_1d_predictions(self) -> SyncJobSnapshot:
        """与一键同步共用后台队列，独立手动执行不要求任何个人实验开关。"""
        return self._start_job(DIRECTION_1D_JOB_TYPE, self._run_direction_1d_predictions)

    def start_multi_predictions(self) -> SyncJobSnapshot:
        """持久公共预测批次与个人综合建议按依赖顺序执行。"""
        return self._start_job(MULTI_PREDICTION_JOB_TYPE, self._run_multi_predictions)

    def start_all(self) -> SyncJobSnapshot:
        """原子登记六个子任务；完整资料由资料更新覆盖，批次不依赖浏览器存活。"""
        with self._lock:
            self._require_idle()
            parent = replace(
                self._new_job(MARKET_ALL_JOB_TYPE),
                progress_total=len(_ALL_JOB_STAGES),
                progress_message=f"一键同步已创建，等待依次执行 {len(_ALL_JOB_STAGES)} 类任务",
            )
            children = tuple(self._new_job(job_type) for job_type, _ in _ALL_JOB_STAGES)
            for snapshot in (parent, *children):
                self._jobs[snapshot.job_id] = snapshot
                self._latest_job_ids[snapshot.job_type] = snapshot.job_id
            self._batch_child_ids[parent.job_id] = tuple(child.job_id for child in children)
            self._active_job_id = parent.job_id
            self._executor.submit(self._run_all, parent.job_id)
            return parent

    def _require_idle(self) -> None:
        """调用方持有锁；直到实际执行与资源清理结束才允许下一次提交。"""
        if self._closed:
            raise RuntimeError("sync job manager is stopped")
        if self._active_job_id is not None:
            raise SyncJobInProgressError("a local sync job is already running")

    @staticmethod
    def _new_job(job_type: str) -> SyncJobSnapshot:
        return SyncJobSnapshot(
            job_id=uuid4(),
            job_type=job_type,
            status="QUEUED",
            requested_nav_date=date.today(),
            fund_codes=(),
            progress_current=0,
            progress_total=0,
            current_fund_code=None,
            progress_message="任务已创建，等待执行",
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

    def _start_job(self, job_type: str, runner: Callable[[UUID], None]) -> SyncJobSnapshot:
        with self._lock:
            self._require_idle()
            snapshot = self._new_job(job_type)
            self._jobs[snapshot.job_id] = snapshot
            self._latest_job_ids[job_type] = snapshot.job_id
            self._active_job_id = snapshot.job_id
            self._executor.submit(runner, snapshot.job_id)
            return snapshot

    def _run_all(self, job_id: UUID) -> None:
        """失败不掩盖成功事实；尝试每项一次，最后统一汇总结果。"""
        self._replace_job(job_id, status="RUNNING", started_at=datetime.now(UTC))
        runners = (
            self._run_spx_manual,
            self._run_market_free_data_completion,
            lambda child_id: self._run_market_nav_incremental(child_id, build_features=False),
            self._run_stock_feature_snapshots,
            self._run_simulation_fees,
            self._run_multi_predictions,
        )
        try:
            child_ids = self._batch_child_ids[job_id]
            for index, (child_id, runner) in enumerate(zip(child_ids, runners, strict=True)):
                logger.info("sync_jobs._run_all >>> stage started, job_id=%s, child_job_id=%s", job_id, child_id)
                try:
                    runner(child_id)
                except Exception:
                    logger.exception(
                        "sync_jobs._run_all >>> stage failed, job_id=%s, child_job_id=%s", job_id, child_id
                    )
                    self._fail_job(child_id, "SYNC_STAGE_FAILED", "本项同步未完整结束，请稍后单独重试。")
                self._replace_job(job_id, progress_current=index + 1)
            children = [self._required_job(child_id) for child_id in child_ids]
            failed_names = [
                title
                for child, (_, title) in zip(children, _ALL_JOB_STAGES, strict=True)
                if child.status != "SUCCEEDED"
            ]
            success_count = len(children) - len(failed_names)
            result_status = "SUCCEEDED" if not failed_names else "PARTIAL_SUCCESS" if success_count else "FAILED"
            message = f"一键同步已结束：成功 {success_count} 项，未完成 {len(failed_names)} 项"
            self._replace_job(
                job_id,
                status=result_status,
                progress_message=message,
                current_fund_code=None,
                error_code="SYNC_ALL_INCOMPLETE" if failed_names else None,
                error_message=("未完成：" + "、".join(failed_names) + "。请查看对应任务详情并单独重试。")
                if failed_names
                else None,
                finished_at=datetime.now(UTC),
            )
            logger.info("sync_jobs._run_all >>> completed, job_id=%s, status=%s", job_id, result_status)
        except Exception:
            logger.exception("sync_jobs._run_all >>> batch failed, job_id=%s", job_id)
            for child_id in self._batch_child_ids[job_id]:
                if self._required_job(child_id).status in _ACTIVE_STATUSES:
                    self._fail_job(child_id, "SYNC_ALL_INTERRUPTED", "一键同步中断，本项未完成，请单独重试。")
            self._fail_job(job_id, "SYNC_ALL_FAILED", "一键同步未完整结束，请检查各项任务状态。")
        finally:
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None

    def _snapshot(self, job_id: UUID | None) -> SyncJobSnapshot | None:
        """调用方持有锁；批次进度附带当前子任务的真实步骤，不估算完成时间。"""
        snapshot = self._jobs.get(job_id) if job_id else None
        if snapshot and snapshot.job_type == MARKET_ALL_JOB_TYPE and snapshot.status in _ACTIVE_STATUSES:
            for index, child_id in enumerate(self._batch_child_ids[job_id]):
                child = self._jobs[child_id]
                if child.status == "RUNNING":
                    return replace(
                        snapshot,
                        current_fund_code=child.current_fund_code,
                        progress_message=f"第 {index + 1}/{len(_ALL_JOB_STAGES)} 项 · {_ALL_JOB_STAGES[index][1]}："
                        f"{child.progress_message}（{child.progress_current}/{child.progress_total} 步）",
                    )
        return snapshot

    def get_job(self, job_id: UUID) -> SyncJobSnapshot | None:
        """按任务标识读取最新进度。"""
        with self._lock:
            return self._snapshot(job_id)

    def get_latest_job(self, job_type: str = MARKET_NAV_INCREMENTAL_JOB_TYPE) -> SyncJobSnapshot | None:
        """读取当前进程中指定类型最近一次创建的同步任务。"""
        with self._lock:
            job_id = self._latest_job_ids.get(job_type)
            return self._snapshot(job_id)

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

    def _run_market_nav_incremental(self, job_id: UUID, *, build_features: bool = True) -> None:
        service: TushareFundSyncService | None = None
        self._replace_job(job_id, status="RUNNING", started_at=datetime.now(UTC), progress_message="正在准备同步")
        try:
            service = self._service_factory()
            outcome = service.sync_market_nav_incremental(
                progress_reporter=lambda current, total, fund_code, message: self._update_progress(
                    job_id, current, total, fund_code, message
                ),
            )
            if outcome.status != "SUCCEEDED":
                # 来源部分完成不能被后面的指标异常盖掉；预测会独立检查每只基金的完整净值。
                self._complete_job(job_id, outcome)
                return
            if not build_features:
                self._complete_job(job_id, outcome, completion_message="净值增量同步完成，历史指标将在后续步骤计算")
                return
            self._record_source_outcome(job_id, outcome)
            try:
                feature_summary = self._build_feature_snapshots(job_id)
            except FeatureSnapshotBuildInProgressError:
                self._mark_feature_stage_partial(job_id, error_code="FEATURE_SYNC_IN_PROGRESS")
                return
            except Exception:
                logger.exception(
                    "sync_jobs._run_market_nav_incremental >>> feature stage failed after source sync, job_id=%s",
                    job_id,
                )
                self._mark_feature_stage_partial(job_id, error_code="FEATURE_SNAPSHOT_BUILD_FAILED")
                return
            if feature_summary.status != "COMPLETED":
                self._mark_feature_stage_partial(job_id, error_code="FEATURE_SOURCE_NOT_READY")
                return
            self._complete_job(
                job_id,
                outcome,
                completion_message=_feature_completion_message(feature_summary),
            )
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
            try:
                if service is not None:
                    service.close()
            finally:
                with self._lock:
                    if self._active_job_id == job_id:
                        self._active_job_id = None

    def _run_spx_manual(self, job_id: UUID) -> None:
        """与独立按钮共享真实时钟、文件锁和每日预算；旧成功记录不能冒充本批次新取数。"""
        self._replace_job(
            job_id,
            status="RUNNING",
            started_at=datetime.now(UTC),
            progress_total=1,
            progress_message="正在获取并保存标普500行情",
        )
        # synchronize返回安全状态；异常由批次外层记录为本项失败，后续四项仍继续。
        result = self._spx_synchronizer()
        attempt = result.get("lastAttempt") or {}
        if not result.get("performedNow"):
            self._fail_job(job_id, "SPX_NOT_EXECUTED", result.get("message") or "标普500本次未执行，请查看采集状态。")
            return
        if attempt.get("state") not in {"ON_TIME", "LATE", "REFERENCE_ONLY"}:
            self._fail_job(job_id, "SPX_SYNC_INCOMPLETE", attempt.get("message") or "标普500所需行情未完整取得。")
            return
        # 同步成功与早间输入合格分开：晚到、非交易日可算取数成功，资格由SPX详情回执说明。
        self._replace_job(
            job_id,
            status="SUCCEEDED",
            progress_current=1,
            fetched_count=attempt["rowCount"],
            finished_at=datetime.now(UTC),
            progress_message=attempt["message"],
        )

    def _run_stock_feature_snapshots(self, job_id: UUID) -> None:
        """手动重试特征构建；来源未就绪时不写入、不伪造成功状态。"""
        self._replace_job(job_id, status="RUNNING", started_at=datetime.now(UTC), progress_message="正在读取已同步净值")
        try:
            feature_summary = self._build_feature_snapshots(job_id)
            if feature_summary.status != "COMPLETED":
                self._fail_job(
                    job_id,
                    "FEATURE_SOURCE_NOT_READY",
                    "历史指标所需净值尚未就绪，请先完成净值同步后再重试。",
                )
                return
            self._complete_feature_job(job_id, feature_summary)
            logger.info(
                "sync_jobs._run_stock_feature_snapshots >>> completed job_id=%s, source_sync_run_id=%s, "
                "attempted=%s, created=%s, updated=%s, skipped=%s",
                job_id,
                feature_summary.source_sync_run_id,
                feature_summary.attempted_fund_count,
                feature_summary.created_count,
                feature_summary.updated_count,
                feature_summary.skipped_count,
            )
        except FeatureSnapshotBuildInProgressError:
            self._fail_job(job_id, "FEATURE_SYNC_IN_PROGRESS", "历史指标正在由其他任务计算，请稍后重试。")
        except Exception:
            logger.exception("sync_jobs._run_stock_feature_snapshots >>> unexpected task failure, job_id=%s", job_id)
            self._fail_job(job_id, "FEATURE_SNAPSHOT_BUILD_FAILED", "历史指标计算未完成，请稍后重试。")
        finally:
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
            try:
                if service is not None:
                    service.close()
            finally:
                with self._lock:
                    if self._active_job_id == job_id:
                        self._active_job_id = None

    def _run_market_free_data_completion(self, job_id: UUID) -> None:
        """执行一次已验权数据补齐，并将父运行汇总映射为同步中心状态。"""
        service: TushareFreeDataCompletionService | None = None
        self._replace_job(job_id, status="RUNNING", started_at=datetime.now(UTC), progress_message="正在校验数据源权限")
        try:
            service = self._free_data_completion_service_factory()
            result: FreeDataCompletionResult = service.sync(
                progress_reporter=lambda current, total, fund_code, message: self._update_progress(
                    job_id, current, total, fund_code, message
                )
            )
            self._complete_job(
                job_id,
                result.overall_outcome,
                completion_message="基金资料与市场数据更新完成",
            )
            logger.info(
                "sync_jobs._run_market_free_data_completion >>> completed job_id=%s, sync_run_id=%s, "
                "fetched=%s, created=%s, updated=%s",
                job_id,
                result.overall_outcome.sync_run_id,
                result.overall_outcome.fetched_count,
                result.overall_outcome.created_count,
                result.overall_outcome.updated_count,
            )
        except MarketReferenceSyncInProgressError:
            self._fail_job(job_id, "FREE_DATA_SYNC_IN_PROGRESS", "已有基金资料与市场数据更新任务正在执行，请稍后重试。")
        except SourceCapabilityError:
            self._fail_job(
                job_id,
                "FREE_DATA_SYNC_CAPABILITY_DENIED",
                "当前来源未完成接口授权核验，请先核对数据源能力登记。",
            )
        except TushareIntegrationError:
            self._fail_job(job_id, "FREE_DATA_SYNC_FAILED", "基金资料与市场数据更新未完成，请检查来源限额或稍后重试。")
        except ValueError:
            self._fail_job(job_id, "FREE_DATA_SYNC_UNAVAILABLE", "数据源权限或本地配置尚未完成校验。")
        except Exception:
            logger.exception(
                "sync_jobs._run_market_free_data_completion >>> unexpected task failure, job_id=%s",
                job_id,
            )
            self._fail_job(job_id, "FREE_DATA_SYNC_FAILED", "基金资料与市场数据更新未完成，请稍后重试。")
        finally:
            try:
                if service is not None:
                    service.close()
            finally:
                with self._lock:
                    if self._active_job_id == job_id:
                        self._active_job_id = None

    def _run_direction_1d_predictions(self, job_id: UUID) -> None:
        """只有 Java 确认留档才计为生成；部分不适用、缺数据、过期均作为未完成展示。"""
        service = None
        self._replace_job(
            job_id,
            status="RUNNING",
            started_at=datetime.now(UTC),
            progress_message="正在读取全部关注基金",
        )
        try:
            service = self._prediction_service_factory()
            result = service.sync(
                progress_reporter=lambda current, total, code, message: self._update_progress(
                    job_id,
                    current,
                    total,
                    code,
                    message,
                )
            )
            status = (
                "SUCCEEDED"
                if not result.issues
                else "PARTIAL_SUCCESS"
                if result.created + result.existing
                else "FAILED"
            )
            self._replace_job(
                job_id,
                status=status,
                requested_nav_date=result.target_date,
                progress_current=result.total,
                progress_total=result.total,
                current_fund_code=None,
                fetched_count=result.total,
                created_count=result.created,
                updated_count=result.existing,
                skipped_count=len(result.issues),
                progress_message=(
                    f"目标日 {result.target_date}：共检查 {result.total} 只，新生成 {result.created}，"
                    f"已有 {result.existing}，未生成 {len(result.issues)}"
                ),
                error_code="PREDICTION_INCOMPLETE" if result.issues else None,
                error_message="；".join(result.issues[:100]) if result.issues else None,
                finished_at=datetime.now(UTC),
            )
            logger.info(
                "sync_jobs._run_direction_1d_predictions >>> job_id=%s, status=%s, total=%s, created=%s, existing=%s",
                job_id,
                status,
                result.total,
                result.created,
                result.existing,
            )
        except Exception:
            logger.exception("sync_jobs._run_direction_1d_predictions >>> task failed, job_id=%s", job_id)
            self._fail_job(job_id, "PREDICTION_SYNC_FAILED", "预测任务未完成，请检查核心服务连接或稍后重试。")
        finally:
            try:
                if service is not None:
                    service.close()
            finally:
                with self._lock:
                    if self._active_job_id == job_id:
                        self._active_job_id = None

    def _run_multi_predictions(self, job_id: UUID) -> None:
        """合并一日及其他周期的真实回执，单位为基金周期项；窗口外也保留未生成原因。"""
        service = None
        self._replace_job(
            job_id,
            status="RUNNING",
            started_at=datetime.now(UTC),
            progress_message="正在读取全部关注基金与已开放周期",
        )
        try:
            service = self._multi_prediction_service_factory()
            result = service.sync(
                progress_reporter=lambda current, total, code, message: self._update_progress(
                    job_id,
                    current,
                    total,
                    code,
                    message,
                )
            )
            status = (
                "SUCCEEDED"
                if not result.issues
                else "PARTIAL_SUCCESS"
                if result.created + result.existing
                else "FAILED"
            )
            self._replace_job(
                job_id,
                status=status,
                requested_nav_date=result.target_date,
                progress_current=result.total,
                progress_total=result.total,
                current_fund_code=None,
                fetched_count=result.total,
                created_count=result.created,
                updated_count=result.existing,
                skipped_count=len(result.issues),
                progress_message=(
                    f"一日、五日、二十日和半年共处理 {result.total} 个基金周期项，新生成 {result.created}，"
                    f"已有 {result.existing}，未生成 {len(result.issues)}"
                ),
                error_code="PREDICTION_INCOMPLETE" if result.issues else None,
                error_message="；".join(result.issues[:100]) if result.issues else None,
                finished_at=datetime.now(UTC),
            )
            logger.info(
                "sync_jobs._run_multi_predictions >>> job_id=%s, status=%s, total=%s, created=%s, existing=%s",
                job_id,
                status,
                result.total,
                result.created,
                result.existing,
            )
        except Exception:
            logger.exception("sync_jobs._run_multi_predictions >>> task failed, job_id=%s", job_id)
            self._fail_job(job_id, "PREDICTION_SYNC_FAILED", "预测任务未完成，请检查核心服务连接或稍后重试。")
        finally:
            try:
                if service is not None:
                    service.close()
            finally:
                with self._lock:
                    if self._active_job_id == job_id:
                        self._active_job_id = None

    def _run_simulation_fees(self, job_id: UUID, fund_code: str | None = None) -> None:
        """按实际保存回执累计进度；部分失败保留成功计数，整批仍继续汇总。"""
        service = None
        self._replace_job(
            job_id,
            status="RUNNING",
            started_at=datetime.now(UTC),
            fund_codes=(fund_code,) if fund_code else (),
            progress_message="正在读取费率同步范围",
        )
        try:
            service = self._fee_service_factory()
            result = service.sync(
                fund_code,
                progress_reporter=lambda current, total, code, message: self._update_progress(
                    job_id, current, total, code, message
                ),
            )
            status = "SUCCEEDED" if not result.failures else "PARTIAL_SUCCESS" if result.updated else "FAILED"
            self._replace_job(
                job_id,
                status=status,
                progress_current=result.total,
                progress_total=result.total,
                current_fund_code=None,
                fetched_count=result.total,
                updated_count=result.updated,
                progress_message=(
                    f"费率同步结束：共 {result.total} 只，成功 {result.updated} 只，失败 {len(result.failures)} 只"
                ),
                error_code="SIM_FEE_SYNC_INCOMPLETE" if result.failures else None,
                error_message="；".join(result.failures) if result.failures else None,
                finished_at=datetime.now(UTC),
            )
            logger.info(
                "sync_jobs._run_simulation_fees >>> completed, job_id=%s, status=%s, total=%s, saved=%s, failed=%s",
                job_id,
                status,
                result.total,
                result.updated,
                len(result.failures),
            )
        except Exception:
            logger.exception("sync_jobs._run_simulation_fees >>> task failed, job_id=%s", job_id)
            self._fail_job(job_id, "SIM_FEE_SYNC_FAILED", "费率同步未完成，请检查核心服务连接或稍后重试。")
        finally:
            try:
                if service is not None:
                    service.close()
            finally:
                with self._lock:
                    if self._active_job_id == job_id:
                        self._active_job_id = None

    def _update_progress(self, job_id: UUID, current: int, total: int, fund_code: str | None, message: str) -> None:
        self._replace_job(
            job_id,
            progress_current=current,
            progress_total=total,
            current_fund_code=fund_code,
            progress_message=message,
        )

    def _build_feature_snapshots(self, job_id: UUID) -> StockFeatureBuildSummary:
        """以当前成功来源为唯一输入构建特征，并将逐基金进度映射到同步快照。"""
        return self._feature_service_factory().build(
            progress_reporter=lambda current, total, fund_code, message: self._update_progress(
                job_id, current, total, fund_code, message
            )
        )

    def _record_source_outcome(self, job_id: UUID, outcome: SyncOutcome) -> None:
        """保留来源同步事实，但在特征阶段完成前不结束父任务。"""
        self._replace_job(
            job_id,
            sync_run_id=outcome.sync_run_id,
            fetched_count=outcome.fetched_count,
            created_count=outcome.created_count,
            updated_count=outcome.updated_count,
            skipped_count=outcome.skipped_count,
            progress_message="基金市场净值同步完成，正在计算历史指标",
        )

    def _complete_job(self, job_id: UUID, outcome: SyncOutcome, *, completion_message: str = "同步完成") -> None:
        snapshot = self._required_job(job_id)
        self._replace_job(
            job_id,
            status=outcome.status,
            progress_current=snapshot.progress_total,
            current_fund_code=None,
            progress_message=completion_message
            if not outcome.issues
            else "净值同步尚有未完成项，成功数据已保存；待处理项目将按状态继续补拉",
            error_code="NAV_SYNC_INCOMPLETE" if outcome.issues else None,
            error_message="；".join(outcome.issues[:100]) or None,
            sync_run_id=outcome.sync_run_id,
            fetched_count=outcome.fetched_count,
            created_count=outcome.created_count,
            updated_count=outcome.updated_count,
            skipped_count=outcome.skipped_count,
            finished_at=datetime.now(UTC),
        )

    def _complete_feature_job(self, job_id: UUID, summary: StockFeatureBuildSummary) -> None:
        """完成独立特征任务，统计字段仅表达特征构建结果。"""
        snapshot = self._required_job(job_id)
        self._replace_job(
            job_id,
            status="SUCCEEDED",
            progress_current=snapshot.progress_total,
            current_fund_code=None,
            progress_message=_feature_completion_message(summary),
            sync_run_id=summary.source_sync_run_id,
            fetched_count=summary.attempted_fund_count,
            created_count=summary.created_count,
            updated_count=summary.updated_count,
            skipped_count=summary.skipped_count,
            finished_at=datetime.now(UTC),
        )

    def _mark_feature_stage_partial(self, job_id: UUID, *, error_code: str) -> None:
        """来源同步已成功但特征未能生成时，保留来源统计并提供独立重试入口。"""
        self._replace_job(
            job_id,
            status="PARTIAL_SUCCESS",
            current_fund_code=None,
            progress_message="基金市场净值同步完成，历史指标尚未计算完成",
            error_code=error_code,
            error_message="基金市场净值已同步，但历史指标尚未计算完成，可在同步中心的“历史指标计算”中重试。",
            finished_at=datetime.now(UTC),
        )
        logger.warning("sync_jobs._mark_feature_stage_partial >>> job_id=%s, code=%s", job_id, error_code)

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
