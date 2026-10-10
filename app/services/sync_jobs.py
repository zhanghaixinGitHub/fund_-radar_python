"""本机数据同步任务中心。

手动同步不依赖 Celery Worker：该模块只维护一个受控的后台线程，并向 Java 提供
安全的任务进度摘要。真实净值拉取和写库仍由 ``TushareFundSyncService`` 完成。
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from threading import Lock
from time import monotonic
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
from app.services.prediction_sync_status import pending_daily, reconcile
from app.services.prediction_sync_status import summary as prediction_summary
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
FUND_MATERIALS_JOB_TYPE = "FUND_MATERIALS"
FUND_NEWS_JOB_TYPE = "FUND_NEWS"
FUND_INPUTS_JOB_TYPE = "FUND_INPUTS"
_ALL_JOB_STAGES = (
    # SPX有早上08:00的观测边界，先取这一小份数据，避免被较长的全市场同步拖到截止后。
    (SPX_MANUAL_JOB_TYPE, "标普500"),
    # 先更新日常最关心的净值；批次内不在此重复计算指标，等基础资料完成后统一计算一次。
    (MARKET_NAV_INCREMENTAL_JOB_TYPE, "净值增量"),
    # 预测读取分红等基础资料，因此这一步仍须在预测之前；内部已覆盖完整资料同步。
    (MARKET_FREE_DATA_COMPLETION_JOB_TYPE, "基金资料与市场数据更新"),
    (STOCK_FEATURE_SNAPSHOT_JOB_TYPE, "历史指标计算"),
    # 预测使用前面已同步净值；独立校验输入完整性，来源失败时不能默认算作预测成功。
    (SIMULATION_FEE_JOB_TYPE, "模拟费率"),
    # 002112综合分析已读取公告、持仓行情和公司资料，必须先更新输入再生成本次预测。
    (FUND_NEWS_JOB_TYPE, "近期基金公告核验"),
    (FUND_MATERIALS_JOB_TYPE, "基金持仓与公司资料更新"),
    # 等报告、行情和基金净值更新后再留存；仍检查真实取得时间，不能因后置而放宽截止规则。
    (FUND_INPUTS_JOB_TYPE, "002112 分析资料更新与留存"),
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
        ("历史指标部分未完成：" if summary.issues else "历史指标计算完成：")
        + f"处理 {summary.attempted_fund_count} 只，新建 {summary.created_count}，"
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
    # 可选业务摘要；旧持久记录缺失时按未知处理，不推断为 0 或成功。
    result_summary: dict | None = None


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
        materials_service_factory=None,
        state_path: Path | None = None,
        news_synchronizer=None,
        input_synchronizer=None,
    ) -> None:
        self._service_factory = service_factory
        self._feature_service_factory = feature_service_factory
        self._free_data_completion_service_factory = free_data_completion_service_factory
        self._spx_synchronizer = spx_synchronizer
        self._fee_service_factory = fee_service_factory
        self._prediction_service_factory = prediction_service_factory
        self._multi_prediction_service_factory = multi_prediction_service_factory
        self._materials_service_factory = materials_service_factory
        self._news_synchronizer = news_synchronizer
        self._input_synchronizer = input_synchronizer
        self._state_path = state_path
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fund-sync-job")
        # 原回执核对按查询请求排队，不设定时循环；独立线程避免慢响应阻塞同步状态接口。
        self._status_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prediction-status")
        self._jobs: dict[UUID, SyncJobSnapshot] = {}
        self._latest_job_ids: dict[str, UUID] = {}
        self._batch_child_ids: dict[UUID, tuple[UUID, ...]] = {}
        self._active_job_id: UUID | None = None
        self._lock = Lock()
        self._closed = False
        # 只由状态查询触发原任务核对，不启动采集/预测定时循环；并发查询共享五秒节流。
        self._refreshing: set[UUID] = set()
        self._refresh_after: dict[UUID, float] = {}
        self._refresh_offset: dict[UUID, int] = {}
        self._restore()

    def _persist(self):
        """调用方持有队列锁；生产单例保存任务树，测试可使用独立临时目录。"""
        if self._state_path:
            from app.services.fund_exposure_common import save

            save(
                self._state_path,
                {
                    "jobs": [asdict(j) for j in self._jobs.values()],
                    "latest": self._latest_job_ids,
                    "children": {str(k): v for k, v in self._batch_child_ids.items()},
                },
                replace=True,
            )

    def _restore(self):
        """重启不把运行中任务冒充成功；保留断点提示，重试由采集器复用完成成果。"""
        if not self._state_path or not self._state_path.exists():
            return
        from app.services.fund_exposure_common import read

        saved = read(self._state_path)
        for raw in saved["jobs"]:
            raw["job_id"] = UUID(raw["job_id"])
            raw["sync_run_id"] = UUID(raw["sync_run_id"]) if raw["sync_run_id"] else None
            raw["requested_nav_date"] = date.fromisoformat(raw["requested_nav_date"])
            raw["fund_codes"] = tuple(raw["fund_codes"])
            for field in ("started_at", "finished_at"):
                raw[field] = datetime.fromisoformat(raw[field]) if raw[field] else None
            if raw["status"] in _ACTIVE_STATUSES:
                raw.update(
                    status="FAILED",
                    error_code="SYNC_INTERRUPTED",
                    finished_at=datetime.now(UTC),
                    error_message="服务重启导致任务中断，已保存成果可在重试时复用。",
                    progress_message="上次任务中断，请重试未完成项目",
                )
            self._jobs[raw["job_id"]] = SyncJobSnapshot(**raw)
        self._latest_job_ids = {k: UUID(v) for k, v in saved["latest"].items()}
        self._batch_child_ids = {UUID(k): tuple(UUID(v) for v in values) for k, values in saved["children"].items()}
        self._persist()

    def start_fund_materials(self, fund_code: str) -> SyncJobSnapshot:
        """基金范围必须明确；空白和其他基金绝不退化为全市场任务。"""
        if fund_code != "002112":
            raise ValueError("当前仅支持 002112 的持仓与公司资料更新。")
        return self._start_job(FUND_MATERIALS_JOB_TYPE, self._run_fund_materials)

    def start_fund_news(self, fund_code: str) -> SyncJobSnapshot:
        """明确单基金范围，复用任务中心互斥队列；不与历史研究共用运行进度。"""
        if fund_code != "002112":
            raise ValueError("仅支持明确指定 002112。")
        return self._start_job(FUND_NEWS_JOB_TYPE, self._run_fund_news)

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
        """原子登记全部子任务；完整资料由资料更新覆盖，批次不依赖浏览器存活。"""
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
            self._persist()
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
            self._persist()
            self._executor.submit(runner, snapshot.job_id)
            return snapshot

    def _run_all(self, job_id: UUID) -> None:
        """失败不掩盖成功事实；尝试每项一次，最后统一汇总结果。"""
        self._replace_job(job_id, status="RUNNING", started_at=datetime.now(UTC))
        runners = (
            self._run_spx_manual,
            lambda child_id: self._run_market_nav_incremental(child_id, build_features=False),
            self._run_market_free_data_completion,
            self._run_stock_feature_snapshots,
            self._run_simulation_fees,
            self._run_fund_news,
            self._run_fund_materials,
            self._run_fund_inputs,
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
                error_message=(
                    "未完成：" + "、".join(failed_names) + "。请检查结果后重试；分析资料留存可重新执行一键同步。"
                )
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
        """按任务标识读取进度，并续查已提交的异步预测；不创建新预测或重新采集。"""
        self._refresh_predictions(job_id)
        with self._lock:
            return self._snapshot(job_id)

    def get_latest_job(self, job_type: str = MARKET_NAV_INCREMENTAL_JOB_TYPE) -> SyncJobSnapshot | None:
        """读取当前进程中指定类型最近一次创建的同步任务。"""
        with self._lock:
            job_id = self._latest_job_ids.get(job_type)
        return self.get_job(job_id) if job_id else None

    def _refresh_predictions(self, job_id: UUID) -> None:
        """网络调用在锁外执行；终态回填原批次，不能拿后来一次同步的成功覆盖旧批次。"""
        with self._lock:
            snapshot = self._jobs.get(job_id)
            child_ids = self._batch_child_ids.get(job_id, ())
        if not snapshot or snapshot.status in _ACTIVE_STATUSES:
            return
        if child_ids:
            for child_id in child_ids:
                self._refresh_predictions(child_id)
            self._finish_batch(job_id)
            return
        if snapshot.job_type not in {MULTI_PREDICTION_JOB_TYPE, DIRECTION_1D_JOB_TYPE}:
            return
        if not any(pending_daily(i) for i in (snapshot.result_summary or {}).get("items", [])):
            return
        with self._lock:
            if (
                self._closed
                or len(self._refreshing) >= 8
                or job_id in self._refreshing
                or monotonic() < self._refresh_after.get(job_id, 0)
            ):
                return
            self._refreshing.add(job_id)
            self._refresh_after[job_id] = monotonic() + 5
            offset = self._refresh_offset.get(job_id, 0)
            self._refresh_offset[job_id] = offset + 4
            self._status_executor.submit(self._reconcile_prediction_snapshot, snapshot, offset)

    def _reconcile_prediction_snapshot(self, snapshot: SyncJobSnapshot, offset: int) -> None:
        """按查询需求核对一次回执；慢网络不阻塞状态读取，完成后同步更新所属父批次。"""
        job_id = snapshot.job_id
        service = None
        try:
            service = self._prediction_service_factory()
            changes = reconcile(snapshot, service, offset=offset, recover_legacy=self._state_path is not None)
            if changes:
                self._replace_job(job_id, **changes)
        except Exception:
            logger.exception("sync_jobs._refresh_predictions >>> 原任务状态核对暂未完成, job_id=%s", job_id)
        finally:
            try:
                if service is not None:
                    service.close()
            finally:
                with self._lock:
                    self._refreshing.discard(job_id)
                    parents = [
                        parent
                        for parent, children in self._batch_child_ids.items()
                        if job_id in children and self._jobs[parent].status not in _ACTIVE_STATUSES
                    ]
                for parent in parents:
                    self._finish_batch(parent)

    def _finish_batch(self, job_id: UUID) -> None:
        """按子任务类型汇总，兼容执行顺序调整前已保存的任务树。"""
        with self._lock:
            parent = self._jobs[job_id]
            children = [self._jobs[i] for i in self._batch_child_ids[job_id]]
            if any(c.status in _ACTIVE_STATUSES for c in children):
                return
            titles = dict(_ALL_JOB_STAGES)
            failed = [titles.get(c.job_type, c.job_type) for c in children if c.status != "SUCCEEDED"]
            success = len(children) - len(failed)
            changes = {
                "status": "SUCCEEDED" if not failed else "PARTIAL_SUCCESS" if success else "FAILED",
                "progress_message": f"一键同步已结束：成功 {success} 项，未完成 {len(failed)} 项",
                "error_code": "SYNC_ALL_INCOMPLETE" if failed else None,
                "error_message": (
                    "未完成：" + "、".join(failed) + "。请检查结果后重试；分析资料留存可重新执行一键同步。"
                    if failed
                    else None
                ),
            }
            if any(getattr(parent, key) != value for key, value in changes.items()):
                self._jobs[job_id] = replace(parent, **changes)
                self._persist()

    def close(self) -> None:
        """停止接受新任务；不阻断进行中的同步写库。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=False)
        self._status_executor.shutdown(wait=False, cancel_futures=False)

    def get_last_successful_time(self, job_type: str) -> datetime | None:
        """从持久化运行记录读取指定任务最近一次完整成功时间。"""
        if job_type == FUND_NEWS_JOB_TYPE:
            from app.services.fund_exposure_common import read
            from app.services.fund_news_sync import directory

            path = directory() / "state.json"
            value = read(path).get("last_success_at") if path.exists() else None
            return datetime.fromisoformat(value) if value else None
        if job_type == FUND_MATERIALS_JOB_TYPE:
            from app.services.fund_exposure_common import read
            from app.services.fund_materials_store import LIVE

            path = LIVE / "state.json"
            value = read(path).get("last_success_at") if path.exists() else None
            return datetime.fromisoformat(value) if value else None
        sync_types = _SYNC_TYPES_BY_JOB_TYPE.get(job_type)
        if sync_types is None:
            raise ValueError("unsupported sync job type")
        with Session(get_engine()) as session:
            return get_latest_successful_sync_time(session, sync_types=sync_types)

    def _run_fund_news(self, job_id: UUID) -> None:
        """公告事实独立更新，部分来源失败不能写成无事件，也不自动触发训练或交易。"""
        from app.services.fund_news_sync import synchronize

        self._replace_job(
            job_id,
            status="RUNNING",
            started_at=datetime.now(UTC),
            fund_codes=("002112",),
            progress_message="正在核对近期基金公告",
        )
        try:
            result = (self._news_synchronizer or synchronize)(
                "002112",
                progress=lambda current, total, code, message: self._update_progress(
                    job_id, current, total, code, message
                ),
            )
            self._replace_job(
                job_id,
                status=result["status"],
                progress_message=result["message"],
                fetched_count=result["items"],
                skipped_count=result["items"] if result.get("reused") else 0,
                error_code="NEWS_INCOMPLETE" if result["status"] != "SUCCEEDED" else None,
                error_message=result["message"] if result["status"] != "SUCCEEDED" else None,
                finished_at=datetime.now(UTC),
            )
        except Exception:
            logger.exception("sync_jobs._run_fund_news >>> 公告核对失败, job_id=%s", job_id)
            self._fail_job(job_id, "NEWS_SYNC_FAILED", "近期公告核对未完成，保留原有资料，请稍后重试。")
        finally:
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None

    def _run_fund_materials(self, job_id: UUID) -> None:
        """同步结束后才设置终态；批次和单独入口共用采集器、文件断点和后台锁。"""
        from app.services.fund_materials_sync import FundMaterialsSyncService, failure_message

        self._replace_job(
            job_id,
            status="RUNNING",
            started_at=datetime.now(UTC),
            fund_codes=("002112",),
            progress_message="正在检查基金持仓与公司资料",
        )
        try:
            service = (self._materials_service_factory or FundMaterialsSyncService)()
            result = service.sync(
                "002112",
                progress_reporter=lambda current, total, code, message: self._update_progress(
                    job_id, current, total, code, message
                ),
            )
            failed_stages = list(dict.fromkeys(e["stage"] + "：" + failure_message(e) for e in result["errors"]))
            self._replace_job(
                job_id,
                status=result["status"],
                progress_message=result["message"],
                result_summary=result.get("result_summary"),
                created_count=result["created"],
                updated_count=result["updated"],
                skipped_count=result["skipped"],
                fetched_count=sum(result[k] for k in ("created", "updated", "skipped")),
                error_code="MATERIALS_INCOMPLETE" if result["errors"] else None,
                error_message=("未完成：" + "、".join(failed_stages) + "。重试将复用已保存成果。")
                if failed_stages
                else None,
                finished_at=datetime.now(UTC),
            )
        except Exception:
            logger.exception("sync_jobs._run_fund_materials >>> 资料同步异常, job_id=%s", job_id)
            self._fail_job(job_id, "MATERIALS_SYNC_FAILED", "资料更新未完成，已保存成果可复用，请稍后重试。")
        finally:
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None

    def _run_fund_inputs(self, job_id: UUID) -> None:
        """一键同步内串行更新 002112 输入；沿用批次锁、真实进度和持久化任务结果。"""
        from app.services.fund_exposure_runtime import synchronize_inputs

        self._replace_job(
            job_id,
            status="RUNNING",
            started_at=datetime.now(UTC),
            fund_codes=("002112",),
            current_fund_code="002112",
            progress_total=4,
            progress_message="正在更新 002112 分析资料",
        )
        try:
            result = (self._input_synchronizer or synchronize_inputs)(
                progress_reporter=lambda current, total, code, message: self._update_progress(
                    job_id, current, total, code, message
                ),
            )
            complete = result["status"] == "SUCCEEDED"
            self._replace_job(
                job_id,
                status=result["status"],
                current_fund_code=None,
                progress_message=result["message"],
                created_count=result["created"],
                skipped_count=result["skipped"],
                error_code=None if complete else "FUND_INPUTS_INCOMPLETE",
                error_message=None if complete else result["message"],
                finished_at=datetime.now(UTC),
            )
        except Exception:
            logger.exception("sync_jobs._run_fund_inputs >>> 002112 分析资料更新异常, job_id=%s", job_id)
            self._fail_job(job_id, "FUND_INPUTS_FAILED", "002112 分析资料未更新，请稍后重新执行一键同步。")

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
            if feature_summary.status == "SOURCE_NOT_READY":
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
        """只有 Java 确认留档才计为生成；不适用单列，缺数据和执行故障仍显示未完成。"""
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
                result_summary=prediction_summary(list(result.items)),
                progress_current=result.total,
                progress_total=result.total,
                current_fund_code=None,
                fetched_count=result.total,
                created_count=result.created,
                updated_count=result.existing,
                skipped_count=len(result.issues) + len(result.unsupported),
                progress_message=(
                    f"目标日 {result.target_date}：共检查 {result.total} 只，新生成 {result.created}，"
                    f"已有 {result.existing}，暂不支持 {len(result.unsupported)}，待完成 {len(result.issues)}"
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
                if not result.issues and not result.followup_issues
                else "PARTIAL_SUCCESS"
                if result.created + result.existing
                else "FAILED"
            )
            states = {
                state: sum(item["state"] == state for item in result.items)
                for state in ("COMPLETED", "WAITING", "UNSUPPORTED", "ERROR")
            }
            summary = prediction_summary(
                list(result.items),
                {
                    "savedResults": list(result.saved_results),
                    "followupIssues": list(result.followup_issues),
                },
            )
            all_issues = result.issues + result.followup_issues
            self._replace_job(
                job_id,
                result_summary=summary,
                status=status,
                requested_nav_date=result.target_date,
                progress_current=result.total,
                progress_total=result.total,
                current_fund_code=None,
                fetched_count=result.total,
                created_count=result.created,
                updated_count=result.existing,
                skipped_count=len(result.issues) + len(result.unsupported),
                progress_message=(
                    f"一日、五日、二十日和半年共处理 {result.total} 个基金周期项，新生成 {result.created}，"
                    f"已有 {result.existing}，暂不支持 {len(result.unsupported)}，待完成 {len(result.issues)}"
                    + (
                        f"（正在生成 {summary['pendingDailyCount']}，"
                        f"等待资料 {states['WAITING'] - summary['pendingDailyCount']}，执行异常 {states['ERROR']}）"
                        if result.items
                        else ""
                    )
                    + (f"；另有 {len(result.followup_issues)} 项收尾待处理" if result.followup_issues else "")
                ),
                error_code="PREDICTION_INCOMPLETE" if all_issues else None,
                error_message="；".join(all_issues[:100]) if all_issues else None,
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
            else "部分资料尚未完成，成功数据已保存；请按具体原因重试",
            error_code=(
                "NAV_SYNC_INCOMPLETE" if outcome.sync_type == "MARKET_NAV_INCREMENTAL" else "DATA_SYNC_INCOMPLETE"
            )
            if outcome.issues
            else None,
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
            status="SUCCEEDED" if not summary.issues else summary.status,
            progress_current=snapshot.progress_total,
            current_fund_code=None,
            error_code="FEATURE_BUILD_INCOMPLETE" if summary.issues else None,
            error_message="；".join(summary.issues) or None,
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
        # 内部执行只读取本地状态；对外查询才核对异步回执，避免收尾时重入网络调用。
        with self._lock:
            snapshot = self._snapshot(job_id)
        if snapshot is None:
            raise LookupError(f"sync job does not exist: {job_id}")
        return snapshot

    def _replace_job(self, job_id: UUID, **changes: object) -> None:
        with self._lock:
            snapshot = self._jobs.get(job_id)
            if snapshot is not None:
                self._jobs[job_id] = replace(snapshot, **changes)
                self._persist()


_manager_lock = Lock()
_manager: LocalSyncJobManager | None = None


def get_sync_job_manager() -> LocalSyncJobManager:
    """返回当前 FastAPI 进程的同步任务中心单例。"""
    global _manager
    with _manager_lock:
        if _manager is None:
            from app.services.fund_exposure_common import ROOT

            _manager = LocalSyncJobManager(state_path=ROOT.parent / "sync-center" / "jobs.json")
        return _manager


def close_sync_job_manager() -> None:
    """应用关闭时释放后台线程资源。"""
    global _manager
    with _manager_lock:
        manager = _manager
        _manager = None
    if manager is not None:
        manager.close()
