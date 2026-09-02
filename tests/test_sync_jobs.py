"""本机同步任务中心的离线测试。"""

from datetime import date
from threading import Event
from time import sleep
from uuid import UUID

from app.services.stock_feature_snapshot import FeatureSnapshotBuildInProgressError, StockFeatureBuildSummary
from app.services.sync_jobs import (
    MARKET_DETAIL_JOB_TYPE,
    MARKET_FREE_DATA_COMPLETION_JOB_TYPE,
    STOCK_FEATURE_SNAPSHOT_JOB_TYPE,
    LocalSyncJobManager,
)
from app.services.tushare_free_data_completion import FreeDataCompletionResult
from app.services.tushare_fund_sync import MarketDetailSyncResult, SyncOutcome


def _completed_feature_summary() -> StockFeatureBuildSummary:
    return StockFeatureBuildSummary(
        status="COMPLETED",
        source_code="TUSHARE_PRO_FUND",
        source_sync_run_id=UUID("00000000-0000-0000-0000-000000000303"),
        attempted_fund_count=2,
        scorable_count=1,
        data_insufficient_count=1,
        no_nav_count=0,
        created_count=1,
        updated_count=0,
        skipped_count=1,
    )


def test_local_sync_job_manager_reports_progress_and_final_counts() -> None:
    """后台任务必须保留逐步进度和最终写入统计，且不需要真实 Tushare 调用。"""
    completed = Event()

    class StubService:
        def sync_market_nav_incremental(self, *, progress_reporter):
            progress_reporter(1, 3, "002112.OF", "已读取 002112.OF 的待补齐净值")
            progress_reporter(2, 3, "010710.OF", "已读取 010710.OF 的待补齐净值")
            progress_reporter(3, 3, None, "同步完成")
            return SyncOutcome(
                sync_run_id=UUID("00000000-0000-0000-0000-000000000303"),
                sync_type="MARKET_NAV_INCREMENTAL",
                requested_nav_date=date(2026, 8, 27),
                fetched_count=4,
                created_count=2,
                updated_count=1,
                skipped_count=1,
            )

        def close(self) -> None:
            completed.set()

    class StubFeatureService:
        def build(self, *, progress_reporter):
            progress_reporter(1, 2, "002112.OF", "正在生成 002112.OF 的特征快照")
            progress_reporter(2, 2, "010710.OF", "正在生成 010710.OF 的特征快照")
            return _completed_feature_summary()

    manager = LocalSyncJobManager(service_factory=StubService, feature_service_factory=StubFeatureService)
    started = manager.start_market_nav_incremental()

    assert started.status in {"QUEUED", "RUNNING"}
    assert completed.wait(timeout=1)
    result = manager.get_job(started.job_id)
    manager.close()

    assert result is not None
    assert result.status == "SUCCEEDED"
    assert result.progress_current == 2
    assert result.progress_total == 2
    assert result.progress_message == "特征快照同步完成：处理 2 只，新建 1，更新 0，未变化 1"
    assert result.sync_run_id == UUID("00000000-0000-0000-0000-000000000303")
    assert (result.fetched_count, result.created_count, result.updated_count, result.skipped_count) == (4, 2, 1, 1)


def test_market_nav_sync_keeps_source_success_when_feature_build_fails() -> None:
    """特征阶段异常不得覆盖已成功的市场同步事实，管理员可从独立入口重试。"""
    completed = Event()

    class StubService:
        def sync_market_nav_incremental(self, *, progress_reporter):
            progress_reporter(1, 1, "002112.OF", "基金市场净值同步完成")
            return SyncOutcome(
                sync_run_id=UUID("00000000-0000-0000-0000-000000000305"),
                sync_type="MARKET_NAV_INCREMENTAL",
                requested_nav_date=date(2026, 9, 1),
                fetched_count=3,
                created_count=1,
                updated_count=1,
                skipped_count=1,
            )

        def close(self) -> None:
            completed.set()

    class FailingFeatureService:
        def build(self, *, progress_reporter):
            raise RuntimeError("feature persistence failed")

    manager = LocalSyncJobManager(service_factory=StubService, feature_service_factory=FailingFeatureService)
    started = manager.start_market_nav_incremental()

    assert completed.wait(timeout=1)
    result = manager.get_job(started.job_id)
    manager.close()

    assert result is not None
    assert result.status == "PARTIAL_SUCCESS"
    assert result.error_code == "FEATURE_SNAPSHOT_BUILD_FAILED"
    assert result.sync_run_id == UUID("00000000-0000-0000-0000-000000000305")
    assert (result.fetched_count, result.created_count, result.updated_count, result.skipped_count) == (3, 1, 1, 1)


def test_manual_feature_snapshot_job_reports_local_write_counts() -> None:
    """独立特征任务只汇总本地计算结果，不要求或触发基金市场外部调用。"""
    completed = Event()

    class StubFeatureService:
        def build(self, *, progress_reporter):
            progress_reporter(1, 2, "002112.OF", "正在生成 002112.OF 的特征快照")
            progress_reporter(2, 2, "010710.OF", "正在生成 010710.OF 的特征快照")
            completed.set()
            return _completed_feature_summary()

    manager = LocalSyncJobManager(feature_service_factory=StubFeatureService)
    started = manager.start_stock_feature_snapshots()

    assert started.job_type == STOCK_FEATURE_SNAPSHOT_JOB_TYPE
    assert completed.wait(timeout=1)
    for _ in range(100):
        result = manager.get_job(started.job_id)
        if result is not None and result.status not in {"QUEUED", "RUNNING"}:
            break
        sleep(0.01)
    manager.close()

    assert result is not None
    assert result.status == "SUCCEEDED"
    assert result.sync_run_id == UUID("00000000-0000-0000-0000-000000000303")
    assert (result.fetched_count, result.created_count, result.updated_count, result.skipped_count) == (2, 1, 0, 1)


def test_manual_feature_snapshot_job_reports_cross_process_lock_conflict() -> None:
    """计划任务占用特征锁时，手动重试必须安全结束并提示稍后重试。"""

    class LockedFeatureService:
        def build(self, *, progress_reporter):
            raise FeatureSnapshotBuildInProgressError("stock feature snapshot build is already running")

    manager = LocalSyncJobManager(feature_service_factory=LockedFeatureService)
    started = manager.start_stock_feature_snapshots()

    for _ in range(100):
        result = manager.get_job(started.job_id)
        if result is not None and result.status not in {"QUEUED", "RUNNING"}:
            break
        sleep(0.01)
    manager.close()

    assert result is not None
    assert result.status == "FAILED"
    assert result.error_code == "FEATURE_SYNC_IN_PROGRESS"


def test_local_sync_job_manager_reports_market_detail_stage_progress() -> None:
    """完整资料任务必须回传阶段与逐基金进度，并汇总父运行统计。"""
    completed = Event()

    class StubService:
        def sync_market_details(self, *, progress_reporter):
            progress_reporter(1, 9, None, "基金基础资料已写入")
            progress_reporter(5, 9, "010710.OF", "已读取 010710.OF 的基金经理资料")
            progress_reporter(9, 9, None, "完整资料同步完成")
            overall = SyncOutcome(
                sync_run_id=UUID("00000000-0000-0000-0000-000000000304"),
                sync_type=MARKET_DETAIL_JOB_TYPE,
                requested_nav_date=date(2026, 8, 28),
                fetched_count=12,
                created_count=7,
                updated_count=3,
                skipped_count=2,
            )
            return MarketDetailSyncResult(overall_outcome=overall, outcomes=(overall,))

        def close(self) -> None:
            completed.set()

    manager = LocalSyncJobManager(service_factory=StubService)
    started = manager.start_market_details()

    assert started.job_type == MARKET_DETAIL_JOB_TYPE
    assert completed.wait(timeout=1)
    result = manager.get_job(started.job_id)
    manager.close()

    assert result is not None
    assert result.status == "SUCCEEDED"
    assert result.progress_current == 9
    assert result.progress_total == 9
    assert result.progress_message == "同步完成"
    assert result.sync_run_id == UUID("00000000-0000-0000-0000-000000000304")
    assert (result.fetched_count, result.created_count, result.updated_count, result.skipped_count) == (12, 7, 3, 2)


def test_free_data_completion_job_reports_parent_run_summary() -> None:
    """免费数据补齐必须以独立父运行回传汇总，仍受同步中心单并发控制。"""
    completed = Event()

    class StubFreeDataCompletionService:
        def sync(self, *, progress_reporter):
            progress_reporter(1, 2, "510300.SH", "正在同步场内基金日线")
            progress_reporter(2, 2, None, "当前免费数据补齐完成")
            outcome = SyncOutcome(
                sync_run_id=UUID("00000000-0000-0000-0000-000000000306"),
                sync_type=MARKET_FREE_DATA_COMPLETION_JOB_TYPE,
                requested_nav_date=date(2026, 9, 2),
                fetched_count=10,
                created_count=7,
                updated_count=2,
                skipped_count=1,
            )
            return FreeDataCompletionResult(overall_outcome=outcome, outcomes=(outcome,))

        def close(self) -> None:
            completed.set()

    manager = LocalSyncJobManager(free_data_completion_service_factory=StubFreeDataCompletionService)
    started = manager.start_market_free_data_completion()

    assert started.job_type == MARKET_FREE_DATA_COMPLETION_JOB_TYPE
    assert completed.wait(timeout=1)
    result = manager.get_job(started.job_id)
    manager.close()

    assert result is not None
    assert result.status == "SUCCEEDED"
    assert result.progress_current == 2
    assert result.progress_total == 2
    assert result.progress_message == "当前 2000 积分已授权数据补齐完成"
    assert result.sync_run_id == UUID("00000000-0000-0000-0000-000000000306")
    assert (result.fetched_count, result.created_count, result.updated_count, result.skipped_count) == (10, 7, 2, 1)
