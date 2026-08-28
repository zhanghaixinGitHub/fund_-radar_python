"""本机同步任务中心的离线测试。"""

from datetime import date
from threading import Event
from uuid import UUID

from app.services.sync_jobs import MARKET_DETAIL_JOB_TYPE, LocalSyncJobManager
from app.services.tushare_fund_sync import MarketDetailSyncResult, SyncOutcome


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

    manager = LocalSyncJobManager(service_factory=StubService)
    started = manager.start_market_nav_incremental()

    assert started.status in {"QUEUED", "RUNNING"}
    assert completed.wait(timeout=1)
    result = manager.get_job(started.job_id)
    manager.close()

    assert result is not None
    assert result.status == "SUCCEEDED"
    assert result.progress_current == 3
    assert result.progress_total == 3
    assert result.sync_run_id == UUID("00000000-0000-0000-0000-000000000303")
    assert (result.fetched_count, result.created_count, result.updated_count, result.skipped_count) == (4, 2, 1, 1)


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
