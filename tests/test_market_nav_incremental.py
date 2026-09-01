"""基金市场日常增量同步的离线边界测试。"""

from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest
from app.core.config import Settings
from app.integrations.tushare import TushareFundNav
from app.services.stock_feature_snapshot import StockFeatureBuildSummary
from app.services.tushare_fund_sync import (
    MarketNavIncrementalPreconditionError,
    SyncOutcome,
    _build_market_nav_incremental_windows,
    _normalize_market_nav_incremental_records,
)
from app.workers import tasks
from app.workers.celery_app import build_beat_schedule


def test_incremental_windows_start_after_each_fund_tushare_watermark() -> None:
    """每只基金按自己的同源水位补数，不能由其他基金日期推进。"""
    windows = _build_market_nav_incremental_windows(
        ts_codes=("002112.OF", "010710.OF"),
        latest_nav_dates={"002112": date(2026, 8, 25), "010710": date(2026, 8, 26)},
        as_of_date=date(2026, 8, 27),
    )

    assert [(item.ts_code, item.start_date, item.end_date) for item in windows] == [
        ("002112.OF", date(2026, 8, 26), date(2026, 8, 27)),
        ("010710.OF", date(2026, 8, 27), date(2026, 8, 27)),
    ]


def test_incremental_windows_reject_missing_history_baseline() -> None:
    """首次完整历史回填未完成时，日常任务不得把缺口当成已同步。"""
    with pytest.raises(MarketNavIncrementalPreconditionError, match="baseline is missing"):
        _build_market_nav_incremental_windows(
            ts_codes=("002112.OF", "010710.OF"),
            latest_nav_dates={"002112": date(2026, 8, 25)},
            as_of_date=date(2026, 8, 26),
        )


def test_incremental_normalization_rejects_another_fund_and_window_outside_record() -> None:
    """日常任务只接受对应基金自身窗口内的值，避免错配或倒灌历史。"""
    windows = _build_market_nav_incremental_windows(
        ts_codes=("002112.OF",),
        latest_nav_dates={"002112": date(2026, 8, 25)},
        as_of_date=date(2026, 8, 27),
    )
    records, invalid_count = _normalize_market_nav_incremental_records(
        (
            TushareFundNav("002112.OF", None, date(2026, 8, 26), Decimal("4.9"), Decimal("5.1")),
            TushareFundNav("002112.OF", None, date(2026, 8, 25), Decimal("4.8"), Decimal("5.0")),
            TushareFundNav("010710.OF", None, date(2026, 8, 26), Decimal("1.1"), Decimal("1.2")),
        ),
        windows=windows,
    )

    assert invalid_count == 2
    assert [(record.fund_code, record.nav_date) for record in records] == [("002112", date(2026, 8, 26))]


def test_beat_schedule_is_configurable_and_can_be_disabled() -> None:
    """只有显式启用时注册单一工作日任务，时间从本机配置读取。"""
    enabled = Settings(
        tushare_market_incremental_enabled=True,
        tushare_market_incremental_hour=21,
        tushare_market_incremental_minute=5,
    )
    schedule = build_beat_schedule(enabled)

    trigger = schedule["market-nav-incremental-weekdays"]["schedule"]
    assert schedule["market-nav-incremental-weekdays"]["task"] == "fund_ai.tushare.sync_market_nav_incremental"
    assert trigger.hour == {21}
    assert trigger.minute == {5}
    assert build_beat_schedule(Settings(tushare_market_incremental_enabled=False)) == {}


def test_beat_schedule_defaults_to_weekday_2000() -> None:
    """未配置环境变量时，日常任务默认在工作日 20:00 投递。"""
    trigger = build_beat_schedule(Settings())["market-nav-incremental-weekdays"]["schedule"]

    assert trigger.hour == {20}
    assert trigger.minute == {0}


def test_incremental_task_uses_market_scope_and_optional_as_of_date(monkeypatch) -> None:
    """Celery 任务只提供日期；基金市场范围由同步服务从数据库读取。"""
    received: dict[str, object] = {}

    class StubService:
        def sync_market_nav_incremental(self, *, as_of_date: date | None) -> SyncOutcome:
            received["as_of_date"] = as_of_date
            return SyncOutcome(
                sync_run_id=uuid4(),
                sync_type="MARKET_NAV_INCREMENTAL",
                requested_nav_date=as_of_date,
                fetched_count=0,
                created_count=0,
                updated_count=0,
                skipped_count=0,
            )

        def close(self) -> None:
            received["closed"] = True

    class StubFeatureService:
        def build(self) -> StockFeatureBuildSummary:
            received["feature_build"] = True
            return StockFeatureBuildSummary(
                status="COMPLETED",
                source_code="TUSHARE_PRO_FUND",
                source_sync_run_id=uuid4(),
                attempted_fund_count=2,
                scorable_count=1,
                data_insufficient_count=1,
                no_nav_count=0,
                created_count=1,
                updated_count=0,
                skipped_count=1,
            )

    monkeypatch.setattr(tasks, "TushareFundSyncService", StubService)
    monkeypatch.setattr(tasks, "StockFeatureSnapshotService", StubFeatureService)
    payload = tasks.sync_market_nav_incremental.run("2026-08-27")

    assert received == {
        "as_of_date": date(2026, 8, 27),
        "feature_build": True,
        "closed": True,
    }
    assert payload["sync_type"] == "MARKET_NAV_INCREMENTAL"
    assert payload["requested_nav_date"] == "2026-08-27"
    assert payload["feature_snapshot"]["status"] == "COMPLETED"


def test_feature_failure_after_incremental_sync_does_not_retry_the_source(monkeypatch) -> None:
    """特征失败应以部分结果返回，避免 Celery 为本地计算重跑外部市场同步。"""
    outcome = SyncOutcome(
        sync_run_id=uuid4(),
        sync_type="MARKET_NAV_INCREMENTAL",
        requested_nav_date=date(2026, 8, 27),
        fetched_count=3,
        created_count=1,
        updated_count=1,
        skipped_count=1,
    )

    class FailingFeatureService:
        def build(self) -> StockFeatureBuildSummary:
            raise RuntimeError("local feature persistence failed")

    monkeypatch.setattr(tasks, "StockFeatureSnapshotService", FailingFeatureService)
    payload = tasks._with_feature_snapshot_payload(outcome)

    assert payload["sync_type"] == "MARKET_NAV_INCREMENTAL"
    assert payload["fetched_count"] == 3
    assert payload["feature_snapshot"] == {
        "status": "FAILED",
        "error_code": "FEATURE_SNAPSHOT_BUILD_FAILED",
        "message": "来源净值同步成功，但特征快照未完成；请在同步中心手动重试。",
    }
