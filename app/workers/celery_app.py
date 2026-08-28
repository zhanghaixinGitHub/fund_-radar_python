"""隔离数据处理与模型任务的 Celery 配置。"""

from celery import Celery
from celery.schedules import crontab

from app.core.config import Settings, get_settings


def build_beat_schedule(settings: Settings) -> dict[str, dict[str, object]]:
    """构建基金市场日常增量同步计划；禁用时不注册外部数据调用。"""
    if not settings.tushare_market_incremental_enabled:
        return {}
    return {
        "market-nav-incremental-weekdays": {
            "task": "fund_ai.tushare.sync_market_nav_incremental",
            "schedule": crontab(
                day_of_week="1-5",
                hour=settings.tushare_market_incremental_hour,
                minute=settings.tushare_market_incremental_minute,
            ),
        }
    }

"""后台任务使用的进程级配置快照。"""
settings = get_settings()

"""基金 AI 后台任务应用，任务实现集中在 app.workers.tasks。"""
celery_app = Celery(
    "fund_ai",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.tasks"],
)
celery_app.conf.update(
    task_default_queue="fund_ai",
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
    beat_schedule=build_beat_schedule(settings),
)
