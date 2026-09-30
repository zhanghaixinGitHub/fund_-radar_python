"""隔离数据处理与模型任务的 Celery 配置。"""

from celery import Celery

from app.core.config import Settings, get_settings


def build_beat_schedule(settings: Settings) -> dict[str, dict[str, object]]:
    """保留旧调用契约，但不再注册净值自动同步，即使旧环境仍配置为启用。"""
    return {}

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
