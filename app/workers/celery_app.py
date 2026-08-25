"""隔离数据处理与模型任务的 Celery 配置。"""

from celery import Celery

from app.core.config import get_settings

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
)
