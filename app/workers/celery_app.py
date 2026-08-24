"""Celery configuration for isolated data and model tasks."""

from celery import Celery

from app.core.config import get_settings

settings = get_settings()

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
