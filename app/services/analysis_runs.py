"""M3-05 受控分析运行编排；任务状态可跨 FastAPI 进程重启查询。"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.session import get_engine
from app.models.analysis import AnalysisRun
from app.schemas.analysis_run import InternalAnalysisRunStatus, InternalModelReleaseStatus
from app.services.analysis_explanation import (
    FundExplanationSourceNotReadyError,
    generate_fund_explanation,
    validate_deepseek_explanation_configuration,
)
from app.services.baseline_analysis import STOCK_FUND_TYPE, BaselineAnalysisService, RollingBacktestConfig
from app.services.deepseek_explanation_client import DeepSeekExplanationError

logger = get_logger(__name__)

ROLLING_BACKTEST_RUN_TYPE = "ROLLING_BACKTEST"
FUND_EXPLANATION_RUN_TYPE = "FUND_EXPLANATION"
_ANALYSIS_RUN_LOCK_KEY = 7_089_123_103
_ACTIVE_RUN_STATUSES = ("QUEUED", "RUNNING")


class AnalysisRunInProgressError(RuntimeError):
    """存在相同受控分析任务时拒绝重复提交。"""


class AnalysisRunNotFoundError(LookupError):
    """请求的持久分析任务不存在时抛出。"""


def start_stock_rolling_backtest(
    *,
    fee_rate: Decimal,
    benchmark_code: str | None,
    trace_id: str,
) -> InternalAnalysisRunStatus:
    """创建并提交股票型滚动回测；基准不可用时任务保留可追溯失败结果。"""
    config = RollingBacktestConfig(fee_rate=fee_rate, benchmark_id=benchmark_code)
    config.validate()
    engine = get_engine()
    with Session(engine) as session:
        session.execute(select(func.pg_advisory_xact_lock(_ANALYSIS_RUN_LOCK_KEY)))
        active = session.scalar(
            select(AnalysisRun.analysis_run_id)
            .where(AnalysisRun.status.in_(_ACTIVE_RUN_STATUSES))
            .limit(1)
        )
        if active is not None:
            raise AnalysisRunInProgressError("an analysis run is already queued or running")
        run = AnalysisRun(
            run_type=ROLLING_BACKTEST_RUN_TYPE,
            status="QUEUED",
            fund_type=STOCK_FUND_TYPE,
            config_hash=config.config_hash,
            request_payload={"fee_rate": str(config.fee_rate), "benchmark_id": config.benchmark_id},
            trace_id=trace_id,
        )
        session.add(run)
        session.flush()
        analysis_run_id = run.analysis_run_id
        session.commit()

    try:
        from app.workers.tasks import run_controlled_stock_rolling_backtest

        task = run_controlled_stock_rolling_backtest.delay(str(analysis_run_id))
    except Exception as error:
        _mark_failed(analysis_run_id, "analysis task could not be queued")
        logger.exception(
            "analysis_runs.start_stock_rolling_backtest >>> queue submission failed, analysis_run_id=%s",
            analysis_run_id,
        )
        raise RuntimeError("analysis task could not be queued") from error

    with Session(engine) as session:
        run = _required_run(session, analysis_run_id, lock=True)
        run.task_id = task.id
        session.commit()
        session.refresh(run)
        payload = _to_status(run)
    logger.info(
        "analysis_runs.start_stock_rolling_backtest >>> queued, analysis_run_id=%s, task_id=%s",
        analysis_run_id,
        task.id,
    )
    return payload


def execute_stock_rolling_backtest(analysis_run_id: UUID) -> dict[str, str | None]:
    """由 Celery Worker 执行一条已持久化运行，并将结果或失败原因回写。"""
    engine = get_engine()
    with Session(engine) as session:
        run = _required_run(session, analysis_run_id, lock=True)
        if run.status == "COMPLETED":
            return _task_payload(run)
        if run.status == "FAILED":
            return _task_payload(run)
        if run.status != "QUEUED":
            raise RuntimeError(f"analysis run has unsupported execution status={run.status}")
        run.status = "RUNNING"
        run.started_at = datetime.now(UTC)
        fee_rate = Decimal(str(run.request_payload["fee_rate"]))
        benchmark_id = run.request_payload.get("benchmark_id")
        if benchmark_id is not None and not isinstance(benchmark_id, str):
            raise RuntimeError("analysis run benchmark_id is invalid")
        session.commit()

    try:
        summary = BaselineAnalysisService().run_rolling_backtest(
            RollingBacktestConfig(fee_rate=fee_rate, benchmark_id=benchmark_id)
        )
    except Exception:
        _mark_failed(analysis_run_id, "rolling backtest execution failed")
        logger.exception(
            "analysis_runs.execute_stock_rolling_backtest >>> execution failed, analysis_run_id=%s",
            analysis_run_id,
        )
        raise

    result_payload = summary.to_payload()
    with Session(engine) as session:
        run = _required_run(session, analysis_run_id, lock=True)
        run.status = "COMPLETED"
        run.backtest_run_id = summary.run_id
        run.result_payload = result_payload
        run.failure_reason = summary.failure_reason
        run.finished_at = datetime.now(UTC)
        session.commit()
    logger.info(
        "analysis_runs.execute_stock_rolling_backtest >>> completed, analysis_run_id=%s, backtest_run_id=%s",
        analysis_run_id,
        summary.run_id,
    )
    return result_payload


def start_fund_explanation(*, fund_code: str, trace_id: str) -> InternalAnalysisRunStatus:
    """排队生成一条已发布评分解释；不接受自由提示词、外部数据或任意模型选择。"""
    normalized_fund_code = _normalize_fund_code(fund_code)
    provider_model = validate_deepseek_explanation_configuration()
    engine = get_engine()
    with Session(engine) as session:
        session.execute(select(func.pg_advisory_xact_lock(_ANALYSIS_RUN_LOCK_KEY)))
        active = session.scalar(
            select(AnalysisRun.analysis_run_id)
            .where(AnalysisRun.status.in_(_ACTIVE_RUN_STATUSES))
            .limit(1)
        )
        if active is not None:
            raise AnalysisRunInProgressError("an analysis run is already queued or running")
        run = AnalysisRun(
            run_type=FUND_EXPLANATION_RUN_TYPE,
            status="QUEUED",
            fund_type=STOCK_FUND_TYPE,
            config_hash=_explanation_config_hash(normalized_fund_code, provider_model),
            request_payload={"fund_code": normalized_fund_code, "provider_model": provider_model},
            trace_id=trace_id,
        )
        session.add(run)
        session.flush()
        analysis_run_id = run.analysis_run_id
        session.commit()

    try:
        from app.workers.tasks import run_controlled_fund_explanation

        task = run_controlled_fund_explanation.delay(str(analysis_run_id))
    except Exception as error:
        _mark_failed(analysis_run_id, "fund explanation task could not be queued")
        logger.exception(
            "analysis_runs.start_fund_explanation >>> queue submission failed, analysis_run_id=%s",
            analysis_run_id,
        )
        raise RuntimeError("fund explanation task could not be queued") from error

    with Session(engine) as session:
        run = _required_run(session, analysis_run_id, lock=True)
        run.task_id = task.id
        session.commit()
        session.refresh(run)
        payload = _to_status(run)
    logger.info(
        "analysis_runs.start_fund_explanation >>> queued, analysis_run_id=%s, fund_code=%s, task_id=%s",
        analysis_run_id,
        normalized_fund_code,
        task.id,
    )
    return payload


def execute_fund_explanation(analysis_run_id: UUID) -> dict[str, str | None]:
    """由 Celery 执行一条解释任务；任何外部失败只写入稳定错误码。"""
    engine = get_engine()
    with Session(engine) as session:
        run = _required_run(session, analysis_run_id, lock=True)
        if run.status in {"COMPLETED", "FAILED"}:
            return _task_payload(run)
        if run.status != "QUEUED" or run.run_type != FUND_EXPLANATION_RUN_TYPE:
            raise RuntimeError(f"analysis run has unsupported explanation execution status={run.status}")
        fund_code = run.request_payload.get("fund_code")
        if not isinstance(fund_code, str):
            raise RuntimeError("analysis explanation run fund_code is invalid")
        run.status = "RUNNING"
        run.started_at = datetime.now(UTC)
        session.commit()

    try:
        generated = generate_fund_explanation(fund_code)
    except (DeepSeekExplanationError, FundExplanationSourceNotReadyError) as error:
        _mark_failed(analysis_run_id, str(error))
        logger.warning(
            "analysis_runs.execute_fund_explanation >>> controlled failure, analysis_run_id=%s, reason=%s",
            analysis_run_id,
            error,
        )
        return _task_payload_for_id(analysis_run_id)
    except Exception:
        _mark_failed(analysis_run_id, "fund explanation execution failed")
        logger.exception(
            "analysis_runs.execute_fund_explanation >>> execution failed, analysis_run_id=%s",
            analysis_run_id,
        )
        raise

    result_payload = {
        "explanation_id": generated.explanation_id,
        "provider_model": generated.provider_model,
        "reused": generated.reused,
    }
    with Session(engine) as session:
        run = _required_run(session, analysis_run_id, lock=True)
        run.status = "COMPLETED"
        run.result_payload = result_payload
        run.finished_at = datetime.now(UTC)
        session.commit()
    logger.info(
        "analysis_runs.execute_fund_explanation >>> completed, analysis_run_id=%s, explanation_id=%s, reused=%s",
        analysis_run_id,
        generated.explanation_id,
        generated.reused,
    )
    return _task_payload_for_id(analysis_run_id)


def get_analysis_run(analysis_run_id: UUID) -> InternalAnalysisRunStatus:
    """读取一条持久分析运行状态；读取本身不触发任务、评分或回测。"""
    with Session(get_engine()) as session:
        run = _required_run(session, analysis_run_id)
        return _to_status(run)


def activate_model_release(model_release_id: UUID, *, reason: str) -> InternalModelReleaseStatus:
    """管理员显式激活满足 M3-04 闸门的发布版本。"""
    release = BaselineAnalysisService().activate_model_release(model_release_id, reason=reason)
    return _to_release_status(release)


def suspend_model_release(model_release_id: UUID, *, reason: str) -> InternalModelReleaseStatus:
    """管理员显式暂停发布版本；不会删除历史评分和回测。"""
    release = BaselineAnalysisService().suspend_model_release(model_release_id, reason=reason)
    return _to_release_status(release)


def _mark_failed(analysis_run_id: UUID, failure_reason: str) -> None:
    """将可安全公开的失败摘要持久化，不写入底层连接或堆栈信息。"""
    with Session(get_engine()) as session:
        run = _required_run(session, analysis_run_id, lock=True)
        run.status = "FAILED"
        run.failure_reason = failure_reason
        run.finished_at = datetime.now(UTC)
        session.commit()


def _required_run(session: Session, analysis_run_id: UUID, *, lock: bool = False) -> AnalysisRun:
    """按需对运行记录加锁，缺失时返回稳定的领域错误。"""
    statement = select(AnalysisRun).where(AnalysisRun.analysis_run_id == analysis_run_id)
    if lock:
        statement = statement.with_for_update()
    run = session.scalar(statement)
    if run is None:
        raise AnalysisRunNotFoundError(f"analysis run not found: {analysis_run_id}")
    return run


def _to_status(run: AnalysisRun) -> InternalAnalysisRunStatus:
    """将 ORM 状态转换为不暴露请求详情的内部响应。"""
    result = run.result_payload or {}
    model_release_id = result.get("model_release_id")
    return InternalAnalysisRunStatus(
        analysis_run_id=run.analysis_run_id,
        run_type=run.run_type,
        status=run.status,
        fund_type=run.fund_type,
        task_id=run.task_id,
        backtest_run_id=run.backtest_run_id,
        model_release_id=UUID(model_release_id) if isinstance(model_release_id, str) else None,
        model_release_status=(
            result.get("model_release_status") if isinstance(result.get("model_release_status"), str) else None
        ),
        failure_reason=run.failure_reason,
        requested_at=run.requested_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
    )


def _to_release_status(release: object) -> InternalModelReleaseStatus:
    """从发布领域对象映射 Python 到 Java 的最小状态契约。"""
    return InternalModelReleaseStatus(
        model_release_id=release.model_release_id,
        model_code=release.model_code,
        model_version=release.model_version,
        feature_version=release.feature_version,
        fund_type=release.fund_type,
        backtest_run_id=release.backtest_run_id,
        release_status=release.release_status,
        effective_at=release.effective_at,
        suspended_at=release.suspended_at,
        release_reason=release.release_reason,
    )


def _task_payload(run: AnalysisRun) -> dict[str, str | None]:
    """返回 Celery 可序列化的现有状态，避免重复执行已经完成的任务。"""
    result = run.result_payload or {}
    return {
        "run_id": str(run.backtest_run_id) if run.backtest_run_id else None,
        "status": run.status,
        "publication_status": (
            result.get("publication_status") if isinstance(result.get("publication_status"), str) else None
        ),
        "failure_reason": run.failure_reason,
        "model_release_id": (
            result.get("model_release_id") if isinstance(result.get("model_release_id"), str) else None
        ),
        "model_release_status": (
            result.get("model_release_status") if isinstance(result.get("model_release_status"), str) else None
        ),
    }


def _task_payload_for_id(analysis_run_id: UUID) -> dict[str, str | None]:
    """读取刚刚持久化的任务结果，保证 Celery 重试也只返回数据库事实。"""
    with Session(get_engine()) as session:
        return _task_payload(_required_run(session, analysis_run_id))


def _normalize_fund_code(fund_code: str) -> str:
    """限制解释目标为标准六位基金代码，避免自由文本成为模型输入的一部分。"""
    normalized = fund_code.strip()
    if len(normalized) != 6 or not normalized.isdigit():
        raise ValueError("fund_code must be a six digit code")
    return normalized


def _explanation_config_hash(fund_code: str, provider_model: str) -> str:
    """为解释任务写入可追溯配置摘要，不保存提示词、密钥或外部请求正文。"""
    return hashlib.sha256(
        json.dumps(
            {"run_type": FUND_EXPLANATION_RUN_TYPE, "fund_code": fund_code, "provider_model": provider_model},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
