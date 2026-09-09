"""不可变现金批次：一次源快照、完整校验、原子落库、读回核对和幂等重试。"""

import hashlib
import json
from collections import Counter
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, localcontext
from time import perf_counter
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from app.db.session import get_nav_sample_storage_engine
from app.models.cash_reinvestment import CashSampleBatch
from app.repositories.cash_reinvestment_storage import find_batch, insert_rows, read_rows
from app.schemas.cash_reinvestment_batch import CashBatchRequest, CashBatchResponse
from app.schemas.cash_reinvestment_storage import CashBatchSaveRequest, CashStoredBatch
from app.services.cash_reinvestment_batch import build_cash_batch_in_session, plan_cash_batch
from app.services.cash_reinvestment_samples import _hash
from app.services.historical_nav_evaluation import FEATURE_NAMES
from app.services.historical_nav_samples import _build_metrics
from app.services.historical_nav_storage import HistoricalNavStorageError
from app.services.trading_calendar import load_calendar


def cash_hash(value) -> str:
    """有限JSON内容指纹；不含代码，不反序列化可执行模型。"""
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def validate_cash_batch(raw: CashBatchResponse) -> CashBatchResponse:
    """写入与读回共用；防止标签错配、日期越界、指标与历史序列不同、汇总漂移。"""
    preview = CashBatchResponse.model_validate_json(raw.model_dump_json())
    req = CashBatchRequest(
        fundCode=preview.fund_code, startDate=preview.start_date, endDate=preview.end_date, pageSize=preview.page_size
    )
    calendar = load_calendar()
    plans = plan_cash_batch(req, calendar)
    if preview.calendar_hash != calendar.content_hash or preview.calendar_version != calendar.definition.version:
        raise ValueError("cash calendar mismatch")
    if tuple(i.cutoff_date for i in preview.items) != tuple(p.cutoff_date for p in plans):
        raise ValueError("cash cutoff selection mismatch")
    if (
        cash_hash(preview.model_dump(mode="json", exclude={"page_size", "page_count", "batch_hash"}))
        != preview.batch_hash
    ):
        raise ValueError("cash batch hash mismatch")
    counts = Counter(i.status for i in preview.items)
    if (
        preview.sample_count != len(plans)
        or preview.ready_count != counts["RESEARCH_SAMPLE_READY"]
        or preview.input_unavailable_count != counts["INPUT_UNAVAILABLE"]
        or preview.label_unavailable_count != counts["LABEL_UNAVAILABLE"]
        or preview.input_available_count != sum(i.feature_payload is not None for i in preview.items)
        or preview.label_available_count != sum(i.offline_label is not None for i in preview.items)
        or preview.page_count != (len(plans) + preview.page_size - 1) // preview.page_size
    ):
        raise ValueError("cash counts mismatch")
    for item in preview.items:
        feature, label = item.feature_payload, item.offline_label
        expected_status = (
            "INPUT_UNAVAILABLE"
            if feature is None
            else "LABEL_UNAVAILABLE"
            if label is None
            else "RESEARCH_SAMPLE_READY"
        )
        if (
            item.fund_code != preview.fund_code
            or item.status != expected_status
            or (feature is None and label is not None)
        ):
            raise ValueError("cash sample identity/status mismatch")
        expected_future = tuple(calendar.future_sessions(item.cutoff_date, 20))
        if item.future_dates != expected_future or item.label_end_date != expected_future[-1]:
            raise ValueError("cash future calendar mismatch")
        if item.feature_hash != (_hash(feature) if feature else None) or item.label_hash != (
            _hash(label) if label else None
        ):
            raise ValueError("cash feature/label hash mismatch")
        if feature:
            if (
                feature.fund_code != item.fund_code
                or feature.cutoff_date != item.cutoff_date
                or feature.available_at > item.cutoff_date
                or feature.source_code != preview.source_code
                or feature.source_sync_run_id != preview.source_sync_run_id
                or feature.calendar_hash != preview.calendar_hash
                or feature.anchor_nav_date != item.anchor_nav_date
                or len(feature.history_series) != 61
                or tuple(p.nav_date for p in feature.history_series) != item.history_dates
                or any(
                    p.available_at > item.cutoff_date or p.nav_date > item.cutoff_date for p in feature.history_series
                )
                or set(feature.metrics) != set(FEATURE_NAMES)
            ):
                raise ValueError("cash feature metadata mismatch")
            values = tuple(Decimal(p.growth_index) for p in feature.history_series)
            with localcontext() as context:
                context.prec, context.rounding = 40, ROUND_HALF_UP
                if any(not v.is_finite() or v <= 0 for v in values) or _build_metrics(values) != feature.metrics:
                    raise ValueError("cash feature metrics mismatch")
        if label:
            if (
                label.label_base_date != item.label_base_date
                or label.label_end_date != item.label_end_date
                or label.label_available_at <= item.cutoff_date
                or label.label_available_at > date(2024, 12, 31)
                or tuple(p.nav_date for p in label.label_series) != (item.label_base_date, *item.future_dates)
                or label.label_available_at != label.label_series[-1].available_at
                or label.label_up_20d != int(Decimal(label.future_return_20d) > 0)
            ):
                raise ValueError("cash label metadata mismatch")
    return preview


def restore_cash_batch(batch, rows) -> CashStoredBatch:
    """只组合既存字段，绝不重新访问源数据或自动修复已损坏批次。"""
    if batch is None:
        raise HistoricalNavStorageError("CASH_BATCH_NOT_FOUND", "现金样本批次不存在。", 404)
    try:
        items = []
        for sample, label in rows:
            if sample.batch_id != batch.batch_id or sample.cutoff_date.isoformat() != sample.context["cutoff_date"]:
                raise ValueError("row identity mismatch")
            items.append(
                {
                    **sample.context,
                    "feature_payload": sample.feature_payload,
                    "offline_label": label.label_payload if label else None,
                }
            )
        preview = validate_cash_batch(CashBatchResponse.model_validate({**batch.summary, "items": items}))
        if (batch.fund_code, batch.start_date, batch.end_date, batch.batch_hash) != (
            preview.fund_code,
            preview.start_date,
            preview.end_date,
            preview.batch_hash,
        ):
            raise ValueError("batch identity mismatch")
        return CashStoredBatch(
            batch_id=batch.batch_id, request_key=batch.request_key, created_at=batch.created_at, preview=preview
        )
    except (ValueError, TypeError, KeyError, ArithmeticError) as error:
        raise HistoricalNavStorageError(
            "CASH_BATCH_CORRUPTED", "已存现金样本校验失败，不能继续训练或自动修复。", 503
        ) from error


def read_cash_batch_in_session(session: Session, batch_id: UUID) -> CashStoredBatch:
    batch = find_batch(session, batch_id=batch_id)
    return restore_cash_batch(batch, read_rows(session, batch_id) if batch else ())


def get_cash_batch(batch_id: UUID) -> CashStoredBatch:
    with Session(get_nav_sample_storage_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        return read_cash_batch_in_session(session, batch_id)


def save_cash_batch(request: CashBatchSaveRequest) -> tuple[CashStoredBatch, bool]:
    """仅新请求计算；唯一凭证竞争/序列化冲突最多重试一次，不吞掉其他数据库错误。"""
    deadline = perf_counter() + 15
    plan_cash_batch(request, load_calendar())  # 2025保护必须在建立连接之前执行。
    for attempt in range(2):
        try:
            with Session(get_nav_sample_storage_engine()) as session, session.begin():
                session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ WRITE"))
                previous = find_batch(session, request_key=request.request_key)
                if previous:
                    if (previous.fund_code, previous.start_date, previous.end_date) != (
                        request.fund_code,
                        request.start_date,
                        request.end_date,
                    ):
                        raise HistoricalNavStorageError(
                            "REQUEST_KEY_CONFLICT", "同一requestKey不能用于不同基金或日期。", 409
                        )
                    return restore_cash_batch(previous, read_rows(session, previous.batch_id)), False
                preview = validate_cash_batch(build_cash_batch_in_session(session, request, deadline=deadline))
                batch = CashSampleBatch(
                    batch_id=uuid4(),
                    request_key=request.request_key,
                    fund_code=request.fund_code,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    summary=preview.model_dump(mode="json", exclude={"items"}),
                    batch_hash=preview.batch_hash,
                )
                samples, labels = [], []
                for item in preview.items:
                    sample_id = uuid4()
                    samples.append(
                        dict(
                            sample_id=sample_id,
                            batch_id=batch.batch_id,
                            cutoff_date=item.cutoff_date,
                            context=item.model_dump(mode="json", exclude={"feature_payload", "offline_label"}),
                            feature_payload=item.feature_payload.model_dump(mode="json")
                            if item.feature_payload
                            else None,
                        )
                    )
                    if item.offline_label:
                        labels.append(
                            dict(sample_id=sample_id, label_payload=item.offline_label.model_dump(mode="json"))
                        )
                if perf_counter() >= deadline:
                    raise TimeoutError("cash storage phase timeout")
                insert_rows(session, batch, samples, labels)
                result = restore_cash_batch(batch, read_rows(session, batch.batch_id))
                if perf_counter() >= deadline:
                    raise TimeoutError("cash storage phase timeout")
                return result, True  # 离开事务上下文成功提交之后，调用者才会收到此结果。
        except DBAPIError as error:
            state = getattr(error.orig, "sqlstate", None)
            constraint = getattr(getattr(error.orig, "diag", None), "constraint_name", None)
            retryable = state == "40001" or state == "23505" and constraint == "uq_cash_batch_request"
            if attempt or not retryable or perf_counter() >= deadline:
                raise
    raise RuntimeError("unreachable")
