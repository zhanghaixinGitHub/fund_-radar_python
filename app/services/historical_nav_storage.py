"""将已有构建器接到三表存储：整批事务、并发幂等、原样读回，不训练或更新旧批次。"""

from collections.abc import Sequence
from dataclasses import asdict
from time import perf_counter
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from app.db.session import get_nav_sample_storage_engine
from app.models.historical_nav_sample import (
    HistoricalNavSampleBatch,
    HistoricalNavSampleLabel,
    HistoricalNavSampleRecord,
)
from app.repositories.historical_nav_storage import (
    find_batch_by_request_key,
    get_batch,
    insert_batch_rows,
    read_sample_rows,
)
from app.schemas.historical_nav_storage import HistoricalNavBatchSaveRequest, HistoricalNavStoredBatch
from app.services.historical_nav_preview import HistoricalNavBatchTimeoutError, build_stored_historical_nav_batch
from app.services.historical_nav_samples import (
    HISTORICAL_NAV_FEATURE_VERSION,
    HISTORICAL_NAV_LABEL_VERSION,
    HISTORICAL_NAV_SAMPLE_RULE_VERSION,
    HistoricalNavSample,
    OfflineDirectionLabel,
)
from app.services.historical_nav_storage_validation import validate_batch_samples


class HistoricalNavStorageError(RuntimeError):
    """可对外展示的存储业务错误；不包含SQL、连接地址或凭据。"""

    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def _check_deadline(deadline: float) -> None:
    if perf_counter() >= deadline:
        raise HistoricalNavBatchTimeoutError("样本保存超时，请缩短范围并沿用原requestKey重试。")


def _request_matches(batch: HistoricalNavSampleBatch, request: HistoricalNavBatchSaveRequest) -> bool:
    # 页大小和最新来源水位不参与身份；重试必须返回旧快照，而不是偷偷用新数据重算。
    return (
        batch.fund_code,
        batch.start_date,
        batch.end_date,
        batch.feature_version,
        batch.sample_rule_version,
        batch.label_version,
        batch.purpose,
        batch.fund_type,
    ) == (
        request.fund_code,
        request.start_date,
        request.end_date,
        HISTORICAL_NAV_FEATURE_VERSION,
        HISTORICAL_NAV_SAMPLE_RULE_VERSION,
        HISTORICAL_NAV_LABEL_VERSION,
        "LEARNING_ONLY",
        "STOCK",
    )


def _is_retryable_conflict(error: DBAPIError) -> bool:
    """唯一键冲突或一致性快照并发冲突才自动重试；其他数据库错误不吞掉。"""
    code = getattr(error.orig, "sqlstate", None)
    constraint = getattr(getattr(error.orig, "diag", None), "constraint_name", None)
    return code == "40001" or (code == "23505" and constraint == "uq_historical_nav_batch_request")


def save_historical_nav_batch(request: HistoricalNavBatchSaveRequest) -> tuple[HistoricalNavStoredBatch, bool]:
    """返回(完整批次,是否新建)；最多一次并发重试，共享15秒阶段预算。

    所有读取/计算/三表写入处于同一REPEATABLE READ事务。两个进程同时用同一凭证时，
    由数据库唯一索引仲裁；输的一方回滚后开启新快照读回赢家，不使用进程内锁冒充跨实例幂等。
    """
    deadline = perf_counter() + 15
    for attempt in range(2):
        _check_deadline(deadline)
        try:
            with Session(get_nav_sample_storage_engine()) as session, session.begin():
                session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ WRITE"))
                existing = find_batch_by_request_key(session, request.request_key)
                if existing is not None:
                    if not _request_matches(existing, request):
                        raise HistoricalNavStorageError(
                            "REQUEST_KEY_CONFLICT",
                            "该requestKey已用于其他范围或规则版本；明确重算请换新凭证。",
                            409,
                        )
                    result = _read_result(session, existing)
                    _check_deadline(deadline)
                    return result, False
                source, preview = build_stored_historical_nav_batch(session, request, deadline=deadline)
                if source.source_sync_run_id is None:
                    raise HistoricalNavStorageError("SOURCE_NOT_READY", "来源尚无成功同步记录，不能保存。", 409)
                batch = HistoricalNavSampleBatch(
                    batch_id=uuid4(),
                    request_key=request.request_key,
                    fund_code=request.fund_code,
                    fund_type="STOCK",
                    start_date=request.start_date,
                    end_date=request.end_date,
                    source_code=source.source_code,
                    source_sync_run_id=source.source_sync_run_id,
                    feature_version=HISTORICAL_NAV_FEATURE_VERSION,
                    sample_rule_version=HISTORICAL_NAV_SAMPLE_RULE_VERSION,
                    label_version=HISTORICAL_NAV_LABEL_VERSION,
                    purpose="LEARNING_ONLY",
                    sample_count=preview.sample_count,
                    scorable_count=preview.scorable_count,
                    data_insufficient_count=preview.data_insufficient_count,
                    label_not_matured_count=preview.label_not_matured_count,
                    unavailable_reasons=preview.unavailable_reasons,
                )
                validate_batch_samples(batch, preview.items)
                sample_rows, label_rows = [], []
                for sample in preview.items:
                    sample_id = uuid4()
                    sample_rows.append(
                        {
                            "sample_id": sample_id,
                            "batch_id": batch.batch_id,
                            "as_of_date": sample.as_of_date,
                            "available_at": sample.available_at,
                            "nav_value_basis": sample.nav_value_basis,
                            "eligibility_status": sample.eligibility_status,
                            "unavailable_reason": sample.unavailable_reason,
                            "feature_payload": sample.feature_payload,
                            "feature_hash": sample.feature_hash,
                        }
                    )
                    if sample.offline_label is not None:
                        answer = asdict(sample.offline_label)
                        answer.pop("label_version")  # 全批共享的答案规则已写在封面。
                        label_rows.append({"sample_id": sample_id, **answer})
                _check_deadline(deadline)
                insert_batch_rows(session, batch, sample_rows, label_rows)
                # 从刚写入的三表重组返回并校验，不把内存预览冒充已存结果。
                result = _read_result(session, batch)
                _check_deadline(deadline)
                # return离开with前先提交；提交失败不会给HTTP调用者返回成功。
                return result, True
        except DBAPIError as error:
            if attempt == 1 or not _is_retryable_conflict(error):
                raise
    raise RuntimeError("unreachable storage retry state")


def get_historical_nav_batch(batch_id: UUID) -> HistoricalNavStoredBatch:
    """只读已存快照；不重读净值、不重新计算，也不要求历史来源现在仍启用。"""
    with Session(get_nav_sample_storage_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        batch = get_batch(session, batch_id)
        if batch is None:
            raise HistoricalNavStorageError("BATCH_NOT_FOUND", "没有找到该样本批次。", 404)
        return _read_result(session, batch)


def _read_result(session: Session, batch: HistoricalNavSampleBatch) -> HistoricalNavStoredBatch:
    """从封面、题目和可选答案恢复与单日预览相同的样本对象，并核对保存完整性。"""
    return restore_stored_batch(batch, read_sample_rows(session, batch.batch_id))


def restore_stored_batch(
    batch: HistoricalNavSampleBatch,
    rows: Sequence[tuple[HistoricalNavSampleRecord, HistoricalNavSampleLabel | None]],
) -> HistoricalNavStoredBatch:
    """复用保存/读回校验；批量数据准备可传预加载行，避免逐批查询造成N+1。"""
    samples = []
    for row, answer in rows:
        label = (
            None
            if answer is None
            else OfflineDirectionLabel(
                label_version=batch.label_version,
                horizon_trading_days=answer.horizon_trading_days,
                label_end_date=answer.label_end_date,
                label_available_at=answer.label_available_at,
                future_return_20d=answer.future_return_20d,
                label_up_20d=answer.label_up_20d,
            )
        )
        samples.append(
            HistoricalNavSample(
                fund_code=batch.fund_code,
                as_of_date=row.as_of_date,
                available_at=row.available_at,
                nav_value_basis=row.nav_value_basis,
                feature_version=batch.feature_version,
                sample_rule_version=batch.sample_rule_version,
                eligibility_status=row.eligibility_status,
                unavailable_reason=row.unavailable_reason,
                feature_payload=row.feature_payload,
                feature_hash=row.feature_hash,
                offline_label=label,
            )
        )
    try:
        validate_batch_samples(batch, tuple(samples))
        return HistoricalNavStoredBatch(
            **{column.name: getattr(batch, column.name) for column in HistoricalNavSampleBatch.__table__.columns},
            items=tuple(samples),
        )
    except (ValueError, ArithmeticError, TypeError, KeyError) as error:
        raise HistoricalNavStorageError(
            "STORED_BATCH_INCONSISTENT",
            "已存批次完整性校验失败，请联系维护人员；不会自动重算或覆盖。",
            503,
        ) from error
