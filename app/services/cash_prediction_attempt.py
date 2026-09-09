"""将真实闸门拒绝保存为可追溯回执；当前协议不执行推理或产生产品预测。"""

from datetime import UTC, datetime
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from app.db.session import get_nav_sample_storage_engine
from app.models.cash_prediction_attempt import CashPredictionAttemptRecord
from app.repositories.cash_prediction_attempt import find_attempt
from app.schemas.cash_prediction_attempt import CashPredictionAttempt, CashPredictionAttemptRequest
from app.schemas.cash_prediction_check import CashPredictionCheck
from app.services.cash_prediction_check import check_cash_prediction_in_session
from app.services.cash_reinvestment_research import BLOCKERS
from app.services.cash_reinvestment_storage import cash_hash
from app.services.historical_nav_storage import HistoricalNavStorageError

ATTEMPT_VERSION = "CASH_GENERATION_ATTEMPT_V1"


def attempt_request_hash(request: CashPredictionAttemptRequest) -> str:
    return cash_hash({"version": ATTEMPT_VERSION, **request.model_dump(mode="json", exclude={"request_key"})})


def _validated_rejection(check: CashPredictionCheck) -> CashPredictionCheck:
    """当前研究协议没有正式放行资格；空原因或替换状态必须报错，不走隐式成功分支。"""
    check = CashPredictionCheck.model_validate_json(check.model_dump_json())
    if not set(BLOCKERS).issubset(check.blocking_codes) or len(set(check.blocking_codes)) != len(check.blocking_codes):
        raise ValueError("cash generation attempt lacks mandatory rejection evidence")
    return check


def restore_attempt(row: CashPredictionAttemptRecord | None) -> CashPredictionAttempt:
    """历史回执校验自己的内容，不重新查询/覆盖历史研究，也不声称当前仍有相同状态。"""
    if row is None:
        raise HistoricalNavStorageError("CASH_ATTEMPT_NOT_FOUND", "生成尝试回执不存在。", 404)
    try:
        request = CashPredictionAttemptRequest(
            request_key=row.request_key,
            fund_code=row.fund_code,
            cutoff_date=row.cutoff_date,
            research_run_id=row.research_run_id,
            expected_report_hash=row.report_hash,
        )
        check = _validated_rejection(CashPredictionCheck.model_validate(row.check_payload))
        if (
            (check.fund_code, check.research_run_id, check.report_hash)
            != (row.fund_code, row.research_run_id, row.report_hash)
            or row.request_hash != attempt_request_hash(request)
            or row.receipt_hash != cash_hash({"request_hash": row.request_hash, "check": check.model_dump(mode="json")})
        ):
            raise ValueError("cash generation attempt linkage or hash mismatch")
        return CashPredictionAttempt(
            attempt_id=row.attempt_id,
            request_key=row.request_key,
            cutoff_date=row.cutoff_date,
            created_at=row.created_at,
            check=check,
            receipt_hash=row.receipt_hash,
        )
    except (ValueError, TypeError, KeyError) as error:
        raise HistoricalNavStorageError("CASH_ATTEMPT_CORRUPTED", "生成尝试回执完整性校验失败。", 503) from error


def get_cash_prediction_attempt(attempt_id: UUID) -> CashPredictionAttempt:
    with Session(get_nav_sample_storage_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        return restore_attempt(find_attempt(session, attempt_id=attempt_id))


def save_cash_prediction_attempt(request: CashPredictionAttemptRequest) -> tuple[CashPredictionAttempt, bool]:
    """一致性事务检查并保存；唯一键竞争仅重试一次，重试不重算或修改既有回执。"""
    request = CashPredictionAttemptRequest.model_validate(request.model_dump())
    now = datetime.now(UTC)
    if request.cutoff_date >= now.astimezone(ZoneInfo("Asia/Shanghai")).date():
        raise HistoricalNavStorageError("CUTOFF_DAY_NOT_CLOSED", "信息截止日必须是已经结束的自然日。", 409)
    request_hash = attempt_request_hash(request)
    for retry in range(2):
        try:
            with Session(get_nav_sample_storage_engine()) as session, session.begin():
                session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"))
                existing = find_attempt(session, request_key=request.request_key)
                if existing is not None:
                    receipt = restore_attempt(existing)
                    if existing.request_hash != request_hash:
                        raise HistoricalNavStorageError(
                            "REQUEST_KEY_CONFLICT", "这个requestKey已用于另一份请求，请勿换参数重试。", 409
                        )
                    return receipt, False
                check = _validated_rejection(
                    check_cash_prediction_in_session(session, request.check_request(), now=now)
                )
                row = CashPredictionAttemptRecord(
                    attempt_id=uuid4(),
                    request_key=request.request_key,
                    request_hash=request_hash,
                    fund_code=request.fund_code,
                    cutoff_date=request.cutoff_date,
                    research_run_id=request.research_run_id,
                    report_hash=request.expected_report_hash,
                    check_payload=check.model_dump(mode="json"),
                    receipt_hash=cash_hash({"request_hash": request_hash, "check": check.model_dump(mode="json")}),
                )
                session.add(row)
                session.flush()
                return restore_attempt(row), True
        except DBAPIError as error:
            if retry or getattr(getattr(error.orig, "diag", None), "constraint_name", None) != (
                "uq_cash_prediction_attempt_request"
            ):
                raise
    raise RuntimeError("unreachable")
