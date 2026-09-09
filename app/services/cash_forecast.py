"""授权→已知输入→真实计算→原子保存；结果读取重新查授权/时效，不重新执行模型。"""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from time import perf_counter
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from app.db.session import get_nav_sample_storage_engine
from app.models.cash_forecast import CashForecastRecord
from app.repositories.cash_forecast import (
    current_cash_source_revision,
    find_forecast,
    find_identical_forecast,
    latest_announced_nav_date,
)
from app.repositories.feature_snapshot import FeatureSourceReadiness
from app.repositories.historical_nav import HistoricalNavPreviewReadError, read_historical_nav_source
from app.schemas.cash_forecast import CashAuthorizedModel, CashForecastRequest, CashForecastSnapshot, CashForecastView
from app.services.cash_prediction_features import prepare_cash_feature_read, read_cash_prediction_feature_in_session
from app.services.cash_prediction_inference import calculate_cash_inference
from app.services.cash_publication import (
    CashPublicationUnavailable,
    resolve_cash_authorization,
    validate_cash_authorization,
)
from app.services.cash_reinvestment_samples import _hash
from app.services.cash_reinvestment_storage import cash_hash
from app.services.historical_nav_storage import HistoricalNavStorageError
from app.services.trading_calendar import CalendarCoverageError, load_prediction_calendar


def utc_now() -> datetime:
    return datetime.now(UTC)


def forecast_request_hash(request: CashForecastRequest) -> str:
    return cash_hash(
        {"version": "CASH_FORECAST_STORAGE_V1", **request.model_dump(mode="json", exclude={"request_key"})}
    )


def restore_forecast(row: CashForecastRecord | None) -> CashForecastSnapshot:
    """验证快照完整性及各字段绑定；哈希不是签名，不能替代数据库写权限边界。"""
    if row is None:
        raise HistoricalNavStorageError("CASH_FORECAST_NOT_FOUND", "现金预测结果不存在。", 404)
    try:
        saved = CashForecastSnapshot.model_validate(row.payload)
        req, feature, value = saved.request, saved.feature, saved.value
        if (
            row.content_hash != cash_hash(saved.model_dump(mode="json"))
            or row.request_hash != forecast_request_hash(req)
            or (row.request_key, row.fund_code, row.cutoff_date, row.research_run_id, row.model_hash)
            != (req.request_key, req.fund_code, req.cutoff_date, req.research_run_id, req.expected_model_hash)
            or row.authorization_hash != saved.authorization_hash
            or row.source_revision_id != saved.source_revision_id
            or (feature.fund_code, feature.cutoff_date, feature.feature_hash)
            != (req.fund_code, req.cutoff_date, row.feature_hash)
            or feature.status != "INPUT_READY"
            or feature.input_issues
            or feature.feature_payload is None
            or feature.feature_hash != _hash(feature.feature_payload)
            or (value.fund_code, value.cutoff_date, value.feature_hash, value.model_hash, value.anchor_nav_date)
            != (req.fund_code, req.cutoff_date, row.feature_hash, row.model_hash, feature.anchor_nav_date)
            or value.predicted_up != (value.up_score > Decimal("0.5"))
        ):
            raise ValueError("cash forecast snapshot linkage or hash mismatch")
        calendar = load_prediction_calendar(req.cutoff_date)
        if (
            value.target_base_date != calendar.sessions[calendar.at_or_before_index(req.cutoff_date)]
            or value.target_end_date != calendar.future_sessions(req.cutoff_date, 20)[-1]
        ):
            raise ValueError("cash forecast target interval mismatch")
        return saved
    except (ValueError, TypeError, KeyError) as error:
        raise HistoricalNavStorageError("CASH_FORECAST_CORRUPTED", "现金预测快照完整性校验失败。", 503) from error


def cash_forecast_stale_reasons(
    saved: CashForecastSnapshot, source: FeatureSourceReadiness, latest_date: date | None, *, closed_day: date
) -> tuple[str, ...]:
    """只检查日期和同步水位；不通过读取时自动重算，把旧预测期限向后挪。"""
    feature, value = saved.feature.feature_payload, saved.value
    reasons = []
    if closed_day < saved.request.cutoff_date:
        reasons.append("FORECAST_CUTOFF_NOT_CLOSED")
    if closed_day >= value.target_end_date:
        reasons.append("FORECAST_WINDOW_ENDED")
    if source.source_code != feature.source_code or source.source_sync_run_id != feature.source_sync_run_id:
        reasons.append("SOURCE_REVISION_CHANGED")
    if latest_date != value.anchor_nav_date:
        reasons.append("LATEST_NAV_CHANGED")
    try:
        calendar = load_prediction_calendar(closed_day)
        if (
            latest_date is None
            or calendar.at_or_before_index(closed_day) - calendar.at_or_before_index(latest_date) > 1
        ):
            reasons.append("NAV_TOO_OLD")
    except CalendarCoverageError:
        reasons.append("CALENDAR_COVERAGE_INSUFFICIENT")
    return tuple(reasons)


def _view(
    session: Session,
    row: CashForecastRecord,
    *,
    now: datetime,
    created: bool = False,
    grant: CashAuthorizedModel | None = None,
) -> CashForecastView:
    saved = restore_forecast(row)
    req, value = saved.request, saved.value
    fields = dict(
        fund_code=req.fund_code,
        cutoff_date=req.cutoff_date,
        forecast_id=row.forecast_id,
        generated_at=row.created_at,
        target_base_date=value.target_base_date,
        target_end_date=value.target_end_date,
        model_hash=value.model_hash,
        created=created,
    )
    try:
        grant = validate_cash_authorization(grant or resolve_cash_authorization(session, req, now=now), req)
        if (saved.authorization_id, saved.authorization_hash) != (grant.authorization_id, grant.authorization_hash):
            raise CashPublicationUnavailable(("PUBLICATION_AUTHORIZATION_CHANGED",))
    except CashPublicationUnavailable as error:
        return CashForecastView(**fields, status="MODEL_NOT_RELEASED", reason_codes=error.codes)
    except HistoricalNavStorageError as error:
        if error.code not in {
            "MODEL_HASH_MISMATCH",
            "REPORT_HASH_MISMATCH",
            "CASH_AUTHORIZATION_MISMATCH",
            "CASH_SOURCE_UNAVAILABLE",
            "CASH_PREDICTION_NOT_APPLICABLE",
        }:
            raise
        return CashForecastView(**fields, status="MODEL_NOT_RELEASED", reason_codes=(error.code,))
    try:
        source = read_historical_nav_source(session, fund_code=req.fund_code)
    except HistoricalNavPreviewReadError as error:
        return CashForecastView(**fields, status="STALE", reason_codes=(error.code,))
    revision = current_cash_source_revision(session, fund_code=req.fund_code)
    if revision is None or revision != saved.source_revision_id:
        return CashForecastView(
            **fields,
            status="STALE",
            reason_codes=("SOURCE_UPDATE_NOT_READY" if revision is None else "SOURCE_REVISION_CHANGED",),
        )
    closed_day = now.astimezone(ZoneInfo("Asia/Shanghai")).date() - timedelta(days=1)
    latest = latest_announced_nav_date(
        session, fund_code=req.fund_code, source_id=source.source_id, closed_day=closed_day
    )
    reasons = cash_forecast_stale_reasons(saved, source, latest, closed_day=closed_day)
    if reasons:
        return CashForecastView(**fields, status="STALE", reason_codes=reasons)
    return CashForecastView(
        **fields, status="AVAILABLE", up_probability=value.up_score, direction="UP" if value.predicted_up else "NON_UP"
    )


def get_cash_forecast(forecast_id: UUID) -> CashForecastView:
    """GET只读已存快照与当前授权/来源元数据；不触发输入制作、模型计算或结果写入。"""
    with Session(get_nav_sample_storage_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        row = find_forecast(session, forecast_id=forecast_id)
        if row is None:
            raise HistoricalNavStorageError("CASH_FORECAST_NOT_FOUND", "现金预测结果不存在。", 404)
        return _view(session, row, now=utc_now())


def generate_cash_forecast(request: CashForecastRequest) -> CashForecastView:
    """失败关闭；当前真实授权解析器只会拒绝，因此不会在真实资料上先算分数再隐藏。"""
    request = CashForecastRequest.model_validate(request.model_dump())
    prepare_cash_feature_read(request.feature_request())  # 日期不合法/2025保护在任何数据库访问前拒绝。
    request_hash = forecast_request_hash(request)
    now = utc_now()
    for retry in range(2):
        try:
            with Session(get_nav_sample_storage_engine()) as session, session.begin():
                session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"))
                existing = find_forecast(session, request_key=request.request_key)
                if existing is not None:
                    restore_forecast(existing)
                    if existing.request_hash != request_hash:
                        raise HistoricalNavStorageError(
                            "REQUEST_KEY_CONFLICT", "该requestKey已用于另一份生成请求。", 409
                        )
                    return _view(session, existing, now=now)
                try:
                    grant = validate_cash_authorization(resolve_cash_authorization(session, request, now=now), request)
                except CashPublicationUnavailable as error:
                    return CashForecastView(
                        status="MODEL_NOT_RELEASED",
                        fund_code=request.fund_code,
                        cutoff_date=request.cutoff_date,
                        reason_codes=error.codes,
                    )
                revision = current_cash_source_revision(session, fund_code=request.fund_code)
                if revision is None:
                    return CashForecastView(
                        status="DATA_INSUFFICIENT",
                        fund_code=request.fund_code,
                        cutoff_date=request.cutoff_date,
                        reason_codes=("SOURCE_UPDATE_NOT_READY",),
                    )
                feature = read_cash_prediction_feature_in_session(
                    session, request.feature_request(), deadline=perf_counter() + 15
                )
                if feature.status != "INPUT_READY":
                    return CashForecastView(
                        status="DATA_INSUFFICIENT",
                        fund_code=request.fund_code,
                        cutoff_date=request.cutoff_date,
                        reason_codes=tuple(dict.fromkeys(i.code for i in feature.input_issues))
                        or ("INPUT_INCOMPLETE",),
                    )
                if feature.feature_payload.calendar_hash not in grant.calendar_hashes:
                    raise CashPublicationUnavailable(("CALENDAR_NOT_AUTHORIZED",))
                duplicate = find_identical_forecast(
                    session,
                    fund_code=request.fund_code,
                    cutoff=request.cutoff_date,
                    authorization_hash=grant.authorization_hash,
                    feature_hash=feature.feature_hash,
                    source_revision_id=revision,
                )
                if duplicate is not None:
                    raise HistoricalNavStorageError(
                        "CASH_FORECAST_ALREADY_EXISTS", "相同输入与授权已有结果，请使用原requestKey或查询原结果。", 409
                    )
                value = calculate_cash_inference(
                    feature, grant.artifact, expected_model_hash=request.expected_model_hash
                )
                snapshot = CashForecastSnapshot(
                    request=request,
                    authorization_id=grant.authorization_id,
                    authorization_hash=grant.authorization_hash,
                    feature=feature,
                    source_revision_id=revision,
                    value=value,
                )
                row = CashForecastRecord(
                    forecast_id=uuid4(),
                    request_key=request.request_key,
                    request_hash=request_hash,
                    fund_code=request.fund_code,
                    cutoff_date=request.cutoff_date,
                    research_run_id=request.research_run_id,
                    authorization_hash=grant.authorization_hash,
                    model_hash=value.model_hash,
                    feature_hash=value.feature_hash,
                    source_revision_id=revision,
                    payload=snapshot.model_dump(mode="json"),
                    content_hash=cash_hash(snapshot.model_dump(mode="json")),
                )
                session.add(row)
                session.flush()
                return _view(session, row, now=now, created=True, grant=grant)
        except CashPublicationUnavailable as error:
            return CashForecastView(
                status="MODEL_NOT_RELEASED",
                fund_code=request.fund_code,
                cutoff_date=request.cutoff_date,
                reason_codes=error.codes,
            )
        except DBAPIError as error:
            constraint = getattr(getattr(error.orig, "diag", None), "constraint_name", None)
            if constraint == "uq_cash_forecast_business":
                raise HistoricalNavStorageError(
                    "CASH_FORECAST_ALREADY_EXISTS", "同一输入已有计算结果，请查询原记录。", 409
                ) from error
            if retry or constraint != "uq_cash_forecast_request":
                raise
    raise RuntimeError("unreachable")
