"""检查某研究截点前有没有本地写入观察；不追认历史首次版本、不读取行情数值。"""

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.db.session import get_nav_preview_engine
from app.models.fund import FundShareClass, SourceRegistry
from app.repositories.cash_source_observation import observation_counts
from app.schemas.cash_source_observation import CashObservationCheck, CashObservationCheckRequest
from app.services.historical_nav_storage import HistoricalNavStorageError


def check_cash_source_observations(request: CashObservationCheckRequest) -> CashObservationCheck:
    # 服务内调用同样校验，不能用model_copy跳过保护期或把日期字符串塞进数据库。
    request = CashObservationCheckRequest.model_validate_json(request.model_dump_json())
    cutoff_end = datetime.combine(request.cutoff_date + timedelta(days=1), time(), tzinfo=ZoneInfo("Asia/Shanghai"))
    with Session(get_nav_preview_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        source_id = session.scalar(
            select(SourceRegistry.source_id)
            .join(
                FundShareClass,
                FundShareClass.source_code == SourceRegistry.source_code,
            )
            .where(
                FundShareClass.fund_code == request.fund_code,
                FundShareClass.fund_type == "STOCK",
                FundShareClass.status == "ACTIVE",
                SourceRegistry.source_code == "TUSHARE_PRO_FUND",
                SourceRegistry.enabled,
            )
        )
        if source_id is None:
            raise HistoricalNavStorageError("CASH_SOURCE_UNAVAILABLE", "试点基金或来源未启用，未检查观察记录。", 409)
        checked_at = session.scalar(select(func.clock_timestamp()))
        counts = observation_counts(
            session, source_id=source_id, request=request, checked_at=checked_at, cutoff_end=cutoff_end
        )
        status = (
            "NO_LOCAL_RECORDS"
            if not counts["active_record_count"]
            else "ONLY_AFTER_CUTOFF"
            if not counts["recorded_before_cutoff_count"]
            else "LOCAL_WRITE_RECORDS_ONLY"
        )
        return CashObservationCheck(request=request, checked_at=checked_at, status=status, **counts)
