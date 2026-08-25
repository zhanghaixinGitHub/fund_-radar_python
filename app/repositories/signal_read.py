"""AI 服务拥有的可复现评分结果只读查询。"""

from uuid import UUID

from sqlalchemy import Select, and_, or_, select
from sqlalchemy.orm import Session

from app.models.analysis import FeatureSnapshot, ForecastResult


def list_signal_results_for_fund(
    session: Session, fund_code: str, page_size: int, cursor: UUID | None
) -> tuple[tuple[ForecastResult, FeatureSnapshot], ...]:
    """查询指定基金的评分结果与特征快照，并按稳定顺序返回一页加一条探测记录。"""
    statement: Select[tuple[ForecastResult, FeatureSnapshot]] = (
        select(ForecastResult, FeatureSnapshot)
        .join(FeatureSnapshot, ForecastResult.feature_id == FeatureSnapshot.feature_id)
        .where(ForecastResult.fund_code == fund_code)
        .order_by(ForecastResult.as_of_date.desc(), ForecastResult.scored_at.desc(), ForecastResult.forecast_id.desc())
    )
    if cursor is not None:
        cursor_result = session.get(ForecastResult, cursor)
        if cursor_result is None:
            raise ValueError("Signal cursor does not exist.")
        statement = statement.where(
            or_(
                ForecastResult.as_of_date < cursor_result.as_of_date,
                and_(
                    ForecastResult.as_of_date == cursor_result.as_of_date,
                    ForecastResult.scored_at < cursor_result.scored_at,
                ),
                and_(
                    ForecastResult.as_of_date == cursor_result.as_of_date,
                    ForecastResult.scored_at == cursor_result.scored_at,
                    ForecastResult.forecast_id < cursor_result.forecast_id,
                ),
            )
        )
    return tuple(session.execute(statement.limit(page_size + 1)).all())
