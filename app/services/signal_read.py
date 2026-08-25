"""将已持久化的评分结果映射为 Java 到 Python 的内部读取契约。"""

from uuid import UUID

from sqlalchemy.orm import Session

from app.db.session import get_engine
from app.repositories.signal_read import list_signal_results_for_fund
from app.schemas.signal import InternalSignalPage, InternalSignalSummary


def list_completed_signals(fund_code: str, page_size: int, cursor: UUID | None) -> InternalSignalPage:
    """只返回本地已持久化的可复现结果；处理请求期间绝不运行模型。"""
    with Session(get_engine()) as session:
        rows = list_signal_results_for_fund(session, fund_code, page_size, cursor)
    page_rows = rows[:page_size]
    return InternalSignalPage(
        items=tuple(
            InternalSignalSummary(
                forecast_id=result.forecast_id,
                fund_code=result.fund_code,
                as_of_date=result.as_of_date,
                score_status=result.score_status,
                direction=result.direction,
                directional_probability=result.directional_probability,
                confidence=result.confidence,
                risk_level=result.risk_level,
                max_drawdown_estimate=result.max_drawdown_estimate,
                explanation=result.explanation,
                model_version=result.model_version,
                feature_version=result.feature_version,
                feature_completeness=feature.completeness,
                scored_at=result.scored_at,
            )
            for result, feature in page_rows
        ),
        next_cursor=str(page_rows[-1][0].forecast_id) if len(rows) > page_size else None,
    )
