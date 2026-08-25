"""安全、只读的数据源治理诊断服务层。"""

from sqlalchemy.orm import Session

from app.db.session import get_engine
from app.repositories.source_registry import list_sources
from app.schemas.source import SourceDiagnostic


def list_source_diagnostics() -> tuple[SourceDiagnostic, ...]:
    """将持久化的数据源状态映射为不含凭据的内部诊断响应。"""
    with Session(get_engine()) as session:
        sources = list_sources(session)
        return tuple(
            SourceDiagnostic(
                source_code=source.source_code,
                display_name=source.display_name,
                source_kind=source.source_kind,
                license_scope=source.license_scope,
                rate_limit_per_minute=source.rate_limit_per_minute,
                retention_days=source.retention_days,
                enabled=source.enabled,
                last_success_at=source.last_success_at,
                last_error_at=source.last_error_at,
                last_error_summary=source.last_error_summary,
            )
            for source in sources
        )
