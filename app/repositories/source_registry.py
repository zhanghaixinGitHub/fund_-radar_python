"""AI 服务拥有的数据源登记表只读访问。"""

from collections.abc import Sequence

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.models.fund import SourceRegistry


def list_sources(session: Session) -> Sequence[SourceRegistry]:
    """按数据源编码稳定排序返回登记元数据，不访问任何外部系统。"""
    statement: Select[tuple[SourceRegistry]] = select(SourceRegistry).order_by(SourceRegistry.source_code)
    return session.scalars(statement).all()
