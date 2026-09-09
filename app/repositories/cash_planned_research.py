"""只按唯一身份读取新研究绑定，不允许批量改写或关联任意旧报告。"""

from sqlalchemy import select

from app.models.cash_planned_research import CashPlannedResearchBinding


def find_planned_research(session, *, binding_id=None, request_key=None):
    if (binding_id is None) == (request_key is None):
        raise ValueError("exactly one planned research identity required")
    column, value = (
        (CashPlannedResearchBinding.binding_id, binding_id)
        if binding_id is not None
        else (CashPlannedResearchBinding.request_key, request_key)
    )
    return session.scalar(select(CashPlannedResearchBinding).where(column == value))
