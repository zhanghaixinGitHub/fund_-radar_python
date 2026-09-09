"""只按唯一身份读取新研究绑定，不允许批量改写或关联任意旧报告。"""

from sqlalchemy import select

from app.models.cash_planned_research import CashPlannedResearchBinding


def find_planned_research(session, *, binding_id=None, request_key=None, research_run_id=None):
    identities = (
        (CashPlannedResearchBinding.binding_id, binding_id),
        (CashPlannedResearchBinding.request_key, request_key),
        (CashPlannedResearchBinding.research_run_id, research_run_id),
    )
    selected = [(column, value) for column, value in identities if value is not None]
    if len(selected) != 1:
        raise ValueError("exactly one planned research identity required")
    column, value = selected[0]
    return session.scalar(select(CashPlannedResearchBinding).where(column == value))
