"""仅按唯一编号/请求号/版本读取一份快照；没有UPDATE或DELETE仓储操作。"""

from sqlalchemy import select

from app.models.cash_policy_freeze import CashPolicyFreezeRecord


def find_policy_freeze(session, *, freeze_id=None, request_key=None, policy_version=None):
    values = [
        (CashPolicyFreezeRecord.freeze_id, freeze_id),
        (CashPolicyFreezeRecord.request_key, request_key),
        (CashPolicyFreezeRecord.policy_version, policy_version),
    ]
    selected = [(column, value) for column, value in values if value is not None]
    if len(selected) != 1:
        raise ValueError("exactly one policy identity required")
    column, value = selected[0]
    return session.scalar(select(CashPolicyFreezeRecord).where(column == value))
