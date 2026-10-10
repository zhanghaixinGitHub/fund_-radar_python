"""保留完整核验档案，同时避免把重复审计索引塞入面向页面的预测正文。"""

AUDIT_FIELDS = frozenset({"quote_receipts", "matters", "member_event_ids"})


def public_evidence(facts: dict) -> dict:
    """只投影展示不需要的重复索引；逐字原文、公司关系、来源和事实身份全部保留。

    调用方必须先将完整 facts 写入不可变快照，并在结果中保留快照身份与哈希。
    不修改传入对象，不截断引文，不删除失败信息，不改变方向或分析结论。
    """
    return {ref: {key: value for key, value in fact.items() if key not in AUDIT_FIELDS} for ref, fact in facts.items()}
