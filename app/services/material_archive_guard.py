"""资料迁移进程的数据库写入边界，拒绝修改评级、预测、快照及实验结果。"""

from __future__ import annotations

import re
from contextlib import contextmanager

from sqlalchemy import event

ALLOWED = {"material_extracted_content", "material_source_location", "material_page_representation",
           "material_document_content"}


@contextmanager
def material_writes_only(engine):
    """仅允许四张资料表新增/更新，禁止删除和DDL；计数不包含任何业务写入。"""
    writes = {}

    def guard(connection, cursor, statement, parameters, context, many):
        match = re.match(r'\s*(INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s+(?:public\.)?"?([a-z_]+)', statement, re.I)
        if match:
            name = match.group(2).lower()
            if name not in ALLOWED or match.group(1).upper().startswith("DELETE"):
                raise RuntimeError("MATERIAL_BATCH_BUSINESS_WRITE_FORBIDDEN:" + name)
            writes[name] = writes.get(name, 0) + 1
        elif statement.lstrip().split()[0].upper() in {"DROP", "TRUNCATE", "ALTER", "CREATE", "MERGE", "CALL", "DO"}:
            raise RuntimeError("MATERIAL_BATCH_BUSINESS_WRITE_FORBIDDEN:DDL_OR_PROCEDURE")

    event.listen(engine, "before_cursor_execute", guard)
    try:
        yield writes
    finally:
        event.remove(engine, "before_cursor_execute", guard)
