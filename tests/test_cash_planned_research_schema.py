"""新表字段/约束与冻结迁移一致，旧迁移保持独立，不连接数据库。"""

import importlib.util

from alembic.script import ScriptDirectory
from app.models.cash_planned_research import CashPlannedResearchBinding
from tests.test_historical_nav_storage_schema import ROOT, _checks, _column_contract, _SchemaRecorder, _unique_columns


def migration_module():
    spec = importlib.util.spec_from_file_location(
        "cash_planned_migration", ROOT / "alembic/versions/20260909_18_cash_planned_research.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_planned_binding_migration_and_model_are_identical():
    class Recorder(_SchemaRecorder):
        def __init__(self):
            super().__init__()
            self.sql = []

        def execute(self, sql):
            self.sql.append(sql)

    migration = migration_module()
    assert migration.down_revision == "20260909_17"
    # 后续独立模块可追加迁移；仍要求唯一主线，并保留本迁移自身的上下游断言。
    assert len(ScriptDirectory(str(ROOT / "alembic")).get_heads()) == 1
    recorder = Recorder()
    migration.op = recorder
    migration.upgrade()
    table = CashPlannedResearchBinding.__table__
    frozen = recorder.metadata.tables[table.name]
    assert recorder.created_names == [table.name]
    assert _column_contract(frozen) == _column_contract(table)
    assert _checks(frozen) == _checks(table) and _unique_columns(frozen) == _unique_columns(table)
    assert table.comment == frozen.comment and all(c.comment for c in table.columns)
    assert {tuple(e.target_fullname for e in fk.elements) for fk in frozen.foreign_key_constraints} == {
        tuple(e.target_fullname for e in fk.elements) for fk in table.foreign_key_constraints
    }
    assert "BEFORE UPDATE OR DELETE OR TRUNCATE" in recorder.sql[-1]
