"""新版迁移冻结定义与ORM契约一致性，避免未来实体漂移改变已执行迁移。"""

import pytest
from alembic.script import ScriptDirectory
from app.models.cash_reinvestment import CashResearchRun, CashSampleBatch, CashSampleLabel, CashSampleRecord
from tests.test_cash_reinvestment_postgres import migration_module
from tests.test_historical_nav_storage_schema import ROOT, _checks, _column_contract, _SchemaRecorder, _unique_columns

TABLES = (CashSampleBatch.__table__, CashSampleRecord.__table__, CashSampleLabel.__table__, CashResearchRun.__table__)


def test_cash_migration_is_single_head():
    script = ScriptDirectory(str(ROOT / "alembic"))
    assert script.get_heads() == ["20260908_14"]
    assert migration_module().down_revision == "20260907_13"


def test_orm_matches_frozen_migration():
    class Recorder(_SchemaRecorder):
        def create_index(self, name, table_name, columns, *, unique=False):
            return super().create_index(name, table_name, columns, unique=unique)

    recorder = Recorder()
    migration = migration_module()
    migration.op = recorder
    migration.upgrade()
    assert recorder.created_names == [t.name for t in TABLES]
    for table in TABLES:
        frozen = recorder.metadata.tables[table.name]
        assert _column_contract(frozen) == _column_contract(table)
        assert _checks(frozen) == _checks(table)
        assert _unique_columns(frozen) == _unique_columns(table)
        assert frozen.comment == table.comment
        assert tuple(frozen.primary_key.columns.keys()) == tuple(table.primary_key.columns.keys())
        assert {tuple(e.target_fullname for e in fk.elements) for fk in frozen.foreign_key_constraints} == {
            tuple(e.target_fullname for e in fk.elements) for fk in table.foreign_key_constraints
        }
        assert {(i.name, tuple(i.columns.keys()), i.unique) for i in frozen.indexes} == {
            (i.name, tuple(i.columns.keys()), i.unique) for i in table.indexes
        }


@pytest.mark.parametrize("table", TABLES, ids=lambda t: t.name)
def test_all_comments_and_keys(table):
    assert table.comment
    assert all(c.comment for c in table.columns)
    assert all(not c.nullable for c in table.primary_key.columns)
