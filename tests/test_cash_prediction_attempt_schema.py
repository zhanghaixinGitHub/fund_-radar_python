"""冻结迁移/ORM一致性及单一迁移链；不连接数据库。"""

from alembic.script import ScriptDirectory
from app.models.cash_prediction_attempt import CashPredictionAttemptRecord
from tests.test_cash_prediction_attempt_postgres import migration_module
from tests.test_historical_nav_storage_schema import ROOT, _checks, _column_contract, _SchemaRecorder, _unique_columns


def test_attempt_migration_matches_orm_and_follows_cash_storage():
    class Recorder(_SchemaRecorder):
        def create_index(self, name, table_name, columns, *, unique=False):
            return super().create_index(name, table_name, columns, unique=unique)

    migration = migration_module()
    assert migration.down_revision == "20260908_14"
    assert ScriptDirectory(str(ROOT / "alembic")).get_heads() == ["20260909_17"]
    recorder = Recorder()
    migration.op = recorder
    migration.upgrade()
    table = CashPredictionAttemptRecord.__table__
    frozen = recorder.metadata.tables[table.name]
    assert recorder.created_names == [table.name]
    assert _column_contract(frozen) == _column_contract(table)
    assert _checks(frozen) == _checks(table)
    assert _unique_columns(frozen) == _unique_columns(table)
    assert frozen.comment == table.comment
    assert all(c.comment for c in table.columns)
    assert tuple(frozen.primary_key.columns.keys()) == tuple(table.primary_key.columns.keys())
    assert {tuple(e.target_fullname for e in fk.elements) for fk in frozen.foreign_key_constraints} == {
        tuple(e.target_fullname for e in fk.elements) for fk in table.foreign_key_constraints
    }
    assert {(i.name, tuple(i.columns.keys()), i.unique) for i in frozen.indexes} == {
        (i.name, tuple(i.columns.keys()), i.unique) for i in table.indexes
    }
