"""冻结迁移与实体逐项一致，不把静态检查冒充实库验收。"""

from alembic.script import ScriptDirectory
from app.models.cash_forecast import CashForecastRecord
from tests.test_cash_forecast_postgres import migration_module
from tests.test_historical_nav_storage_schema import ROOT, _checks, _column_contract, _SchemaRecorder, _unique_columns


def test_forecast_migration_matches_orm():
    class Recorder(_SchemaRecorder):
        def create_index(self, name, table_name, columns, *, unique=False):
            return super().create_index(name, table_name, columns, unique=unique)

    migration = migration_module()
    assert migration.down_revision == "20260909_15"
    assert ScriptDirectory(str(ROOT / "alembic")).get_heads() == [migration.revision]
    recorder = Recorder()
    migration.op = recorder
    migration.upgrade()
    table = CashForecastRecord.__table__
    frozen = recorder.metadata.tables[table.name]
    assert recorder.created_names == [table.name] and frozen.comment == table.comment
    assert _column_contract(frozen) == _column_contract(table)
    assert _checks(frozen) == _checks(table) and _unique_columns(frozen) == _unique_columns(table)
    assert tuple(frozen.primary_key.columns.keys()) == tuple(table.primary_key.columns.keys())
    assert {tuple(e.target_fullname for e in fk.elements) for fk in frozen.foreign_key_constraints} == {
        tuple(e.target_fullname for e in fk.elements) for fk in table.foreign_key_constraints
    }
    assert {(i.name, tuple(i.columns.keys()), i.unique) for i in frozen.indexes} == {
        (i.name, tuple(i.columns.keys()), i.unique) for i in table.indexes
    }
