"""版本观察表/触发器的迁移契约；不导入历史源值或写真实数据库。"""

import importlib.util

from alembic.script import ScriptDirectory
from app.models.cash_source_observation import CashSourceObservation
from tests.test_historical_nav_storage_schema import ROOT, _checks, _SchemaRecorder


def migration_module():
    spec = importlib.util.spec_from_file_location(
        "cash_observation_migration", ROOT / "alembic/versions/20260909_19_cash_source_observation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_and_migration_match_and_never_backfill():
    class Recorder(_SchemaRecorder):
        def __init__(self):
            super().__init__()
            self.sql = []

        def execute(self, sql):
            self.sql.append(sql)

    migration = migration_module()
    assert migration.down_revision == "20260909_18"
    assert ScriptDirectory(str(ROOT / "alembic")).get_heads() == ["20260909_19"]
    recorder = Recorder()
    migration.op = recorder
    migration.upgrade()
    table = CashSourceObservation.__table__
    frozen = recorder.metadata.tables[table.name]
    assert recorder.created_names == [table.name]

    def contract(table):
        return [(c.name, str(c.type), c.nullable, c.comment, c.identity is not None) for c in table.columns]

    assert contract(table) == contract(frozen) and _checks(table) == _checks(frozen)
    assert all(c.comment for c in table.columns) and table.comment == frozen.comment
    assert {tuple(c.name for c in index.columns) for index in table.indexes} == {
        tuple(c.name for c in index.columns) for index in frozen.indexes
    }
    assert all(not sql.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")) for sql in recorder.sql)
