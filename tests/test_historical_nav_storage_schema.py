"""学习样本存储的离线契约测试：不连接项目数据库，不执行真实建表或保存。

本文件核对 ORM、Alembic 和原生 SQL 三种表达是否一致。它不冒充 PostgreSQL 集成测试：
真实数据库对非法 INSERT 的拒绝、事务回滚和并发幂等，要在后续隔离测试库中另行验收。
"""

from __future__ import annotations

import importlib.util
from io import StringIO
from pathlib import Path
from types import ModuleType

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from app.db.base import Base
from app.models import HistoricalNavSampleBatch, HistoricalNavSampleLabel, HistoricalNavSampleRecord
from sqlalchemy import CheckConstraint, Column, Index, MetaData, Numeric, String, Table, UniqueConstraint
from sqlalchemy.dialects import postgresql

ROOT = Path(__file__).resolve().parents[1]
REVISION = "20260907_13"
MIGRATION_FILE = ROOT / "alembic/versions/20260907_13_add_historical_nav_sample_storage.py"
TABLES = (HistoricalNavSampleBatch.__table__, HistoricalNavSampleRecord.__table__, HistoricalNavSampleLabel.__table__)


def load_migration() -> ModuleType:
    """只加载本次迁移文件；不加载项目数据库配置或打开连接。"""
    spec = importlib.util.spec_from_file_location("historical_nav_storage_migration", MIGRATION_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_migration_sql(direction: str) -> str:
    """使用 PostgreSQL 方言把迁移动作打印为 SQL，绝不执行这些 SQL。"""
    migration = load_migration()
    buffer = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql",
        # 与 alembic/env.py 离线配置一致；原生脚本不经驱动，不应把百分号写成两个。
        dialect_opts={"paramstyle": "named"},
        opts={"as_sql": True, "output_buffer": buffer, "transactional_ddl": True},
    )
    migration.op = Operations(context)
    with context.begin_transaction():
        getattr(migration, direction)()
    # 只去掉生成器在行尾附带的空格，不改SQL内容，便于审阅及Git空白检查。
    return "\n".join(line.rstrip() for line in buffer.getvalue().splitlines()) + "\n"


class _SchemaRecorder:
    """将 create_table/create_index 记成内存元数据，供迁移与 ORM 双向核对。"""

    def __init__(self) -> None:
        self.metadata = MetaData()
        # 仅给外键提供目标声明，不复制或访问真实基金档案、来源运行表。
        Table("fund_share_class", self.metadata, Column("fund_code", String(32), primary_key=True))
        Table("source_sync_run", self.metadata, Column("sync_run_id", postgresql.UUID(), primary_key=True))
        self.created_names: list[str] = []

    def create_table(self, name: str, *items: object, **options: object) -> None:
        Table(name, self.metadata, *items, **options)
        self.created_names.append(name)

    def create_index(self, name: str, table_name: str, columns: list[str], *, unique: bool) -> None:
        table = self.metadata.tables[table_name]
        Index(name, *(table.c[column] for column in columns), unique=unique)


def _checks(table: Table) -> dict[str, str]:
    return {c.name: str(c.sqltext) for c in table.constraints if isinstance(c, CheckConstraint)}


def _unique_columns(table: Table) -> set[tuple[str, ...]]:
    return {tuple(c.columns.keys()) for c in table.constraints if isinstance(c, UniqueConstraint)}


def _column_contract(table: Table) -> list[tuple]:
    """包括类型、小数精度、空值、默认值和中文注释，不只比较列名。"""
    dialect = postgresql.dialect()
    return [
        (
            c.name,
            str(c.type.compile(dialect=dialect)),
            c.nullable,
            str(c.server_default.arg) if c.server_default is not None else None,
            c.comment,
        )
        for c in table.columns
    ]


def test_migration_is_the_only_head_and_follows_schema_comments() -> None:
    """接在已发布的注释迁移之后，不修改旧版本，不产生迁移分叉。"""
    script = ScriptDirectory(str(ROOT / "alembic"))
    assert script.get_heads() == [REVISION]
    assert load_migration().down_revision == "20260905_12"


def test_migration_and_orm_have_identical_contracts() -> None:
    migration = load_migration()
    recorder = _SchemaRecorder()
    migration.op = recorder
    migration.upgrade()
    assert recorder.created_names == [t.name for t in TABLES]
    for model_table in TABLES:
        migrated = recorder.metadata.tables[model_table.name]
        assert _column_contract(migrated) == _column_contract(model_table)
        assert migrated.comment == model_table.comment
        assert _checks(migrated) == _checks(model_table)
        assert _unique_columns(migrated) == _unique_columns(model_table)
        assert tuple(migrated.primary_key.columns.keys()) == tuple(model_table.primary_key.columns.keys())
        assert {
            (fk.name, tuple(e.target_fullname for e in fk.elements), fk.ondelete)
            for fk in migrated.foreign_key_constraints
        } == {
            (fk.name, tuple(e.target_fullname for e in fk.elements), fk.ondelete)
            for fk in model_table.foreign_key_constraints
        }
        assert {(i.name, tuple(i.columns.keys()), i.unique) for i in migrated.indexes} == {
            (i.name, tuple(i.columns.keys()), i.unique) for i in model_table.indexes
        }


@pytest.mark.parametrize("table", TABLES, ids=lambda t: t.name)
def test_registered_tables_and_all_columns_have_chinese_comments(table: Table) -> None:
    assert Base.metadata.tables[table.name] is table
    for comment in [table.comment, *(column.comment for column in table.columns)]:
        assert comment and any("\u4e00" <= char <= "\u9fff" for char in comment)
    for column in table.primary_key.columns:
        assert not column.nullable


def test_batch_enforces_learning_scope_bounded_counts_and_request_identity() -> None:
    table = HistoricalNavSampleBatch.__table__
    checks = _checks(table)
    assert checks["ck_historical_nav_batch_purpose"] == "purpose = 'LEARNING_ONLY'"
    assert checks["ck_historical_nav_batch_fund_type"] == "fund_type = 'STOCK'"
    assert checks["ck_historical_nav_batch_window"] == "end_date - start_date BETWEEN 0 AND 30"
    counts = checks["ck_historical_nav_batch_counts"]
    assert "sample_count <= end_date - start_date + 1" in counts
    for name in ("sample_count", "scorable_count", "data_insufficient_count", "label_not_matured_count"):
        assert f"{name} >= 0" in counts
    assert "sample_count = scorable_count + data_insufficient_count + label_not_matured_count" in counts
    assert ("request_key",) in _unique_columns(table)
    assert table.c.request_key.default is None  # 不能在每次重试时悄悄换新凭证。
    assert "page_size" not in table.c and "page_count" not in table.c
    assert [c.name for c in next(iter(table.indexes)).columns] == ["fund_code", "created_at", "batch_id"]


def test_rejected_samples_can_keep_missing_or_invalid_announcement_facts() -> None:
    table = HistoricalNavSampleRecord.__table__
    checks = _checks(table)
    assert ("batch_id", "as_of_date") in _unique_columns(table)
    assert table.c.available_at.nullable
    assert checks["ck_historical_nav_sample_available"].startswith("eligibility_status = 'DATA_INSUFFICIENT' OR ")
    assert "'LABEL_NOT_MATURED'" in checks["ck_historical_nav_sample_status"]
    assert "unavailable_reason IS NOT NULL" in checks["ck_historical_nav_sample_reason"]
    # 只约束 JSON 顶层为对象，不强迫拒收样本丢掉已经合法算出的指标。
    assert checks["ck_historical_nav_sample_payload"] == "jsonb_typeof(feature_payload) = 'object'"
    assert checks["ck_historical_nav_sample_hash"] == "feature_hash ~ '^[0-9a-f]{64}$'"


def test_answers_are_separate_optional_rows_with_one_to_one_identity() -> None:
    sample, label = HistoricalNavSampleRecord.__table__, HistoricalNavSampleLabel.__table__
    answer_fields = {"future_return_20d", "label_up_20d", "label_end_date", "label_available_at"}
    assert not answer_fields.intersection(sample.c.keys())
    assert answer_fields.issubset(label.c.keys())
    assert list(label.primary_key.columns.keys()) == ["sample_id"]
    assert {fk.target_fullname for fk in label.foreign_keys} == {"historical_nav_sample.sample_id"}
    assert all(not c.nullable for c in label.columns)
    assert label.c.label_up_20d.default is None and label.c.label_up_20d.server_default is None
    assert all(fk.ondelete is None for t in TABLES for fk in t.foreign_key_constraints)


def test_label_keeps_decimal_precision_and_declares_real_answer_rules() -> None:
    label = HistoricalNavSampleLabel.__table__
    numeric = label.c.future_return_20d.type
    assert isinstance(numeric, Numeric) and numeric.asdecimal
    assert numeric.precision is None and numeric.scale is None
    checks = _checks(label)
    assert checks["ck_historical_nav_label_horizon"] == "horizon_trading_days = 20"
    assert checks["ck_historical_nav_label_available"] == "label_available_at >= label_end_date"
    assert "'NaN', 'Infinity', '-Infinity'" in checks["ck_historical_nav_label_return"]
    assert "future_return_20d > -1" in checks["ck_historical_nav_label_return"]
    assert checks["ck_historical_nav_label_direction"] == (
        "(future_return_20d > 0 AND label_up_20d = 1) OR (future_return_20d <= 0 AND label_up_20d = 0)"
    )


@pytest.mark.parametrize("direction", ["upgrade", "downgrade"])
def test_native_sql_exactly_matches_offline_alembic_output(direction: str) -> None:
    """原生 SQL 不是另一套手写口径；迁移修改后漏更新副本会失败。"""
    sql = render_migration_sql(direction)
    native = (ROOT / f"docs_zhx/sql/20260907_13_historical_nav_sample_storage_{direction}.sql").read_text(
        encoding="utf-8"
    )
    body = native.split("-- BEGIN GENERATED ALEMBIC SQL\n", 1)[1]
    assert body.strip() == sql.strip()
    assert sql.startswith("BEGIN;") and sql.rstrip().endswith("COMMIT;")


def test_upgrade_only_creates_three_tables_and_full_comments() -> None:
    sql = render_migration_sql("upgrade")
    assert sql.count("CREATE TABLE ") == 3
    assert sql.count("COMMENT ON TABLE ") == 3
    assert sql.count("COMMENT ON COLUMN ") == sum(len(t.columns) for t in TABLES)
    for table in TABLES:
        for column in table.columns:
            literal = column.comment.replace("'", "''")
            assert f"COMMENT ON COLUMN {table.name}.{column.name} IS '{literal}';" in sql
    assert "14%%" not in sql
    assert "CREATE INDEX ix_historical_nav_batch_fund_created" in sql
    assert "future_return_20d NUMERIC NOT NULL" in sql
    for verb in ("DROP ", "ALTER ", "INSERT ", "UPDATE ", "DELETE ", "TRUNCATE "):
        assert verb not in sql
    # 旧迁移没有重写，新迁移也不依赖随业务演化的模型定义。
    assert "from app.models" not in MIGRATION_FILE.read_text(encoding="utf-8")


def test_downgrade_locks_then_refuses_nonempty_tables_before_any_drop() -> None:
    """静态核对安全顺序；真正有数据时拒绝回退仍需隔离 PostgreSQL 集成验收。"""
    sql = render_migration_sql("downgrade")
    assert sql.index("LOCK TABLE ") < sql.index("IF EXISTS") < sql.index("RAISE EXCEPTION") < sql.index("DROP TABLE")
    assert "IN ACCESS EXCLUSIVE MODE NOWAIT" in sql
    for table in TABLES:
        assert f"EXISTS (SELECT 1 FROM {table.name})" in sql
    assert sql.index("DROP TABLE historical_nav_sample_label;") < sql.index("DROP TABLE historical_nav_sample;")
    assert sql.index("DROP TABLE historical_nav_sample;") < sql.index("DROP TABLE historical_nav_sample_batch;")
    assert "CASCADE" not in sql
