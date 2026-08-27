"""为重点基金历史净值回填补齐月分区。

Revision ID: 20260827_04
Revises: 20260826_03
Create Date: 2026-08-27 10:20:00
"""

from collections.abc import Iterator, Sequence
from datetime import date

from alembic import op

revision: str = "20260827_04"
down_revision: str | None = "20260826_03"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FIRST_MONTH = date(2015, 1, 1)
_EXCLUSIVE_END_MONTH = date(2026, 8, 1)


def _month_ranges() -> Iterator[tuple[date, date]]:
    """逐月生成重点基金历史区间，避免落入默认分区。"""
    current = _FIRST_MONTH
    while current < _EXCLUSIVE_END_MONTH:
        next_month = date(current.year + 1, 1, 1) if current.month == 12 else date(current.year, current.month + 1, 1)
        yield current, next_month
        current = next_month


def _partition_name(start_date: date) -> str:
    """返回固定、可信的月分区名称。"""
    return f"nav_daily_{start_date.year}_{start_date.month:02d}"


def upgrade() -> None:
    """创建 2015-01 至 2026-07 的月分区；2026-08 分区已由首个迁移创建。"""
    for start_date, end_date in _month_ranges():
        op.execute(
            f"CREATE TABLE {_partition_name(start_date)} PARTITION OF nav_daily "
            f"FOR VALUES FROM ('{start_date.isoformat()}') TO ('{end_date.isoformat()}')"
        )


def downgrade() -> None:
    """按反向月份删除本迁移新增的空/历史分区；降级前需确认其中数据已另行归档。"""
    for start_date, _end_date in reversed(tuple(_month_ranges())):
        op.execute(f"DROP TABLE {_partition_name(start_date)}")
