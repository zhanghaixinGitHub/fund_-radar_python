"""诊断专用的有界当前快照读取；仅试点基金、2021–2024、同来源、只读事务。"""

from datetime import date

from sqlalchemy import select, text

from app.db.session import get_nav_sample_storage_engine
from app.models.fund import FundProfile, FundShareClass, NavDaily, SourceRegistry

PILOT_FUNDS = ("001632", "006730", "008888")


def read_nav_diagnostic_snapshot(funds: tuple[str, ...], source_code: str) -> dict:
    """一次独立只读快照；不是与已存批次同一事务，所以后续必须明确做内容重放比较。"""
    if not funds or len(set(funds)) != len(funds) or not set(funds) <= set(PILOT_FUNDS):
        raise ValueError("NAV diagnostics is restricted to the three pilot funds")
    with get_nav_sample_storage_engine().connect() as connection, connection.begin():
        connection.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        source = (
            connection.execute(
                select(
                    SourceRegistry.source_id,
                    SourceRegistry.source_code,
                    SourceRegistry.enabled,
                    SourceRegistry.authorized_api_names,
                    SourceRegistry.authorization_verified_at,
                ).where(SourceRegistry.source_code == source_code)
            )
            .mappings()
            .one()
        )
        profiles = (
            connection.execute(
                select(
                    FundShareClass.fund_code,
                    FundShareClass.fund_name,
                    FundShareClass.fund_master_id,
                    FundShareClass.fund_type,
                    FundShareClass.status,
                    FundShareClass.source_code,
                    FundProfile.invest_type,
                    FundProfile.source_fund_type,
                    FundProfile.found_date,
                )
                .outerjoin(
                    FundProfile,
                    (FundProfile.fund_code == FundShareClass.fund_code)
                    & (FundProfile.source_id == source["source_id"]),
                )
                .where(FundShareClass.fund_code.in_(funds))
                .order_by(FundShareClass.fund_code)
                .limit(4)
            )
            .mappings()
            .all()
        )
        if {r["fund_code"] for r in profiles} != set(funds) or len(profiles) != len(funds):
            raise ValueError("pilot catalog is missing or duplicated")
        if any(r["source_code"] != source_code for r in profiles):
            raise ValueError("pilot current source differs from saved source")
        # 三基金四年最多4383个不同自然日；4501哨兵发现超量，不返回截断报告。
        nav = (
            connection.execute(
                select(
                    NavDaily.fund_code,
                    NavDaily.source_id,
                    NavDaily.nav_date,
                    NavDaily.ann_date,
                    NavDaily.unit_nav,
                    NavDaily.accumulated_nav,
                    NavDaily.adjusted_nav,
                    NavDaily.accumulated_dividend,
                    NavDaily.source_published_at,
                    NavDaily.created_at,
                    NavDaily.updated_at,
                    NavDaily.content_hash,
                )
                .where(
                    NavDaily.fund_code.in_(funds),
                    NavDaily.source_id == source["source_id"],
                    NavDaily.nav_date >= date(2021, 1, 1),
                    NavDaily.nav_date <= date(2024, 12, 31),
                )
                .order_by(NavDaily.fund_code, NavDaily.nav_date)
                .limit(4501)
            )
            .mappings()
            .all()
        )
        if len(nav) > 4500:
            raise ValueError("raw snapshot exceeded bounded audit size")
        tables = (
            connection.execute(
                text(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema='public' "
                    "AND table_type='BASE TABLE' AND table_name !~ '_[0-9]{4}' ORDER BY table_name"
                )
            )
            .scalars()
            .all()
        )
    return {
        "source_registry": dict(source),
        "fund_profiles": [dict(r) for r in profiles],
        "nav_rows": [dict(r) for r in nav],
        "public_base_tables": tables,
    }
