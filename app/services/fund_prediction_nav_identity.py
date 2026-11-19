"""同一基准日真实净值的业务身份；读取时刻、归档封套不制造新版本。"""

from datetime import date
from decimal import Decimal

from sqlalchemy import text

from app.services.direction_1d_protocol import ZONE, digest, input_days


def nav_identity(rows, base_date):
    selected = []
    for row in rows:
        if str(row["nav_date"]) != str(base_date):
            continue
        value = Decimal(str(row["unit_nav"]))
        if not value.is_finite() or value <= 0:
            raise ValueError("PREDICTION_BASE_NAV_INVALID")
        selected.append({
            "nav_date": str(row["nav_date"]), "ann_date": str(row["ann_date"]),
            "source_id": str(row["source_id"]), "unit_nav": format(value.normalize(), "f"),
        })
    return digest(sorted(selected, key=lambda row: (row["source_id"], row["ann_date"], row["unit_nav"])))


def current_nav_identity(code, base_date):
    from app.db.session import get_nav_preview_engine

    with get_nav_preview_engine().connect() as connection:
        rows = connection.execute(text(
            "SELECT n.nav_date,n.ann_date,n.source_id,n.unit_nav FROM nav_daily n "
            "JOIN source_registry s ON s.source_id=n.source_id "
            "WHERE n.fund_code=:code AND n.nav_date=:day AND s.enabled=true"
        ), {"code": code, "day": base_date}).mappings()
        return nav_identity(rows, base_date)


def _qdii(profile):
    return "QDII" in str(profile.get("fund_name", "")).upper()


def previous_qdii_nav_date(rows, base_date, profile):
    """只接受已披露的本份额净值，最多回退一个已核验的沪深交易日。"""
    if not _qdii(profile):
        return None
    prior = str(input_days(date.fromisoformat(str(base_date)))[-2])
    if max((str(row["nav_date"]) for row in rows), default=None) != prior:
        return None
    return prior if nav_identity(rows, prior) != digest([]) else None


def available_prediction_nav(connection, profile, source_id, base, at):
    """读取当前可用于预测的供应商记录，基准日净值优先。"""
    from app.repositories import direction_1d as repo

    start = input_days(base)[-2] if _qdii(profile) else base
    today = at.astimezone(ZONE).date()
    rows = [
        {**row, "source_id": source_id}
        for row in repo.navs(connection, profile["fund_code"], source_id, start, base)
        if row["updated_at"] <= at
        and (row["ann_date"] is None or row["ann_date"] <= today)
    ]
    selected = str(base) if any(row["nav_date"] == base for row in rows) else previous_qdii_nav_date(
        rows, base, profile
    )
    return selected, nav_identity(rows, selected or base)


def current_prediction_nav_identity(code, base_date, *, at=None):
    from app.db.session import get_nav_preview_engine
    from app.repositories import direction_1d as repo

    with get_nav_preview_engine().connect() as connection:
        profiles = repo.profiles(connection, [code])
        if not profiles:
            return digest([])
        source = repo.source(connection)
        instant = at or connection.scalar(text("SELECT clock_timestamp()"))
        _, identity = available_prediction_nav(
            connection, profiles[0], source["source_id"], date.fromisoformat(str(base_date)), instant
        )
        return identity
