"""同一基准日真实净值的业务身份；读取时刻、归档封套不制造新版本。"""

from decimal import Decimal

from sqlalchemy import text

from app.services.direction_1d_protocol import digest


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
