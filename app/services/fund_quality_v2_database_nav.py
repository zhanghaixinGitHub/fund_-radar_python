"""重读本机只读提取的规范净值快照，保留数据库来源及真实时间边界。

这是规范记录的冻结来源，不是供应商原始HTTP回执。接口授权上下文、逐行
值及摘要必须真实存在；覆盖事实仍由实际记录重建，不能由成功标志宣告。
"""

from datetime import date, datetime, timedelta

from app.services.fund_quality_v2_nav import rebuild_nav_coverage


def database_nav_coverage(snapshot: dict, *, fund_code: str, as_of: date, frozen: datetime) -> dict:
    """验证基金/来源/提取时点，再复算37个月末与逐估值日覆盖；不生成相对指标。"""
    if (
        snapshot.get("schema") != "QUALITY_DATABASE_NAV_SNAPSHOT_V1"
        or snapshot.get("origin") != "LOCAL_READ_ONLY_NORMALIZED_SOURCE_RECORDS"
        or snapshot.get("fund_code") != fund_code
        or snapshot.get("as_of_date") != str(as_of)
    ):
        raise ValueError("QUALITY_DATABASE_NAV_SCHEMA_IDENTITY")
    captured = datetime.fromisoformat(snapshot["captured_at"])
    source = snapshot.get("source_registry") or {}
    verified = (
        datetime.fromisoformat(source["authorization_verified_at"]) if source.get("authorization_verified_at") else None
    )
    if (
        frozen.tzinfo is None
        or captured.tzinfo is None
        or captured > frozen
        or source.get("source_code") != "TUSHARE_PRO_FUND"
        or not source.get("source_id")
        or source.get("enabled") is not True
        or verified is None
        or verified.tzinfo is None
        or verified > captured
        or "fund_nav" not in source.get("authorized_api_names", [])
        or not 0 < source.get("retention_days", 0) <= 36500
        or captured + timedelta(days=source["retention_days"]) < frozen
    ):
        raise ValueError("QUALITY_DATABASE_NAV_SOURCE_OR_TIME")
    scope = snapshot.get("query_scope") or {}
    if fund_code not in scope.get("fund_codes", []) or scope.get("to") != str(as_of) or not snapshot.get("query"):
        raise ValueError("QUALITY_DATABASE_NAV_QUERY_SCOPE")
    for row in (snapshot.get("history") or {}).get("rows", []):
        if row.get("fund_code") != fund_code:
            raise ValueError("QUALITY_DATABASE_NAV_ROW_FUND_SCOPE")
        for field in ("created_at", "updated_at"):
            observed = datetime.fromisoformat(row[field])
            if observed.tzinfo is None or observed > captured:
                raise ValueError("QUALITY_DATABASE_NAV_ROW_MAINTENANCE_TIME")
    result = rebuild_nav_coverage(snapshot, fund_code=fund_code, as_of=as_of)
    return result | {
        "source_kind": "NORMALIZED_DATABASE_NAV_SNAPSHOT_NOT_RAW_HTTP_RESPONSE",
        "captured_at": snapshot["captured_at"],
        "source_code": source["source_code"],
        "publication_scope": "ANN_DATE_AND_LOCAL_MAINTENANCE_RETAINED_NOT_FIRST_PUBLICATION_PROOF",
        "revision_scope": "CURRENT_DATABASE_ROWS_NOT_HISTORICAL_REVISION_REPLAY",
    }
