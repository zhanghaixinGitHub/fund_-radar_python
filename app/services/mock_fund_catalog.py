"""M0 内存基金读模型；M1 必须以已授权数据源实现替换。"""

from app.schemas.fund import InternalFundDetail, InternalFundPage, InternalFundSummary

_FUND_DETAILS: tuple[InternalFundDetail, ...] = (
    InternalFundDetail(
        fund_code="000001",
        fund_name="M0 示例权益混合基金",
        fund_type="MIXED",
        status="ACTIVE",
        as_of_date="2026-08-21",
        nav_status="MOCK",
        data_source="M0_MOCK",
    ),
    InternalFundDetail(
        fund_code="000002",
        fund_name="M0 示例中短债基金",
        fund_type="BOND",
        status="ACTIVE",
        as_of_date="2026-08-21",
        nav_status="MOCK",
        data_source="M0_MOCK",
    ),
    InternalFundDetail(
        fund_code="000003",
        fund_name="M0 示例指数基金",
        fund_type="INDEX",
        status="ACTIVE",
        as_of_date="2026-08-21",
        nav_status="MOCK",
        data_source="M0_MOCK",
    ),
)


def list_mock_funds(keyword: str | None, page_size: int, cursor: str | None) -> InternalFundPage:
    """在内存示例数据中按关键字过滤并返回游标分页，不访问任何外部数据源。"""
    offset = int(cursor) if cursor else 0
    normalized_keyword = keyword.strip().lower() if keyword else ""
    filtered = tuple(
        item
        for item in _FUND_DETAILS
        if not normalized_keyword
        or normalized_keyword in item.fund_code
        or normalized_keyword in item.fund_name.lower()
    )
    selected = filtered[offset : offset + page_size]
    next_offset = offset + len(selected)
    next_cursor = str(next_offset) if next_offset < len(filtered) else None
    return InternalFundPage(
        items=tuple(
            InternalFundSummary(
                fund_code=item.fund_code,
                fund_name=item.fund_name,
                fund_type=item.fund_type,
                status=item.status,
                as_of_date=item.as_of_date,
            )
            for item in selected
        ),
        next_cursor=next_cursor,
    )


def get_mock_fund(fund_code: str) -> InternalFundDetail | None:
    """返回指定的 Mock 基金详情；代码不存在时返回 None。"""
    return next((item for item in _FUND_DETAILS if item.fund_code == fund_code), None)
