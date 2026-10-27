"""基于已核实合同约束的同类单位净值对照，统一区间、产品和份额口径。"""

import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.repositories.fund_read import (
    FundNavHistorySnapshot,
    FundSameTypeComparisonResult,
    get_current_market_same_type_comparison,
    list_peer_nav_endpoints,
)
from app.schemas.fund import (
    InternalFundComparisonGroup,
    InternalFundSameTypeComparison,
    InternalFundSameTypeComparisonItem,
)
from app.services.fund_peer_classifications import PeerClassification, load_peer_classifications
from app.services.trading_calendar import CalendarCoverageError, load_prediction_calendar

logger = logging.getLogger(__name__)
SCOPE = "VERIFIED_CURRENT_MARKET_PEER_GROUP"


def comparison_start(end: date, classification: PeerClassification) -> date:
    """境内沪深资产使用已核验休市日历确定近 30 天起点，不由基金缺数倒推休市。"""
    nominal = end - timedelta(days=30)
    if classification.criteria["market"] != "CN_A":
        # 无完整其他市场日历时只接受精确 30 天的两端原值，不猜测工作日回退。
        return nominal
    calendar = load_prediction_calendar(nominal)
    if end not in load_prediction_calendar(end).sessions:
        raise CalendarCoverageError("comparison end is not a verified session")
    return calendar.sessions[calendar.at_or_before_index(nominal)]


def _rate(points: tuple[FundNavHistorySnapshot, ...], start: date, end: date) -> tuple[Decimal, str] | None:
    """固定起止日，并使用相同来源的两端；异常原值不当零值，不跨份额拼接。"""
    by_date = {start: {}, end: {}}
    for point in points:
        if point.nav_date in by_date and point.source_code:
            by_date[point.nav_date][point.source_code] = point.unit_nav
    sources = sorted(set(by_date[start]) & set(by_date[end]))
    if not sources:
        return None
    source = sources[0]
    base, last = by_date[start][source], by_date[end][source]
    if any(value is None or not value.is_finite() or value <= 0 for value in (base, last)):
        return None
    return last / base - Decimal(1), source


def build_peer_comparison(
    result: FundSameTypeComparisonResult,
    classifications: dict[str, PeerClassification],
    points: dict[str, tuple[FundNavHistorySnapshot, ...]],
    start: date | None,
    today: date | None = None,
) -> InternalFundSameTypeComparison:
    """纯业务计算；产品去重、同类匹配与数据准入都完成后才确定本次排列。"""
    target = result.target
    code, end = target.fund.fund_code, target.nav_date
    base = dict(fund_code=code, fund_type=target.fund.fund_type, scope=SCOPE, as_of_date=end, sample_unit="PRODUCT")
    if target.fund.status != "ACTIVE" or target.fund.source_code != "TUSHARE_PRO_FUND":
        return InternalFundSameTypeComparison(**base, status="OUT_OF_SCOPE", reason="当前基金不在已收录的比较范围内。")
    classification = classifications.get(code)
    if classification is None or classification.criteria["fund_type"] != target.fund.fund_type:
        return InternalFundSameTypeComparison(
            **base, status="CLASSIFICATION_INSUFFICIENT", reason="分类资料不足，暂无法确定细分同类。",
        )
    group = InternalFundComparisonGroup(
        name=classification.name,
        description=f"{classification.description}仅比较{classification.share_class}类份额。",
        basis="VERIFIED_DISCLOSURE",
    )
    base.update(comparison_group=group, comparison_start_date=start)
    today = today or datetime.now(ZoneInfo("Asia/Shanghai")).date()
    if end is not None and end > today:
        return InternalFundSameTypeComparison(
            **base, status="DATA_INSUFFICIENT", reason="最新净值日期异常，暂时无法比较。",
        )
    if start is None or end is None:
        return InternalFundSameTypeComparison(
            **base, status="DATA_INSUFFICIENT", reason="同一统计区间的净值或日历资料不足。",
        )
    if not classification.covers(start, end):
        return InternalFundSameTypeComparison(
            **base, status="CLASSIFICATION_INSUFFICIENT", reason="分类资料尚未覆盖整个统计区间，暂无法比较。",
        )
    target_rate = _rate(points.get(code, ()), start, end)
    if target_rate is None:
        return InternalFundSameTypeComparison(
            **base, status="DATA_INSUFFICIENT", reason="当前基金缺少同一统计区间的有效净值。",
        )

    # 先检查共同区间原值的有效性，再固定按代码选代表；不因另一份额涨幅更好而替换。
    representatives = {classification.product_id: target}
    rates = {code: target_rate}
    for item in sorted(result.items, key=lambda row: row.summary.fund.fund_code):
        row = item.summary
        peer = classifications.get(row.fund.fund_code)
        if (peer is None or not classification.same_group(peer) or not peer.covers(start, end)
                or peer.share_class != classification.share_class
                or row.fund.status != "ACTIVE" or row.fund.source_code != "TUSHARE_PRO_FUND"
                or row.fund.fund_type != target.fund.fund_type or row.nav_date != end):
            continue
        value = _rate(points.get(row.fund.fund_code, ()), start, end)
        if value is None:
            continue
        rates[row.fund.fund_code] = value
        representatives.setdefault(peer.product_id, row)
    comparable = []
    for row in representatives.values():
        comparable.append((row, *rates[row.fund.fund_code]))
    comparable.sort(key=lambda item: (-item[1], item[0].fund.fund_code))
    items, previous, rank = [], None, 0
    for index, (row, rate, source) in enumerate(comparable, start=1):
        if rate != previous:
            rank = index
        previous = rate
        items.append(InternalFundSameTypeComparisonItem(
            rank=rank, fund_code=row.fund.fund_code, fund_name=row.fund.fund_name,
            fund_type=row.fund.fund_type, as_of_date=end, month_change_rate=rate, data_source=source,
        ))
    count = len(items)
    logger.info(
        "fund_peer_comparison.build_peer_comparison   >>>   fundCode=%s start=%s end=%s candidates=%s products=%s",
        code, start, end, len(result.items), count,
    )
    return InternalFundSameTypeComparison(
        **base, status="SYNCED" if count > 1 else "NO_PEERS", comparable_count=count,
        comparable_product_count=count,
        target_rank=next((item.rank for item in items if item.fund_code == code), None) if count > 1 else None,
        items=tuple(items), reason=None if count > 1 else "暂无其他满足同类标准和统计区间的已收录基金。",
    )


def read_peer_comparison(session: Session, fund_code: str) -> InternalFundSameTypeComparison | None:
    """在同一数据库会话批量读取目录和两端净值，请求不访问外部来源或写入资料。"""
    result = get_current_market_same_type_comparison(session, fund_code)
    if result is None:
        return None
    classifications = load_peer_classifications()
    classification = classifications.get(fund_code)
    start = None
    if classification is not None and result.target.nav_date is not None:
        try:
            start = comparison_start(result.target.nav_date, classification)
        except (CalendarCoverageError, OSError, ValueError):
            logger.warning("fund_peer_comparison.read_peer_comparison   >>>   fundCode=%s 日历不可用", fund_code,
                           exc_info=True)
    points = {}
    if start is not None and result.target.nav_date is not None:
        codes = tuple({fund_code, *(item.summary.fund.fund_code for item in result.items)})
        points = list_peer_nav_endpoints(session, codes, start, result.target.nav_date)
    return build_peer_comparison(result, classifications, points, start)
