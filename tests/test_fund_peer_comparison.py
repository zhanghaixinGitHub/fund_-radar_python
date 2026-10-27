"""严格同类的风险边界；以下身份与原件均为隔离单测夹具，不是正式分类资料。"""

from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import pytest
from app.repositories.fund_read import (
    FundNavHistorySnapshot,
    FundPerformanceSnapshot,
    FundSameTypeComparisonResult,
    FundSameTypeComparisonSnapshot,
    FundSummarySnapshot,
)
from app.services.fund_peer_classifications import PeerClassification, PeerSnapshot, load_peer_classifications
from app.services.fund_peer_comparison import build_peer_comparison, comparison_start

START, END = date(2026, 9, 23), date(2026, 10, 23)


def classification(code="000001", product="P1", **changes):
    data = dict(
        fund_code=code, product_id=product, share_class="C", name="测试境内灵活配置混合基金",
        description="股票0%-95%；无固定行业配比。",
        criteria=dict(fund_type="MIXED", management="ACTIVE", market="CN_A", currency="CNY",
                      allocation_policy="0%-95%", sector_policy="NONE", reference_id="NONE", fx_policy="DOMESTIC_CNY",
                      vehicle="DIRECT"),
        effective_from="2026-01-01", sources=[dict(file="sources/fixture.json", sha256="a" * 64,
            url="https://example.com/fixture", title="单测虚构原件", published_on="2026-01-01", excerpts=["fixture"])],
    )
    data.update(changes)
    return PeerClassification.model_validate(data)


def summary(code="000001", *, nav_date=END, status="ACTIVE", fund_type="MIXED"):
    return FundSummarySnapshot(
        fund=SimpleNamespace(fund_code=code, fund_name="夹具" + code, fund_type=fund_type,
                             status=status, source_code="TUSHARE_PRO_FUND"),
        nav_date=nav_date, performance=FundPerformanceSnapshot(None, None, Decimal("999")),
    )


def points(last="1.1", *, base="1", source="S", base_source="S", start=START):
    return (FundNavHistorySnapshot(start, Decimal(base), None, base_source),
            FundNavHistorySnapshot(END, Decimal(last), None, source))


def result(*codes):
    return FundSameTypeComparisonResult(summary(), tuple(
        FundSameTypeComparisonSnapshot(summary(code), "S") for code in ("000001", *codes)
    ))


def test_same_contract_and_exact_window_uses_real_endpoints_not_old_month_summary():
    output = build_peer_comparison(result("000002"),
        {"000001": classification(), "000002": classification("000002", "P2")},
        {"000001": points("0.8"), "000002": points("1.1")}, START)
    assert output.status == "SYNCED"
    assert [(x.fund_code, x.month_change_rate) for x in output.items] == [
        ("000002", Decimal("0.1")), ("000001", Decimal("-0.2")),
    ]
    assert output.comparable_product_count == output.comparable_count == 2
    assert output.comparison_start_date == START and output.as_of_date == END
    assert "C类" in output.comparison_group.description


@pytest.mark.parametrize("key,value", [
    ("allocation_policy", "40%-80%"), ("sector_policy", "医药80%"), ("market", "CN_A_HK"),
    ("management", "ENHANCED_INDEX"), ("currency", "USD"), ("fx_policy", "HEDGED"),
    ("reference_id", "OTHER_INDEX"), ("vehicle", "ETF_FEEDER"), ("fund_type", "STOCK"),
])
def test_same_big_type_does_not_override_different_fixed_criteria(key, value):
    target = classification()
    peer = classification("000002", "P2", criteria=target.criteria | {key: value})
    output = build_peer_comparison(result("000002"), {"000001": target, "000002": peer},
                                   {"000001": points(), "000002": points("9")}, START)
    assert output.status == "NO_PEERS" and output.target_rank is None
    assert [row.fund_code for row in output.items] == ["000001"]


def test_target_actual_share_survives_family_dedup_and_higher_return_share_is_not_selected():
    classes = {"000001": classification(), "000002": classification("000002", "P1"),
               "000003": classification("000003", "P2"), "000004": classification("000004", "P2"),
               "000005": classification("000005", "P3", share_class="A")}
    output = build_peer_comparison(result("000002", "000003", "000004", "000005"), classes,
        {"000001": points("0.8"), "000002": points("10"), "000003": points("1.1"),
         "000004": points("20"), "000005": points("30")}, START)
    assert [row.fund_code for row in output.items] == ["000003", "000001"]


def test_same_class_product_can_use_next_fixed_code_when_first_has_no_valid_data():
    classes = {"000001": classification(), "000002": classification("000002", "P2"),
               "000003": classification("000003", "P2")}
    output = build_peer_comparison(result("000002", "000003"), classes,
        {"000001": points(), "000002": points(base="0"), "000003": points("0.9")}, START)
    assert [row.fund_code for row in output.items] == ["000001", "000003"]


def test_future_nav_is_not_used_as_current_comparison():
    output = build_peer_comparison(result(), {"000001": classification()}, {"000001": points()},
                                   START, today=date(2026, 10, 22))
    assert output.status == "DATA_INSUFFICIENT" and not output.items


@pytest.mark.parametrize("peer_points", [
    points(start=date(2026, 9, 22)), points(base_source="ANOTHER"), points(base="0"),
    points(last="NaN"), points(last="Infinity"), points(base="-1"), (),
])
def test_missing_wrong_date_cross_source_and_invalid_values_are_excluded(peer_points):
    output = build_peer_comparison(result("000002"),
        {"000001": classification(), "000002": classification("000002", "P2")},
        {"000001": points(), "000002": peer_points}, START)
    assert output.status == "NO_PEERS" and len(output.items) == 1


@pytest.mark.parametrize("changes", [
    {"effective_from": "2026-09-24"}, {"effective_to": "2026-10-22"}, {"revoked": True},
])
def test_classification_must_cover_entire_window_and_not_be_revoked(changes):
    output = build_peer_comparison(result(), {"000001": classification(**changes)},
                                   {"000001": points()}, START)
    assert output.status == "CLASSIFICATION_INSUFFICIENT" and not output.items


def test_missing_classification_does_not_fall_back_to_large_category():
    output = build_peer_comparison(result("000002"), {}, {"000001": points(), "000002": points()}, START)
    assert output.status == "CLASSIFICATION_INSUFFICIENT" and output.comparable_count == 0


def test_missing_target_endpoint_or_calendar_has_distinct_data_state():
    assert build_peer_comparison(result(), {"000001": classification()}, {}, START).status == "DATA_INSUFFICIENT"
    assert build_peer_comparison(result(), {"000001": classification()}, {}, None).status == "DATA_INSUFFICIENT"


def test_equal_returns_share_rank_and_code_order_without_replacing_target():
    classes = {"000001": classification(), "000002": classification("000002", "P2"),
               "000003": classification("000003", "P3")}
    output = build_peer_comparison(result("000003", "000002"), classes,
        {"000001": points(), "000002": points(), "000003": points("0.9")}, START)
    assert [(x.fund_code, x.rank) for x in output.items] == [("000001", 1), ("000002", 1), ("000003", 3)]


def test_official_holiday_calendar_does_not_invent_missing_trading_days():
    # 2026-10-07 为已核实国庆休市日，起点回退到 9 月 30 日；不是依某基金缺数推定。
    assert comparison_start(date(2026, 11, 6), classification()) == date(2026, 9, 30)


def test_snapshot_rejects_duplicate_identity_and_unknown_criteria():
    with pytest.raises(ValueError):
        PeerSnapshot(version="test", reviewed_on=START, review_expires_on=END,
                     records=(classification(), classification()))
    with pytest.raises(ValueError):
        classification(criteria=classification().criteria | {"market": "UNKNOWN"})


def test_expired_or_future_review_cannot_supply_classification(monkeypatch):
    from app.services import fund_peer_classifications as module

    # 时效测试使用独立快照，不依赖本机原始文件或实际分类复核策略。
    packet = PeerSnapshot(version="fixture", reviewed_on=date(2026, 10, 26),
                          review_expires_on=date(2026, 11, 25), records=(classification(),))
    monkeypatch.setattr(module, "_verified_snapshot", lambda *_args: packet)
    assert load_peer_classifications(date(2027, 1, 1)) == {}
    assert load_peer_classifications(date(2026, 1, 1)) == {}
