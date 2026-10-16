"""公开市场扩展的关键缺失边界；不把其他份额、缺失值或不同费用口径放行。"""

from copy import deepcopy
from datetime import date
from decimal import Decimal

import pytest
from app.services.fund_rating_public_market import amount, key_information, provider_history, provider_team


def summary():
    return [
        "示例主动混合型证券投资基金产品资料概要更新 送出日期：2026年5月20日 "
        "基金代码166002 下属基金简称示例主动混合A 下属基金交易代码166002 "
        "下属基金简称示例主动混合C 下属基金交易代码004237 "
        "基金类型混合型 交易币种人民币 基金经理张三 开始担任本基金基金经理的日期2020年1月1日 "
        "三、投资本基金涉及的费用 本基金C类份额不收取申购费 N≥30天0.00% "
        "示例主动混合A 基金运作综合费率（年化）1.40% "
        "示例主动混合C 基金运作综合费率（年化）2.20%"
    ]


def test_select_c_cost_in_combined_share_summary():
    info = key_information(summary(), "004237", "示例主动混合")
    assert info["cost_1y"] == "0.022"
    assert "management" not in info["fees"]


def test_not_accept_a_only_document_for_c_share():
    with pytest.raises(ValueError, match="DOCUMENT_SCOPE"):
        key_information([summary()[0].replace("004237", "004238")], "004237", "示例主动混合")


def test_missing_operating_cost_is_not_replaced_by_component_fees():
    text = summary()[0].replace("基金运作综合费率（年化）", "管理费")
    with pytest.raises(ValueError, match="OPERATING_COST_MISSING"):
        key_information([text], "004237", "示例主动混合")


def test_missing_redemption_scenario_is_not_assumed_free():
    with pytest.raises(ValueError, match="REDEMPTION_SCENARIO"):
        key_information([summary()[0].replace("N≥30天0.00%", "N<30天1.00%")], "004237", "示例主动混合")


def test_explicit_statement_dash_is_zero_and_missing_row_rejects():
    assert amount("货币资金 6.4.7.1 - 12.00", "货币资金") == "0"
    with pytest.raises(ValueError, match="AMOUNT_ROW"):
        amount("应收股利 - -", "货币资金")


def test_statement_note_and_previous_year_are_not_amount():
    assert amount("货币资金 6.4.7.1 820,541,601.28 602,079,224.85", "货币资金") == "820541601.28"


def test_duplicate_statement_label_is_ambiguous():
    with pytest.raises(ValueError, match="AMOUNT_ROW"):
        amount("货币资金 1.00 2.00\n货币资金 3.00 4.00", "货币资金")


def test_manager_team_must_match_official_current_names():
    snapshot = {"manager_records": [{"manager_name": "张三", "begin_date": "2020-01-01", "end_date": None}]}
    actual = provider_team(snapshot, {"current_managers": ["张三"]}, date(2023, 9, 28), date(2026, 9, 30))
    assert actual["team_months"] == "36" and actual["team_changes"] == 0
    with pytest.raises(ValueError, match="TEAM_ORIGINAL_CONFLICT"):
        provider_team(snapshot, {"current_managers": ["李四"]}, date(2023, 9, 28), date(2026, 9, 30))


def test_team_history_gap_not_silently_joined():
    snapshot = {
        "manager_records": [
            {"manager_name": "张三", "begin_date": "2020-01-01", "end_date": "2025-01-01"},
            {"manager_name": "李四", "begin_date": "2025-01-02", "end_date": None},
        ]
    }
    with pytest.raises(ValueError, match="TEAM_HISTORY_GAP"):
        provider_team(snapshot, {"current_managers": ["李四"]}, date(2023, 9, 28), date(2026, 9, 30))


def test_empty_calendar_does_not_crash_or_fill_history(monkeypatch):
    monkeypatch.setattr("app.services.direction_1d_protocol.calendar", lambda: ([], "hash"))
    snapshot = {"history": {"dates": [], "calendar_hash": "hash", "rows": []}}
    with pytest.raises(ValueError, match="CALENDAR_GAP"):
        provider_history(deepcopy(snapshot), date(2026, 9, 30))


def a_summary():
    return [
        "示例主动混合型证券投资基金(A类份额)产品资料概要更新 送出日期：2026年5月20日 "
        "基金代码166002 下属基金简称示例主动混合A 下属基金交易代码166002 "
        "基金类型混合型 交易币种人民币 基金经理张三 开始担任本基金基金经理的日期2020年1月1日 "
        "三、投资本基金涉及的费用 (一)基金销售相关费用 申购费（前收费）M<100万元1.50% "
        "100万元≤M＜300万元1.20% M≥500万元1,000元/笔 锁定期限后方可赎回，赎回时不收取赎回费 "
        "(二)基金运作相关费用 示例主动混合A 基金运作综合费率（年化）1.40%"
    ]


def a_prospectus():
    return [
        "示例主动混合型证券投资基金招募说明书 本基金A类基金份额不收取销售服务费，"
        "C类基金份额的销售服务费年费率为0.40%。"
    ]


def test_a_share_front_load_subscription_enters_standard_cost():
    info = key_information(a_summary(), "166002", "示例主动混合", a_prospectus())
    assert info["fees"]["subscription"] == "0.015"
    assert info["fees"]["redemption"] == "0"
    assert info["fees"]["operating_cost"] == "0.014"
    expected = Decimal("0.015") / (1 + Decimal("0.015")) + Decimal("0.014")
    assert info["cost_1y"] == str(expected)


def test_a_share_ascii_redemption_tier_is_verified():
    text = a_summary()[0].replace("锁定期限后方可赎回，赎回时不收取赎回费", "N>=365天0.00%")
    info = key_information([text], "166002", "示例主动混合", a_prospectus())
    assert info["fees"]["redemption"] == "0"


def test_a_share_sales_fee_must_be_explicit_not_inferred():
    with pytest.raises(ValueError, match="SALES_FEE_SCENARIO"):
        key_information(
            a_summary(),
            "166002",
            "示例主动混合",
            ["示例主动混合型证券投资基金招募说明书 本基金A类基金份额的费率以公告为准。"],
        )


def test_a_share_ambiguous_first_tier_rejects():
    text = a_summary()[0].replace(
        "申购费（前收费）M<100万元1.50%", "认购费（前收费）M<100万元1.20% 申购费（前收费）M<100万元1.50%"
    )
    with pytest.raises(ValueError, match="SUBSCRIPTION_SCENARIO"):
        key_information([text], "166002", "示例主动混合", a_prospectus())


def test_a_share_identity_must_bind_label_to_code():
    text = (
        a_summary()[0]
        .replace("混合A 下属基金交易代码166002", "混合 下属基金交易代码166002")
        .replace("(A类份额)", "")
    )
    with pytest.raises(ValueError, match="SHARE_CLASS_FEE_IDENTITY"):
        key_information([text], "166002", "示例主动混合", a_prospectus())
