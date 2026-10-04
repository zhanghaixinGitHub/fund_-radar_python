"""信息改进的关键边界：来源字段、历史时点、关联层级和完整价格窗口。"""

from copy import deepcopy

import pytest

from scripts import fund_002112_evening_refine_v1 as m


def fixture():
    report = {
        "fund_code": "002112",
        "report_end": "2023-12-31",
        "available_at": "2024-01-01T00:00:00+08:00",
        "raw": {"sha256": "report"},
        "holdings": [{"stock_code": "A", "stock_name": "甲方科技", "nav_weight_pct": "8"}],
    }
    source = {
        "sessions": ["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"],
        "reports": [report],
        "company_events": [],
        "stocks": {},
    }
    row = {
        "base": "2024-01-08",
        "target": "2024-01-09",
        "as_of": "2024-01-08T23:00:00+08:00",
        "groups": {"information": [None] * len(m.old.INFO_NAMES), "market": [1], "history": [2]},
    }
    event = {
        "id": "e",
        "code": "A",
        "source_kind": "company",
        "status": "QUALIFIED",
        "version_available_at": "2024-01-07T00:00:00+08:00",
        "evening_index": 1,
        "period": "2023FY",
        "links": [{"code": "A", "weight": 0.1}],
        "numeric": {"profit_yoy": {"value": 0.4, "quote": "净利润同比增长40%", "period": "2023FY"}},
    }
    public = {
        "id": "p",
        "title": "加快数据中心建设",
        "facts": ["政策支持数据中心建设"],
        "available_at": "2024-01-07T00:00:00+08:00",
        "targets": [{"term": "数据中心", "quote": "支持数据中心"}],
        "source_kind": "policy",
        "direction": "BENEFIT",
        "kind": "POLICY",
        "stage": "PROPOSAL",
        "channel": "DEMAND",
        "evening_index": 1,
        "repeated_facts": False,
        "new_fraction": 1.0,
    }
    profile = {
        "code": "A",
        "available_at": "2024-01-03T00:00:00+08:00",
        "document_id": "b",
        "term": "光模块",
        "quote": "公司光模块应用于数据中心",
        "source_hash": "hash",
        "topics": {"OPTICAL": "光模块", "COMPUTE": "数据中心"},
    }
    return source, row, event, public, profile


def test_profit_source_alias_and_only_two_changed_columns():
    source, row, event, _, _ = fixture()
    source["company_events"] = [event]
    original = deepcopy(row)
    values, ids = m.fixed_information(row, source)
    assert values[3] == 0.4 and values[16] == 0.08 and ids == ["e"]
    assert all(
        a == b for i, (a, b) in enumerate(zip(values, row["groups"]["information"], strict=True)) if i not in [3, 16]
    )
    assert row == original


@pytest.mark.parametrize("change", ["future", "duplicate", "no_current_holding", "expired"])
def test_profit_cannot_enter_without_time_status_and_both_holdings(change):
    source, row, event, _, _ = fixture()
    if change == "future":
        event["version_available_at"] = "2024-01-09T00:00:00+08:00"
    elif change == "duplicate":
        event["status"] = "DUPLICATE_EVENT"
    elif change == "no_current_holding":
        source["reports"][0]["holdings"] = []
    else:
        event["evening_index"] = -10
    source["company_events"] = [event]
    assert m.fixed_information(row, source)[0][3] is None


def test_profit_period_must_match():
    source, row, event, _, _ = fixture()
    event["numeric"]["profit_yoy"]["period"] = "2022FY"
    source["company_events"] = [event]
    with pytest.raises(ValueError, match="PERIOD"):
        m.fixed_information(row, source)


def test_context_does_not_inherit_benefit():
    source, row, _, public, profile = fixture()
    links = m.relate(public, source["reports"][0], {"A": [profile]}, row["as_of"])
    assert len(links) == 1 and links[0]["basis"] == "INDUSTRY_CONTEXT"
    assert links[0]["direction"] == "UNKNOWN"


@pytest.mark.parametrize("stamp", ["2024-01-08T00:00:00+08:00", "2021-01-01T00:00:00+08:00"])
def test_future_or_stale_profile_not_retroactively_linked(stamp):
    source, row, _, public, profile = fixture()
    profile["available_at"] = stamp
    assert not m.relate(public, source["reports"][0], {"A": [profile]}, row["as_of"])


def test_specific_drug_case_does_not_become_sector_wide_link():
    source, row, _, public, profile = fixture()
    public["targets"] = [{"term": "注射用硫酸多黏菌素B", "quote": "药品价格问题"}]
    profile.update(term="医药", topics={"PHARMA": "药品"})
    assert not m.relate(public, source["reports"][0], {"A": [profile]}, row["as_of"])


def test_issuer_reference_can_link_without_profile():
    source, row, _, public, _ = fixture()
    public["facts"] = ["甲方科技签署订单"]
    link = m.relate(public, source["reports"][0], {}, row["as_of"])[0]
    assert link["basis"] == "ISSUER" and link["direction"] == "BENEFIT"


def test_original_fact_novelty_uses_past_only():
    events = [
        {"id": "later", "available_at": "2024-01-09", "facts": ["旧事实", "新事实"]},
        {"id": "first", "available_at": "2024-01-07", "facts": ["旧事实"]},
    ]
    m.mark_novelty(events)
    assert events[0]["new_fraction"] == 0.5 and events[0]["repeated_source_ids"] == ["first"]
    assert events[1]["new_fraction"] == 1.0 and not events[1]["repeated_source_ids"]


def test_weekend_has_no_completed_price_response_before_monday():
    source, row, _, _, _ = fixture()
    value, proof = m.observed_reaction("A", "2024-01-07T00:00:00+08:00", "2024-01-05", row["as_of"], source)
    assert value is None and proof["status"] == "NO_COMPLETE_SESSION_YET"


def test_intraday_release_skips_partial_first_day():
    source, row, _, _, _ = fixture()
    source["stocks"] = {"2024-01-05": {"A": {"pct_chg": 9}}, "2024-01-08": {"A": {"pct_chg": 2}}}
    value, proof = m.observed_reaction("A", "2024-01-05T12:00:00+08:00", row["base"], row["as_of"], source)
    assert value == pytest.approx(0.02) and proof["price_days"] == ["2024-01-08"]


def test_future_price_revision_not_used():
    source, row, _, public, _ = fixture()
    source["stocks"] = {"2024-01-08": {"A": {"pct_chg": 2, "revised_at": "2024-01-09T08:00:00+08:00"}}}
    value, proof = m.observed_reaction("A", public["available_at"], row["base"], row["as_of"], source)
    assert value is None and proof["status"] == "QUOTE_UNAVAILABLE"


def test_context_direction_missingness_and_dimension():
    source, row, _, public, profile = fixture()
    values, proof = m.revised_features(row, source, {"public": [public]}, {"A": [profile]})
    for group in m.GROUPS:
        assert len(values[group]) == len(m.feature_names(group))
    names = m.feature_names("B_LINK")
    assert values["B_LINK"][names.index("policy_5_direction_BENEFIT")] == 0
    assert values["B_LINK"][names.index("policy_5_direction_UNKNOWN")] == 0.08
    assert proof["extras"]["guidance_revision_delta"] is None
    assert proof["extras"]["public_observed_reaction"] is None
    absent, _ = m.revised_features(row, source, {"public": []}, {})
    assert absent["B_LINK"][names.index("policy_5_direction_UNKNOWN")] is None


def test_buyback_needs_same_currency_and_cap_not_identical_quote():
    previous = {"cap": {"value": 100, "currency": "CNY", "quote": "上限100"}, "ratio": {"value": 0.1}}
    current = {"cap": {"value": 100, "currency": "CNY", "quote": "最多100"}, "ratio": {"value": 0.2}}
    previous["executed"] = {"quote": "已支付的总金额10元"}
    current["executed"] = {"quote": "累计已支付20元"}
    assert m.comparable_buyback(previous, current)
    current["cap"]["currency"] = "USD"
    assert not m.comparable_buyback(previous, current)
    assert not m.comparable_buyback(None, current)


def test_organ_chip_does_not_match_electronics():
    assert not m.topic_terms("器官芯片")
    assert "SEMICONDUCTOR" in m.topic_terms("集成电路")
    assert "COMPUTE" not in m.topic_terms("集成电路")


def test_single_payment_cannot_be_cumulative_ratio():
    source, row, event, _, _ = fixture()
    event["numeric"]["buyback_ratio"] = {"value": 0.022}
    event["buyback"] = {"ratio": {"value": 0.022}, "executed": {"quote": "支付的金额为670万元"}}
    source["company_events"] = [event]
    row["groups"]["information"][6] = 0.022
    row["groups"]["information"][18] = 0.08
    values, _ = m.fixed_information(row, source)
    assert values[6] is None and values[18] == 0
    event["buyback"]["executed"]["quote"] = "已支付的总金额为670万元"
    assert m.fixed_information(row, source)[0][6] == 0.022


def test_buyback_month_must_be_explicit():
    assert m.buyback_month({"title": "关于2026年2月股份回购进展的公告"}) == "2026-02"
    assert m.buyback_month({"title": "关于股份回购实施结果暨股份变动的公告"}) is None
