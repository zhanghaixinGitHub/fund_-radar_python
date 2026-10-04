"""覆盖本轮的停牌误删、未来信息和正文数字混淆，而不是模拟准确率。"""

from datetime import date, timedelta

import pytest

from scripts import fund_002112_semantic_extract_v1 as sem
from scripts import fund_002112_semantic_features_v1 as features
from scripts import fund_002112_semantic_train_v1 as train
from scripts import fund_002112_semantic_upgrade_v1 as core


def report():
    return {
        "fund_code": "002112",
        "fund_master_code": "001412",
        "report_end": "2024-12-31",
        "available_at": "2025-01-01T00:00:00+08:00",
        "raw": {"sha256": "report"},
        "disclosed_nav_pct": 30,
        "stock_nav_pct": 80,
        "full_stock_disclosure": False,
        "holdings": [{"stock_code": "A", "nav_weight_pct": 10}, {"stock_code": "B", "nav_weight_pct": 20}],
    }


def market():
    days = [(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(23)]
    quotes = {d: {"A": {"pct_chg": 2.0, "amount": 100}, "B": {"pct_chg": -1.0, "amount": 100}} for d in days}
    return days, quotes, days[-1] + "T08:00:00+08:00"


def test_one_stock_missing_does_not_zero_return_or_renormalize_others():
    days, quotes, cutoff = market()
    quotes[days[-2]].pop("B")
    values, proof = core.partial_holdings(report(), quotes, days, cutoff, [])
    assert values[0] == pytest.approx(0.1 * 0.02)
    assert values[9] == pytest.approx(0.1)
    assert values[13] == pytest.approx(0.2)
    assert proof["incomplete_stocks"][0]["code"] == "B"


def test_all_returns_missing_is_unknown_not_zero():
    days, quotes, cutoff = market()
    quotes[days[-2]] = {}
    values, _ = core.partial_holdings(report(), quotes, days, cutoff, [])
    assert values[:4] == [None] * 4
    assert values[9:12] == [0.0] * 3


def test_future_resume_cannot_erase_known_suspension():
    days, quotes, cutoff = market()
    quotes[days[-2]].pop("B")
    event = {
        "code": "B",
        "start_date": days[-2],
        "start_available_at": days[-2] + "T00:00:00+08:00",
        "resume_date": days[-1],
        "resume_available_at": "2025-02-01T00:00:00+08:00",
    }
    values, _ = core.partial_holdings(report(), quotes, days, cutoff, [event])
    assert values[12] == pytest.approx(0.2) and values[13] == 0
    event["start_available_at"] = "2025-02-01T00:00:00+08:00"
    values, _ = core.partial_holdings(report(), quotes, days, cutoff, [event])
    assert values[12] == 0 and values[13] == pytest.approx(0.2)


def validate(quote, metric, value, kind="company", title="公司公告"):
    doc = {
        "issuer_codes": ["A"] if kind == "company" else [],
        "type": "POLICY" if kind == "policy" else "ANNOUNCEMENT",
        "kind": kind,
        "text": quote,
        "text_level": "BODY",
        "title": title,
        "published_date": "2025-01-01",
    }
    raw = {
        "kind": "EARNINGS" if kind == "company" else "OTHER",
        "stage": "REALIZED",
        "facts": [quote],
        "direction": "UNKNOWN",
        "channel": "NONE",
        "impact_quote": "",
        "period_quote": "",
        "products": [],
        "targets": [],
        "condition": "",
        "quantities": [{"metric": metric, "value_text": value, "period_text": "", "quote": quote}],
    }
    return sem.validate(raw, doc)


def test_absolute_profit_is_not_yoy_percentage():
    assert not validate("公司2025年实现净利润100万元。", "PROFIT_YOY", "100万元")["quantities"]
    assert validate("公司2025年净利润同比增长20%。", "PROFIT_YOY", "20%")["quantities"]


def test_investment_or_financing_is_not_customer_order():
    assert not validate("本次出售股权合同金额100万元。", "CONTRACT_AMOUNT", "100万元")["quantities"]
    assert validate("本次中标采购合同金额100万元。", "CONTRACT_AMOUNT", "100万元")["quantities"]


def test_custodian_finances_do_not_become_fund_facts():
    result = validate("交通银行2025年实现净利润100万元。", "OTHER", "100万元", kind="fund")
    assert result["status"] == "QUARANTINED" and not result["facts"]


def test_policy_download_entry_is_not_policy_content():
    text = "国家医疗保障局关于印发政策的通知"
    result = validate(text, "OTHER", "", kind="policy", title=text)
    assert result["status"] == "QUARANTINED"


def test_yoy_down_and_interval_preserve_sign_and_missing():
    assert features.signed_yoy({"value_text": "下降20%至30%", "quote": "净利润同比下降20%至30%"}) == -0.25
    assert features.signed_yoy({"value_text": "1亿元", "quote": "利润1亿元"}) is None
    assert features.signed_yoy({"value_text": "20%和30%", "quote": "两列20%和30%"}) is None


def test_policy_does_not_use_future_business_profile():
    doc = {"kind": "policy", "body_available_at": "2025-01-10T00:00:00+08:00"}
    result = {"targets": [{"term": "光模块", "quote": "支持光模块技术研发"}]}
    profile = {
        "event_id": "p",
        "available_at": "2025-01-20T00:00:00+08:00",
        "products": [{"term": "光模块", "quote": "公司主营光模块生产"}],
    }
    assert features.company_links(doc, result, [report()], {"A": [profile]}) == []
    profile["available_at"] = "2025-01-05T00:00:00+08:00"
    linked = features.company_links(doc, result, [report()], {"A": [profile]})
    assert linked[0]["at_event_weight"] == 0.1


def test_future_fact_not_admitted_and_unknown_percentage_is_not_zero():
    days, _, cutoff = market()
    event = {"id": "future", "available_at": "2025-02-01T00:00:00+08:00"}
    values, used = features.vector({"as_of": cutoff}, [event], [report()], days)
    assert used == []
    names = features.feature_names()
    assert values[names.index("s5_profit_yoy_mean")] is None
    assert values[names.index("s5_profit_yoy_covered_weight")] == 0


def test_application_connection_does_not_transfer_policy_benefit_to_company():
    doc = {"kind": "policy", "topic": "TECH", "body_available_at": "2025-01-10T00:00:00+08:00"}
    result = {"targets": [{"term": "算力基础设施", "quote": "优化布局算力基础设施，建设数据中心。"}]}
    profile = {
        "event_id": "p",
        "available_at": "2025-01-05T00:00:00+08:00",
        "products": [{"term": "光模块", "quote": "公司主营光模块研发生产，产品应用于数据中心。"}],
    }
    links = features.company_links(doc, result, [report()], {"A": [profile]})
    assert links[0]["basis"] == "APPLICATION_CONTEXT_ONLY"
    days, _, _ = market()
    event = {
        "id": "policy",
        "available_at": doc["body_available_at"],
        "links": links,
        "repeated_facts": False,
        "source_kind": "policy",
        "status": "SOURCE_GROUNDED",
        "kind": "POLICY",
        "stage": "IMPLEMENTATION",
        "direction": "BENEFIT",
        "channel": "DEMAND",
        "quantities": [],
    }
    values, _ = features.vector({"as_of": "2025-01-10T08:00:00+08:00"}, [event], [report()], days)
    d = dict(zip(features.feature_names(), values, strict=True))
    assert d["s1_direction_BENEFIT"] == 0 and d["s1_direction_UNKNOWN"] == 0.1
    assert d["s1_public_application_weight"] == 0.1
    doc["topic"] = "HEALTH"
    result["targets"] = [{"term": "药品追溯", "quote": "国家医保局大数据中心收集药品追溯信息"}]
    assert features.company_links(doc, result, [report()], {"A": [profile]}) == []


def test_four_groups_do_not_fit_missing_value_or_scale_on_future_rows():
    rows = [
        {"groups": {"N": [1], "NE": [2]}, "H": [None], "F": [None]},
        {"groups": {"N": [3], "NE": [4]}, "H": [6], "F": [8]},
    ]
    future = {"groups": {"N": [999], "NE": [999]}, "H": [999], "F": [999]}
    _, state = train.transform(rows, "HOLDINGS_FACTS")
    assert state["imputer"].statistics_.tolist() == [2, 3, 6, 8]
    before = state["scaler"].mean_.copy()
    train.transform([future], "HOLDINGS_FACTS", state)
    assert state["scaler"].mean_.tolist() == before.tolist()


def test_label_mature_at_equal_cutoff_stays_out_of_training():
    rows = [
        {"target": "2024-12-27", "as_of": "2024-12-26T08:00:00+08:00", "label_mature_at": "2024-12-28T08:00:00+08:00"},
        {"target": "2024-12-30", "as_of": "2024-12-27T08:00:00+08:00", "label_mature_at": "2024-12-31T08:00:00+08:00"},
        {"target": "2025-01-02", "as_of": "2024-12-31T08:00:00+08:00", "label_mature_at": "2025-01-03T08:00:00+08:00"},
    ]
    assert train.calendar(rows, "2025")[0]["training"] == [0]
