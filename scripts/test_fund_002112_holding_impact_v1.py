"""针对错误归因、未来持仓、编造数字和正文缺失的独立回归检查。"""

from copy import deepcopy

import numpy as np

from scripts import fund_002112_holding_impact_fit_v1 as f
from scripts import fund_002112_holding_impact_v1 as e


def fixture():
    text = "2022年度业绩预告。本公司净利润预计同比减少50%。公司主要从事药品研发和生产。"
    doc = {"text": text, "issuer_codes": ["600001"], "type": "ANNOUNCEMENT"}
    raw = {
        "kind": "EARNINGS",
        "stage": "FORECAST",
        "facts": ["本公司净利润预计同比减少50%。"],
        "period_quote": "2022年度",
        "direction": "PRESSURE",
        "channel": "PROFIT",
        "impact_quote": "本公司净利润预计同比减少50%。",
        "condition": "若预计业绩兑现，则盈利承压。",
        "products": [{"term": "药品", "quote": "公司主要从事药品研发和生产。"}],
        "targets": [],
    }
    return doc, raw


def test_facts_need_verbatim_source():
    doc, raw = fixture()
    raw["facts"] = ["本公司净利润预计同比减少80%。"]
    assert e.validate(raw, doc)["status"] == "QUARANTINED"


def test_layout_normalization_does_not_merge_numeric_columns():
    assert e.quote_text("净利 润 50%") == e.quote_text("净利润50%")
    assert e.quote_text("利润 1 2") != e.quote_text("利润12")


def test_earnings_forecast_is_not_realized_or_price_prediction():
    doc, raw = fixture()
    result = e.validate(raw, doc)
    assert result["status"] == "SOURCE_GROUNDED"
    assert result["stage"] == "FORECAST"
    assert result["expected_price_effect"] is None
    assert result["market_expectation"] is None


def test_revenue_increase_does_not_establish_profit_increase():
    doc, raw = fixture()
    doc["text"] += "本公司收入增加。"
    raw["impact_quote"] = "本公司收入增加。"
    assert e.validate(raw, doc)["direction"] == "UNKNOWN"


def test_unsupported_product_is_removed_without_losing_valid_facts():
    doc, raw = fixture()
    raw["products"] = [{"term": "疫苗", "quote": "公司生产疫苗。"}]
    result = e.validate(raw, doc)
    assert result["products"] == []
    assert result["facts"] == raw["facts"]
    assert "UNSUPPORTED_PRODUCTS" in result["field_warnings"]


def test_securities_are_not_company_operating_products():
    doc, raw = fixture()
    doc["text"] += "公司发行股票。"
    raw["products"] = [{"term": "股票", "quote": "公司发行股票。"}]
    assert e.validate(raw, doc)["products"] == []


def report(available="2023-04-01T08:00:00+08:00"):
    return {
        "available_at": available,
        "report_end": "2022-12-31",
        "raw": {"sha256": "report-hash"},
        "holdings": [{"stock_code": "600001.SH", "stock_name": "测试公司", "nav_weight_pct": "7.50"}],
    }


def test_future_report_not_used_and_missing_not_latest_backfilled():
    r = report()
    assert e.select_report([r], "2023-03-31T15:00:00+08:00") is None
    assert e.select_report([r], "2023-04-01T15:00:00+08:00") == r


def test_same_report_period_keeps_publication_order():
    first, later = report(), report("2023-04-10T08:00:00+08:00")
    assert e.select_report([first, later], "2023-04-05T15:00:00+08:00") == first


def test_direct_holding_weight_is_nav_pct_not_predicted_return():
    doc, raw = fixture()
    doc["text_evidence"] = {"source_refs": []}
    relation = f.links(doc, e.validate(raw, doc), report(), {}, "2023-04-05T15:00:00+08:00", "aux")[0]
    assert relation["nav_weight_pct"] == 7.5
    assert "fund_return" not in relation
    assert relation["company_direction"] == "PRESSURE"


def test_policy_needs_two_sources_and_cannot_use_future_business_profile():
    doc = {"issuer_codes": [], "text_evidence": {"source_refs": []}}
    result = {
        "status": "SOURCE_GROUNDED",
        "targets": [{"term": "药品", "quote": "药品价格降低。"}],
        "direction": "PRESSURE",
    }
    profile = {"quote": "公司生产药品。", "effective_aux": "2023-04-06"}
    profiles = {("600001", "药品"): [profile]}
    assert f.links(doc, result, report(), profiles, "2023-04-05T15:00:00+08:00", "aux") == []
    profile["effective_aux"] = "2023-04-04"
    link = f.links(doc, result, report(), profiles, "2023-04-05T15:00:00+08:00", "aux")[0]
    assert link["company_direction"] == "UNKNOWN"


def test_one_policy_does_not_double_count_one_holding_for_multiple_terms():
    doc = {"issuer_codes": [], "text_evidence": {"source_refs": []}}
    result = {"status": "SOURCE_GROUNDED", "direction": "UNKNOWN", "targets": [{"term": "药品"}, {"term": "中成药"}]}
    profile = {"effective_aux": "2023-04-01"}
    links = f.links(
        doc,
        result,
        report(),
        {("600001", t): [profile] for t in ("药品", "中成药")},
        "2023-04-05T15:00:00+08:00",
        "aux",
    )
    assert len(links) == 1


def test_missing_semantics_have_explicit_indicator_and_no_fabricated_direction():
    row = {
        "n8": [1] * 8,
        "context": [1, 0, 0, 0, 0],
        "aux": {"quality": [2] * 15, "global": [0] * len(f.GLOBAL_COLUMNS), "weighted": [0] * len(f.WEIGHT_COLUMNS)},
    }
    x = f.numeric([deepcopy(row)], "M3", "aux")
    assert np.array_equal(x[0, 13:28], np.array([2] * 15))
    assert len(row["n8"]) == 8


def test_profit_yoy_keeps_metric_period_and_sign():
    source = {"kind": "EARNINGS", "period_quote": "2022年度", "facts": ["净利润10亿元，同比减少50%。"]}
    assert f.profit_yoy(source) == -50
    source["facts"] = ["收入10亿元，同比增长50%。"]
    assert f.profit_yoy(source) is None
    source["facts"] = ["净利润10亿元，同比增长50%。", "净利润5亿元，同比增长20%。"]
    assert f.profit_yoy(source) is None
    source["facts"] = ["净利润10亿元，营业收入20亿元，同比增长50%。"]
    assert f.profit_yoy(source) is None
    source["facts"] = ["净利润同比增长50%-80%。"]
    assert f.profit_yoy(source) is None


def test_exact_publication_after_morning_not_visible_at_morning_cutoff():
    doc = {"published_date": "2023-01-10", "published_at": "2023-01-10T09:00:00+08:00", "revision_at": None}
    assert f.known_at(doc) > "2023-01-10T08:00:00+08:00"
    doc["published_at"] = None
    assert f.known_at(doc) == "2023-01-11T08:00:00+08:00"


def test_new_correction_suppresses_old_forecast_even_when_body_missing():
    docs = [
        {"title": "2022年度业绩预告", "issuer_codes": ["600001"], "event_id": "a", "effective_aux": "2023-01-10"},
        {
            "title": "2022年度业绩预告修正公告",
            "issuer_codes": ["600001"],
            "event_id": "b",
            "effective_aux": "2023-01-15",
        },
    ]
    assert f.event_window([[0, 5], [1, 0]], docs, [{"stage": "FORECAST"}, {}], "aux") == [[1, 0]]
