"""本期风险边界：事实错位、历史持仓、连续窗口、缺失、预算及真实事前留存。"""

from __future__ import annotations

import copy
from datetime import date, timedelta

import numpy as np
import pytest

from scripts import fund_002112_signal_common_v1 as c
from scripts import fund_002112_signal_features_v1 as f
from scripts import fund_002112_signal_forward_v1 as forward
from scripts import fund_002112_signal_train_v1 as train


def report(end="2025-03-31", available="2025-04-20T08:00:00+08:00", weight=4):
    return {
        "fund_code": "002112",
        "fund_master_code": "001412",
        "report_end": end,
        "available_at": available,
        "raw": {"sha256": end},
        "holdings": [{"stock_code": "000001.SZ", "nav_weight_pct": weight}],
    }


def event(available="2025-07-01T00:00:00+08:00", weight=0.05, end="2025-03-31"):
    return {
        "id": "event",
        "code": "000001.SZ",
        "version_available_at": available,
        "status": "QUALIFIED",
        "flags": ["buyback_execution", "material"],
        "important": True,
        "numeric": {"revenue_yoy": {"value": 0.2}},
        "links": [
            {
                "code": "000001.SZ",
                "weight": weight,
                "report_hash": end,
                "report_end": end,
                "report_available_at": "2025-04-20T08:00:00+08:00",
            }
        ],
    }


def row():
    return {"as_of": "2025-07-01T08:00:00+08:00", "base": "2025-07-01", "target": "2025-07-02"}


@pytest.mark.parametrize(("days", "expected"), [(90, 1), (91, 0.5), (180, 0.5), (181, 0)])
def test_age_boundaries(days, expected):
    assert f.age_factor(days) == expected


def test_weight_minimum_and_decay_do_not_change_trigger_denominator():
    sessions = ["2025-06-30", "2025-07-01", "2025-07-02"]
    values, proof = f.vector(row(), [event()], [report()], sessions, True)
    assert values[13] == 0.02
    assert proof["trigger_original_weight"] == 0.04
    assert proof["trigger"]
    assert proof["events"][0]["original_weight"] == 0.04


def test_future_report_and_future_event_cannot_change_input():
    sessions = ["2025-06-30", "2025-07-01", "2025-07-02"]
    expected = f.vector(row(), [event()], [report()], sessions)
    later = report("2025-06-30", "2025-07-20T08:00:00+08:00", 99)
    assert f.vector(row(), [event(), event("2025-07-01T08:00:01+08:00")], [report(), later], sessions) == expected


def test_no_processed_body_keeps_unknown_instead_of_no_event():
    values, proof = f.vector(row(), [], [report()], ["2025-07-01", "2025-07-02"])
    assert values[:19] == [None] * 19
    assert not proof["trigger"]


def test_quarantined_body_is_unknown():
    e = event()
    e["status"] = "QUARANTINED_EXTRACTION"
    values, proof = f.vector(row(), [e], [report()], ["2025-07-01", "2025-07-02"])
    assert values[12] is None and values[14] is None and not proof["trigger"]


def test_181_day_old_holdings_do_not_trigger_or_aggregate_b3():
    e = event(end="2024-12-31")
    values, proof = f.vector(row(), [e], [report()], ["2025-07-01", "2025-07-02"], True)
    assert values[13] == 0 and values[2] is None and not proof["trigger"]
    assert values[12] == 1 and values[14] == 1  # 真实计数不乘衰减。


@pytest.mark.parametrize("age,expected", [(0, 1), (1, 0.8), (2, 0.6), (3, 0.4), (4, 0.2)])
def test_five_day_time_decay(age, expected):
    sessions = ["2025-06-25", "2025-06-26", "2025-06-27", "2025-06-30", "2025-07-01", "2025-07-02"]
    e = event(sessions[4 - age] + "T00:00:00+08:00")
    _, proof = f.vector(row(), [e], [report()], sessions, True)
    assert proof["events"][0]["factor"] == pytest.approx(0.5 * expected)


def test_numeric_coverage_does_not_count_company_twice():
    first, second = event(), event("2025-07-01T01:00:00+08:00")
    second["id"] = "second"
    second["numeric"]["revenue_yoy"]["value"] = 0.4
    values, _ = f.vector(row(), [first, second], [report()], ["2025-07-01", "2025-07-02"])
    assert values[2] == 0.4 and values[15] == 0.04


def test_negative_yoy_and_cross_metric_columns():
    q = {
        "quote": "营业收入同比下降20%，归属于上市公司股东的净利润同比增长30%",
        "value_text": "下降20%",
        "period_text": "2024年度",
    }
    assert f.metric_yoy(q, "revenue_yoy", "2024FY")["value"] == -0.2
    assert f.metric_yoy(q, "profit_yoy", "2024FY") is None
    assert f.metric_yoy(q, "revenue_yoy", "2025FY") is None


@pytest.mark.parametrize(
    "quote",
    [
        "归属于上市公司股东的扣除非经常性损益的净利润同比增长20%",
        "利润总额同比增长20%",
        "归属于上市公司股东的净利润亏损同比增加20%",
    ],
)
def test_absolute_deducted_and_negative_base_not_parent_profit_yoy(quote):
    assert (
        f.metric_yoy({"quote": quote, "value_text": "20%", "period_text": "2024年度"}, "profit_yoy", "2024FY") is None
    )


def test_guidance_requires_comparable_nonoverlapping_range():
    old = {"period": "2024FY", "currency": "CNY", "low": 1, "high": 2}
    assert f.compare_guidance(old, {**old, "low": 2.1, "high": 3}) == "guidance_up"
    assert f.compare_guidance(old, {**old, "low": 0.5, "high": 0.9}) == "guidance_down"
    assert f.compare_guidance(old, {**old, "low": 1.5, "high": 3}) is None
    assert f.compare_guidance(old, {**old, "period": "2025FY", "low": 4, "high": 5}) is None


def test_ratio_requires_known_same_currency_denominator():
    n = {"value": 100, "currency": "CNY"}
    assert f.numeric_ratio(n, None) is None
    assert f.numeric_ratio(n, {"value": 0, "currency": "CNY"}) is None
    assert f.numeric_ratio(n, {"value": 200, "currency": "USD"}) is None
    assert f.numeric_ratio(n, {"value": 200, "currency": "CNY"})["value"] == 0.5


def test_buyback_price_cap_cannot_be_amount_cap():
    body = "公司2025年4月1日审议通过了回购方案。支付的金额为6,703,400.00元。回购金额1.5亿元~3亿元回购价格上限40元"
    b = f.buyback(body)
    assert b["cap"]["value"] == 300000000
    assert b["ratio"]["value"] == pytest.approx(6703400 / 300000000)
    assert f.buyback("支付的金额为100元。回购价格上限40元")["ratio"] is None


def test_buyback_repeat_and_cancellation_are_not_new_execution():
    body = "公司2025年4月1日审议通过了回购方案。已支付的总金额为200万元。回购资金总额不超过1000万元"
    docs = [
        {
            "id": str(i),
            "event_id": str(i),
            "kind": "company",
            "stock_code": "000001.SZ",
            "title": title,
            "body": body,
            "body_available_at": stamp,
            "body_sha256": str(i),
            "published_date": stamp[:10],
        }
        for i, title, stamp in [
            (1, "回购股份进展公告", "2025-06-01T00:00:00+08:00"),
            (2, "回购股份进展公告", "2025-07-01T00:00:00+08:00"),
            (3, "回购股份注销完成公告", "2025-07-02T00:00:00+08:00"),
        ]
    ]
    events = f.build_events(docs, {}, [report()])
    assert [e["status"] == "QUALIFIED" for e in events] == [True, False, False]


def test_risk_and_cancel_require_previous_explicit_reference():
    previous = event("2025-06-01T00:00:00+08:00")
    previous.update(title="关于签订日常经营重大合同的公告", order={"value": 100})
    current = event()
    fact = "公司决定终止本次重大合同。"
    assert f.referenced_progress(current, [previous], f.clean(fact), [fact])[0] == []
    assert f.referenced_progress(current, [previous], f.clean(previous["title"] + fact), [fact])[0] == ["order_cancel"]
    previous.pop("order")
    previous["flags"] = ["risk_new"]
    previous["title"] = "关于公司主要生产线停产的公告"
    fact = "公司主要生产线现已恢复生产。"
    assert f.referenced_progress(current, [previous], f.clean(previous["title"] + fact), [fact])[0] == ["risk_resolved"]


def test_late_nav_blocks_contiguous_window_without_skipping():
    sessions = [(date(2024, 1, 1) + timedelta(days=i)).isoformat() for i in range(90)]
    nav = {d: {"unit_nav": str(1 + i * 0.01), "available_at": d + "T00:00:00+08:00"} for i, d in enumerate(sessions)}
    nav[sessions[80]]["available_at"] = sessions[89] + "T08:00:00+08:00"
    end, lag, _ = c.nav_inputs.choose_nav_window(sessions, nav, sessions[85])
    assert end == sessions[79] and lag == 5
    assert c.nav_inputs.choose_nav_window(sessions, nav, sessions[89])[0] == sessions[88]


def test_date_only_and_timezone_boundaries():
    assert c.nav_inputs.available_at("2025-01-01") == "2025-01-02T08:00:00+08:00"
    assert c.moment("2025-01-02T00:00:00+00:00") == c.moment("2025-01-02T08:00:00+08:00")
    with pytest.raises(ValueError):
        c.moment("2025-01-02T08:00:00")


def test_training_only_preprocessing():
    rows = [
        {"groups": {"N": [1.0] * 8, "NE": [2.0] * 7}, "E": [None] * 20},
        {"groups": {"N": [3.0] * 8, "NE": [4.0] * 7}, "E": [None] * 20},
    ]
    _, state = train.transform(rows, "B1")
    before = state["scaler"].mean_.copy()
    extreme = copy.deepcopy(rows[:1])
    extreme[0]["groups"]["N"] = [1e12] * 8
    train.transform(extreme, "B1", state)
    np.testing.assert_array_equal(before, state["scaler"].mean_)
    assert len(before) == 55  # 35业务列 + 20个全空列的缺失指示。


def test_no_trigger_exact_baseline_and_fixed_mix():
    b, e = [0.50000000000001, 0.0, 0.49999999999999], [0.1, 0.2, 0.7]
    assert train.blend(b, e, False) == b
    np.testing.assert_array_equal(train.blend(b, e, True), 0.75 * np.array(b) + 0.25 * np.array(e))


def test_started_failure_consumes_budget_and_cannot_silent_retry(tmp_path, monkeypatch):
    monkeypatch.setitem(c.LIMITS, "fit", 2)
    c.reserve(tmp_path, "fit", "failed")
    with pytest.raises(ValueError, match="ALREADY"):
        c.reserve(tmp_path, "fit", "failed")
    c.reserve(tmp_path, "fit", "retry", {"retry_of": "failed"})
    with pytest.raises(ValueError, match="EXHAUSTED"):
        c.reserve(tmp_path, "fit", "third")


def test_mutated_freeze_refuses_evaluation(tmp_path):
    p = tmp_path / "a.json"
    p.write_text("{}", encoding="utf-8")
    manifest = {"files": {str(p): c.io.sha(p)}}
    p.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        c.verify_files(manifest)


@pytest.mark.parametrize(
    ("stamp", "status"),
    [
        ("2025-07-01T07:59:00+08:00", "TIMELY"),
        ("2025-07-01T08:00:01+08:00", "LATE"),
        ("2025-07-02T07:00:00+08:00", "HISTORICAL_BACKFILL"),
    ],
)
def test_forward_clock_is_not_target_date_backdating(stamp, status):
    assert forward.timing_status("2025-07-01", ["2025-07-01", "2025-07-02"], stamp) == status


def test_answers_append_after_maturity_only(tmp_path):
    p = {"base": "2025-07-01", "target": "2025-07-02", "generated_at": "2025-07-01T07:59:00+08:00"}
    nav = {
        "2025-07-01": {"unit_nav": "1.2", "available_at": "2025-07-02T08:00:00+08:00"},
        "2025-07-02": {"unit_nav": "1.1", "available_at": "2025-07-03T08:00:00+08:00"},
    }
    assert forward.mature_answer(p, nav, "2025-07-03T08:00:00+08:00") is None
    answer = forward.mature_answer(p, nav, "2025-07-03T08:00:01+08:00")
    assert answer["label"] == "DOWN"
    path = tmp_path / "answers.jsonl"
    forward.append_unique(path, answer, ("target", "version"))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        forward.append_unique(path, answer, ("target", "version"))
    assert path.read_bytes() == before


def test_bootstrap_fixed_and_requires_real_sample_size():
    with pytest.raises(ValueError):
        forward.bootstrap_difference([1] * 119)
    assert forward.bootstrap_difference([0] * 120) == [0, 0]
    assert forward.bootstrap_difference([1, -1, 0] * 40) == forward.bootstrap_difference([1, -1, 0] * 40)
