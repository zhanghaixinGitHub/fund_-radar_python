"""第三轮边界与故障注入；全部数值拟合仅用合成行，不读取真实训练包。"""

from copy import deepcopy
from datetime import date, timedelta

import joblib
import numpy as np
import pytest
from app.services import fund_002112_round3_data as d
from app.services import fund_002112_round3_model as m
from app.services import fund_002112_round3_news as news
from app.services import fund_002112_round3_run as run


def nav_fixture():
    days = [(date(2023, 1, 1) + timedelta(days=i)).isoformat() for i in range(90)]
    rows = {day: {"nav": str(1 + i / 1000), "ann_date": day, "source_hash": str(i)} for i, day in enumerate(days)}
    return days, rows


def test_nav_same_day_announcements_excluded_and_target_still_u_t():
    days, rows = nav_fixture()
    rows[days[-2]]["ann_date"] = days[-1]
    result = d.nav_window(rows, days, days[-1])
    assert (result["S"], result["T"], result["U"], result["lag_sessions"]) == (days[-3], days[-2], days[-1], 1)
    before = deepcopy(result)
    rows[days[-2]]["nav"] = "10000000"
    rows[days[-1]]["nav"] = "0.0000001"
    assert d.nav_window(rows, days, days[-1]) == before
    assert d.direction(rows[days[-2]]["nav"], rows[days[-1]]["nav"]) == "DOWN"


def test_nav_gap_does_not_join_noncontiguous_days():
    days, rows = nav_fixture()
    del rows[days[-4]]
    result = d.nav_window(rows, days, days[-1])
    assert result["S"] == days[-5]
    assert len(result["nav_dates"]) == 61


def test_nav_later_revision_and_unknown_publication_are_not_backdated():
    days, rows = nav_fixture()
    rows[days[-2]]["revised_at"] = days[-1] + "T01:00:00+08:00"
    rows[days[-3]]["ann_date"] = None
    assert d.nav_window(rows, days, days[-1])["S"] == days[-4]


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("1.00000000", "1.00000000", "FLAT"),
        ("1.0000000000", "1.0000000001", "UP"),
        ("1.0000000001", "1.0000000000", "DOWN"),
    ],
)
def test_exact_decimal_three_classes(a, b, expected):
    assert d.direction(a, b) == expected


def report(pub="2023-04-01", **extra):
    return {
        "published_date": pub,
        "report_end": "2023-03-31",
        "report_type": "QUARTER",
        "full_stock_disclosure": False,
        "raw": {"sha256": pub},
        **extra,
    }


def test_report_same_day_exclusion_and_full_disclosure_priority():
    old = report("2023-04-01")
    full = report("2023-04-01", full_stock_disclosure=True)
    future = report("2023-04-03", report_end="2023-06-30")
    assert d.select_report([old, full, future], [], "2023-04-03") is full


def test_latest_report_missing_cannot_fall_back():
    old = report()
    with pytest.raises(ValueError, match="LATEST_DISCLOSURE_MISSING"):
        d.select_report([old], [report("2023-04-02", full_stock_disclosure=True)], "2023-04-03")


def test_report_conflicting_versions_rejected():
    with pytest.raises(ValueError, match="VERSION_CONFLICT"):
        d.select_report([report(), report(raw={"sha256": "conflict"})], [], "2023-04-03")


def test_report_revision_cannot_enter_earlier_target():
    with pytest.raises(ValueError, match="NO_PRIOR_PUBLIC"):
        d.select_report([report(version_publication_date="2023-04-04")], [], "2023-04-03")


def test_missing_positive_holding_rejected_even_tiny_weight():
    r = report(
        disclosed_nav_pct="0.00001",
        stock_nav_pct="1",
        holdings=[{"stock_code": "000001.SZ", "nav_weight_pct": "0.00001"}],
    )
    with pytest.raises(ValueError, match="POSITIVE_HOLDING_QUOTE_MISSING"):
        d.exposure(r, lambda day: {}, {}, ["2023-04-01"] * 21)


def test_null_holding_weight_is_not_zero():
    r = report(disclosed_nav_pct="1", stock_nav_pct="1", holdings=[{"stock_code": "000001.SZ", "nav_weight_pct": None}])
    with pytest.raises((ValueError, TypeError)):
        d.exposure(r, lambda day: {}, {}, ["2023-04-01"] * 21)


@pytest.mark.parametrize(
    "publication,mature,ok",
    [
        ("2023-03-30", "2023-03-31T08:00:00+08:00", True),
        ("2023-04-01", "2023-03-31T08:00:00+08:00", False),
        ("2023-03-31", "2023-04-01T00:00:00+08:00", False),
        ("2023-03-31", "2023-04-01T08:00:00+08:00", False),
    ],
)
def test_stage_requires_published_and_mature_before_midnight(publication, mature, ok):
    assert (
        d.eligible_at({"target": "2023-03-30", "label_publication": publication, "mature_at": mature}, "2023-04-01")
        is ok
    )


def synthetic():
    rng = np.random.default_rng(36)
    rows = [
        {
            "fund_code": "002112",
            "family": "F",
            "target": f"2023-{i:04d}",
            "actual_direction": d.CLASSES[i % 3],
            "x": rng.normal(size=20).tolist(),
        }
        for i in range(330)
    ]
    return rows


@pytest.mark.parametrize("variant", list(m.VARIANTS))
def test_fixed_model_reproduction_restore_and_train_only_preprocessing(tmp_path, variant):
    rows = synthetic()
    exams = deepcopy(rows[:9])
    for r in exams:
        r["x"] = [v + 10000 for v in r["x"]]
    weights = [1.0] * len(rows)
    first, first_scores = m.fit(rows, exams, weights, variant)
    replay, replay_scores = m.fit(rows, exams, weights, variant)
    assert m.state_hash(first) == m.state_hash(replay)
    assert m.compare_predictions(first_scores["exam"], replay_scores["exam"]) == 0
    np.testing.assert_allclose(first["scaler"].mean_, np.mean(m.matrix(rows, m.VARIANTS[variant]), axis=0))
    path = tmp_path / "self-generated.joblib"
    joblib.dump(first, path)
    assert m.compare_predictions(first_scores["exam"], m.predict(joblib.load(path), exams)) <= 1e-12
    if variant == "T20":
        assert first["classifier"].n_iter_ == 100
        assert first["classifier"].do_early_stopping_ is False
        assert first["classifier"].validation_fraction is None


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), None])
def test_missing_input_not_accepted_by_tree(invalid):
    rows = synthetic()
    rows[0]["x"][12] = invalid
    with pytest.raises(ValueError):
        m.fit(rows, rows[:3], [1.0] * len(rows), "T20")


def test_fixed_tie_order():
    assert m.directions([[1 / 3] * 3, [0.5, 0.0, 0.5]]) == ["FLAT", "UP"]


def test_family_weights_and_duplicate_gate():
    rows = [
        {"family": "A", "fund_code": "002112", "target": "2023-01-01", "actual_direction": "UP"},
        {"family": "A", "fund_code": "002112", "target": "2023-01-02", "actual_direction": "DOWN"},
        {"family": "B", "fund_code": "002170", "target": "2023-01-01", "actual_direction": "FLAT"},
    ]
    assert d.weights(rows) == [0.75, 0.75, 1.5]
    bad = d.gate(rows + [rows[0]], {"002112": "A", "002170": "B"})
    assert not bad["checks"]["no_duplicate_family_date"]
    assert not bad["checks"]["all_11_families"]


def test_attempt_is_persisted_before_child_and_failure_never_refunds(tmp_path, monkeypatch):
    d.write_once(tmp_path / "protocol.json", {})
    slot = run.SLOTS[0]

    def fail(p, command, s):
        assert (p / "attempts" / (s + ".json")).exists()
        raise ValueError("SYNTHETIC_CRASH")

    monkeypatch.setattr(run, "child", fail)
    with pytest.raises(ValueError, match="SYNTHETIC_CRASH"):
        run.ensure_slot(tmp_path, slot)
    assert len(run.attempts(tmp_path)) == 1
    with pytest.raises(ValueError, match="INCOMPLETE_CONSUMED_SLOT"):
        run.ensure_slot(tmp_path, slot)
    assert len(run.attempts(tmp_path)) == 1


def test_budget_exhaustion_and_unknown_slot(tmp_path):
    d.write_once(tmp_path / "protocol.json", {})
    for i in range(24):
        d.write_once(tmp_path / "attempts" / f"reserved-{i}.json", {})
    with pytest.raises(ValueError, match="BUDGET_EXHAUSTED"):
        run.reserve(tmp_path, run.SLOTS[0])
    with pytest.raises(ValueError, match="UNKNOWN_SLOT"):
        run.reserve(tmp_path, "new-run-new-slot")


def test_checkpoint_reuse_makes_no_child_call(tmp_path, monkeypatch):
    slot = run.SLOTS[0]
    d.write_once(tmp_path / "protocol.json", {})
    run.reserve(tmp_path, slot)
    value = {"slot": slot, "attempt_hash": d.file_hash(tmp_path / "attempts" / (slot + ".json")), "files": {}}
    d.write_once(tmp_path / "checkpoints" / (slot + ".json"), value)
    monkeypatch.setattr(run, "child", lambda *args: pytest.fail("unexpected fitting"))
    assert run.ensure_slot(tmp_path, slot) == value
    assert len(run.attempts(tmp_path)) == 1


def test_immutable_decision_and_path_escape(tmp_path):
    d.write_once(tmp_path / "decision.json", {"passed": False})
    with pytest.raises(ValueError, match="IMMUTABLE"):
        d.write_once(tmp_path / "decision.json", {"passed": True})
    with pytest.raises(ValueError, match="INVALID_RUN_ID"):
        run.directory("../../training-runs")


def test_unreserved_worker_cannot_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "fit", lambda *args: pytest.fail("unexpected fitting"))
    with pytest.raises(ValueError, match="UNRESERVED"):
        run.fit_worker(tmp_path, run.SLOTS[0])


def test_failed_history_blocks_full_worker(tmp_path, monkeypatch):
    slot = "FULL-T20-main"
    d.write_once(tmp_path / "attempts" / (slot + ".json"), {})
    d.write_once(tmp_path / "historical-decision.json", {"passed": False})
    monkeypatch.setattr(run, "load_frozen", lambda *args, **kwargs: {"eligibility": {"passed": True}})
    with pytest.raises(ValueError, match="HISTORICAL_GATE_FAILED"):
        run.fit_worker(tmp_path, slot)
    assert not (tmp_path / "started" / (slot + ".json")).exists()


def news_bundle():
    return {
        "cards": [
            {
                "id": key,
                "announcement_id": key,
                "published_date": "2024-03-01",
                "target": "2024-03-04",
                "holding": {"public_date": "2024-01-16", "nav_weight_pct": "1"},
                "events": {"ids": topics, "aggregation": "EVIDENCE_ONLY_NO_ADDITION"},
                "facts": {"amounts": news.amounts(key)},
                "next_day_direction": None,
                "impact_magnitude": None,
                "prediction_eligible": False,
                "unknowns": ["未知"],
                "analysis": {"horizon": "未知"},
                "uncovered": "其他持仓",
                "evidence": {"anchors": []},
                "company": {"stock_code": "001"},
            }
            for key, topics in news.TOPICS.items()
        ]
    }


def comparison_fixture():
    # 每季 6 天，三类各 2 天；对照每类命中 1 天，候选全部命中。
    rows = [
        {
            "fund_code": "002112",
            "target": f"2023-{month:02d}-{day + 1:02d}",
            "actual_direction": d.CLASSES[day % 3],
            "input_hash": f"{month}-{day}",
            "direction": d.CLASSES[day % 3],
        }
        for month in (4, 7, 10)
        for day in range(6)
    ]
    result = {v: deepcopy(rows) for v in m.VARIANTS}
    for v in ("N7", "L20"):
        for i, r in enumerate(result[v]):
            if i % 6 >= 3:
                r["direction"] = d.CLASSES[(d.CLASSES.index(r["actual_direction"]) + 1) % 3]
    return result


def test_comparison_counts_pairwise_and_fixed_pass_rule():
    c = m.comparison(comparison_fixture(), historical=True)
    assert c["numerical_passed"]
    assert c["models"]["T20"]["correct"] == 18
    assert c["models"]["L20"]["correct"] == 9
    assert c["pairs"]["L20"]["gained"] == 9 and c["pairs"]["L20"]["lost"] == 0
    assert c["models"]["T20"]["confusion"]["FLAT"]["FLAT"] == 6


def test_more_total_days_cannot_compensate_class_regression():
    values = comparison_fixture()
    for r in values["T20"]:
        if r["actual_direction"] == "UP":
            r["direction"] = "DOWN"
    c = m.comparison(values, historical=True)
    assert c["models"]["T20"]["correct"] == 12 > c["models"]["L20"]["correct"]
    assert not c["checks"]["class_not_worse_UP"] and not c["numerical_passed"]


def test_matching_control_total_is_not_strict_improvement():
    values = comparison_fixture()
    values["T20"] = deepcopy(values["L20"])
    c = m.comparison(values, historical=True)
    assert not c["checks"]["total_strictly_above_controls"]
    assert not c["numerical_passed"]


def test_different_exam_date_rejected():
    values = comparison_fixture()
    values["T20"][0]["target"] = "2023-04-30"
    with pytest.raises(ValueError, match="COMPARISON_DATES_CHANGED"):
        m.comparison(values, historical=True)


def test_expired_sources_stop_before_fit(tmp_path):
    p = tmp_path / "source.txt"
    p.write_text("synthetic", encoding="utf-8")
    with pytest.raises(ValueError, match="SOURCE_EXPIRED"):
        run.verify_sources(
            {"files": {str(p): d.file_hash(p)}, "receipts": [{"expires_at": "2000-01-01T00:00:00+08:00"}]}
        )


def test_changed_model_rejected_before_deserialization(tmp_path, monkeypatch):
    slot = run.SLOTS[0]
    p = tmp_path / "models" / (slot + ".joblib")
    p.parent.mkdir()
    p.write_bytes(b"untrusted bytes")
    d.write_once(tmp_path / "model-manifests" / (slot + ".json"), {"file_sha256": "wrong"})
    monkeypatch.setattr(run, "load_frozen", lambda *args, **kwargs: {})
    monkeypatch.setattr(joblib, "load", lambda *args: pytest.fail("must not deserialize"))
    with pytest.raises(ValueError, match="UNTRUSTED_MODEL"):
        run.restore_worker(tmp_path, slot)


@pytest.mark.parametrize(
    "mutation",
    [
        "plan_as_actual",
        "guarantee_as_loss",
        "currency",
        "duplicate",
        "merge",
        "monthly_additive",
        "unknown_flat",
        "unknown_zero",
        "future_report",
    ],
)
def test_news_semantic_boundary_mutants(mutation):
    b = news_bundle()
    assert news.validate(b)["passed"]
    cards = {r["id"]: r for r in b["cards"]}
    if mutation == "plan_as_actual":
        cards["N02"]["facts"]["amounts"][0]["status"] = "ACTUAL"
    elif mutation == "guarantee_as_loss":
        cards["N07"]["facts"]["amounts"][0]["status"] = "REALIZED_LOSS"
    elif mutation == "currency":
        cards["N10"]["facts"]["amounts"][0]["unit"] = "CNY"
    elif mutation == "duplicate":
        cards["N08"]["events"]["ids"] = ["NEW_GUARANTEE"]
    elif mutation == "merge":
        cards["N06"]["events"]["ids"] = ["CR999_KUNYAO_GUARANTEE"]
    elif mutation == "monthly_additive":
        cards["N12"]["events"]["aggregation"] = "SUM_DAILY_AND_MONTHLY"
    elif mutation == "unknown_flat":
        cards["N01"]["next_day_direction"] = "FLAT"
    elif mutation == "unknown_zero":
        cards["N01"]["impact_magnitude"] = 0
    else:
        cards["N01"]["holding"]["public_date"] = "2024-03-04"
    with pytest.raises(ValueError):
        news.validate(b)
