"""公告字段误否决、因果缺失处理、真实时间边界及受限训练作业验证。"""

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import UUID

import pytest
from app.repositories.cash_reinvestment_samples import CashDividend, CashNavPoint
from app.repositories.feature_snapshot import FeatureSourceReadiness
from app.services import direction_nav_data as data
from app.services.direction_nav_models import execute_job
from app.services.direction_nav_protocol import VERSION
from app.services.direction_training_dataset import build_answer as legacy_answer
from app.services.direction_training_dataset import build_input as legacy_input
from app.services.direction_training_process import run_process
from app.services.trading_calendar import load_calendar

FUND = "001632"
CUTOFF = date(2023, 8, 1)
SOURCE = FeatureSourceReadiness(UUID(int=1), "TUSHARE_PRO_FUND", UUID(int=2), datetime(2026, 9, 9, tzinfo=UTC))


def points():
    days = [d for d in load_calendar().sessions if date(2021, 1, 1) <= d <= date(2024, 12, 31)]
    return {
        d: CashNavPoint(d, data.assumed_available(d), Decimal(100 + i % 17) / 100 + Decimal(i) / 10000)
        for i, d in enumerate(days)
    }


def dividend(day, **changes):
    return replace(CashDividend("event", day - timedelta(days=1), None, day, day, Decimal("0.01"), "实施"), **changes)


@pytest.mark.parametrize("ann", [None, date(2020, 1, 1), date(2026, 1, 1)])
def test_nav_ann_date_does_not_veto_or_change_features_and_raw_is_preserved(ann):
    nav = points()
    expected, audit, issues = data.build_input(FUND, CUTOFF, nav, ())
    changed = {d: replace(p, ann_date=ann) for d, p in nav.items()}
    actual, actual_audit, actual_issues = data.build_input(FUND, CUTOFF, changed, ())
    assert not issues and not actual_issues
    assert actual == expected and actual_audit == audit
    assert all(p.ann_date == ann for p in changed.values())
    old, old_issues = legacy_input(FUND, CUTOFF, SOURCE, tuple(changed.values()), ())
    assert old is None and old_issues


def test_common_clean_features_and_twenty_session_answer_match_legacy_math():
    nav = points()
    item, _, _ = data.build_input(FUND, CUTOFF, nav, ())
    old, issues = legacy_input(FUND, CUTOFF, SOURCE, tuple(nav.values()), ())
    assert not issues and item.anchor == old.anchor
    assert item.x == pytest.approx(old.x, abs=1e-12)
    answer, issues = data.build_answer(FUND, CUTOFF, nav, (), include_value=True)
    previous, old_issues = legacy_answer(FUND, CUTOFF, tuple(nav.values()), (), include_value=True)
    assert not issues and not old_issues and answer == previous
    assert answer.end == load_calendar().future_sessions(CUTOFF)[-1]


def test_current_day_and_future_values_never_enter_inputs():
    nav = points()
    expected = data.build_input(FUND, CUTOFF, nav, ())
    for day in nav:
        if day >= CUTOFF:
            nav[day] = replace(nav[day], unit_nav=Decimal(9999), ann_date=date(2021, 1, 1))
    assert data.build_input(FUND, CUTOFF, nav, ()) == expected
    assert expected[0].anchor == data.history_dates(CUTOFF)[-1] < CUTOFF


@pytest.mark.parametrize("indices", [(10,), (10, 45)])
def test_isolated_missing_dates_are_filled_only_in_feature_copy(indices):
    nav, dates = points(), data.history_dates(CUTOFF)
    for i in indices:
        del nav[dates[i]]
    before = deepcopy(nav)
    assert data.build_input(FUND, CUTOFF, nav, ())[0] is None
    item, audit, issues = data.build_input(FUND, CUTOFF, nav, (), tolerate=True)
    assert item is not None and not issues and nav == before
    assert len(audit["dates"]) == len(audit["values_used"]) == 61
    assert audit["missing_dates"] == [str(dates[i]) for i in indices]
    for i in indices:
        assert audit["values_used"][i] == str(nav[dates[i - 1]].unit_nav)
    clean = points()
    hidden = tuple(dates[i] for i in indices)
    simulated, synthetic_audit, _ = data.build_input(FUND, CUTOFF, clean, (), tolerate=True, hidden=hidden)
    assert simulated.x == item.x
    assert synthetic_audit["synthetic_missing_dates"] == list(map(str, hidden))


@pytest.mark.parametrize("index", [0, 40, 55, 56, 57, 58, 59, 60])
def test_endpoints_and_latest_six_observations_cannot_be_filled(index):
    nav, dates = points(), data.history_dates(CUTOFF)
    del nav[dates[index]]
    item, _, issues = data.build_input(FUND, CUTOFF, nav, (), tolerate=True)
    assert item is None and issues == ["NAV_GAP_AT_PROTECTED_DATE"]


@pytest.mark.parametrize(
    "indices,reason", [((10, 11), "NAV_GAP_CONSECUTIVE"), ((10, 20, 30), "NAV_GAP_COUNT_EXCEEDED")]
)
def test_consecutive_or_excessive_gaps_are_rejected(indices, reason):
    nav, dates = points(), data.history_dates(CUTOFF)
    for i in indices:
        del nav[dates[i]]
    assert data.build_input(FUND, CUTOFF, nav, (), tolerate=True)[2] == [reason]


@pytest.mark.parametrize("value", [None, Decimal(0), Decimal(-1), Decimal("NaN")])
def test_invalid_value_is_not_treated_as_fillable_absence(value):
    nav, dates = points(), data.history_dates(CUTOFF)
    nav[dates[10]] = replace(nav[dates[10]], unit_nav=value)
    assert data.build_input(FUND, CUTOFF, nav, (), tolerate=True)[2] == ["UNIT_NAV_INVALID"]


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_known_dividend_neighborhood_gaps_are_rejected(offset):
    nav, dates = points(), data.history_dates(CUTOFF)
    del nav[dates[20 + offset]]
    assert data.build_input(FUND, CUTOFF, nav, (dividend(dates[20]),), tolerate=True)[2] == [
        "NAV_GAP_AT_PROTECTED_DATE"
    ]


def test_dividend_publication_gates_and_future_event_invariance_remain():
    nav, dates = points(), data.history_dates(CUTOFF)
    expected = data.build_input(FUND, CUTOFF, nav, ())
    late = dividend(dates[20], ann_date=CUTOFF + timedelta(days=1))
    assert data.build_input(FUND, CUTOFF, nav, (late,)) == expected
    late_implementation = dividend(dates[20], implementation_ann_date=CUTOFF + timedelta(days=1))
    assert data.build_input(FUND, CUTOFF, nav, (late_implementation,)) == expected
    invalid = dividend(dates[20], process_status="预案")
    assert data.build_input(FUND, CUTOFF, nav, (invalid,))[2] == ["DIVIDEND_NOT_IMPLEMENTED"]
    known = dividend(dates[20])
    assert data.build_input(FUND, CUTOFF, nav, (known,))[0].x != expected[0].x


def test_answers_never_fill_and_maturity_is_next_session_not_ann_date():
    nav = points()
    expected, _ = data.build_answer(FUND, CUTOFF, nav, (), include_value=True)
    changed = {d: replace(p, ann_date=date(2026, 1, 1)) for d, p in nav.items()}
    assert data.build_answer(FUND, CUTOFF, changed, (), include_value=True)[0] == expected
    assert expected.available_at == data.assumed_available(expected.end)
    del changed[load_calendar().future_sessions(CUTOFF)[8]]
    assert data.build_answer(FUND, CUTOFF, changed, (), include_value=True)[0] is None


def test_protected_year_guard_and_label_boundary():
    with pytest.raises(ValueError, match="PROTECTED"):
        data.build_input(FUND, date(2025, 1, 1), {}, ())
    with pytest.raises(ValueError, match="PROTECTED"):
        data.build_answer(FUND, date(2025, 1, 1), {}, ())
    assert data.build_answer(FUND, date(2024, 12, 10), points(), ())[0] is None


def test_stress_masks_are_deterministic_nested_and_no_target_or_values_needed():
    one, two = (data.stress_dates(FUND, CUTOFF, (), n) for n in (1, 2))
    assert len(one) == 1 and len(two) == 2 and set(one).issubset(two)
    assert two == data.stress_dates(FUND, CUTOFF, (), 2)
    assert data.stress_dates(FUND, CUTOFF, (), 0) == ()
    dates = data.history_dates(CUTOFF)
    assert all(dates.index(d) in data.eligible_gap_indices(dates, ()) for d in two)
    assert abs(dates.index(two[0]) - dates.index(two[1])) > 1


def training_job():
    days = [d for d in load_calendar().sessions if date(2021, 1, 4) <= d <= date(2022, 11, 1)][:252]
    fit = []
    for fund in ("001632", "006730", "008888"):
        for i, day in enumerate(days):
            future = load_calendar().future_sessions(day)
            fit.append(
                {
                    "input": {
                        "fund": fund,
                        "cutoff": str(day),
                        "anchor": str(day),
                        "available_at": str(day),
                        "x": [float((i * j + 3) % 19) / 19 for j in range(1, 8)],
                        "input_hash": "a" * 64,
                    },
                    "answer": {
                        "fund": fund,
                        "cutoff": str(day),
                        "end": str(future[-1]),
                        "available_at": str(future[-1]),
                        "y": i % 2,
                        "future_return": "0.01" if i % 2 else "-0.01",
                    },
                }
            )
    exam = {
        "fund": FUND,
        "cutoff": "2023-07-03",
        "anchor": "2023-06-30",
        "available_at": "2023-07-03",
        "x": [0.1] * 7,
        "input_hash": "b" * 64,
    }
    return {
        "version": VERSION,
        "branch": "COMPLETE",
        "window": {"name": "TEST", "fit_end": "2022-12-31", "cal_end": "2023-06-30", "exam_end": "2023-09-30"},
        "fit": fit,
        "exam": {v: [deepcopy(exam)] for v in ("CLEAN", "DROP1", "DROP2", "NATURAL")},
    }


def test_worker_fits_restores_and_predicts_without_source_or_exam_labels():
    import sys

    output = run_process(training_job(), command=[sys.executable, "-m", "scripts.direction_nav_worker"])
    assert output["status"] == "PREDICTED"
    assert output["artifact"]["versions"]["research"] == VERSION
    assert len(output["scores"]["CLEAN"]) == 1
    assert 0 <= output["scores"]["CLEAN"][0] <= 1


def test_worker_rejects_exam_answers_and_fit_future_dates_before_fitting():
    job = training_job()
    job["exam"]["CLEAN"][0]["y"] = 1
    with pytest.raises(ValueError):
        execute_job(job)
    job = training_job()
    job["fit"][0]["answer"]["available_at"] = "2023-01-01"
    with pytest.raises(ValueError, match="TRAINING_STAGE_BOUNDARY"):
        execute_job(job)


def test_scoring_cannot_run_without_sealed_predictions(tmp_path):
    from app.services.direction_nav_runner import score

    with pytest.raises(FileNotFoundError):
        score(tmp_path)
