"""剩余专项的队列、时间、单因素、校准、来源和独立测试门槛。"""

import math
from copy import deepcopy
from datetime import date, timedelta
from types import SimpleNamespace

import pytest
from app.schemas.cash_reinvestment_samples import CashSampleRequest
from app.schemas.direction_followup import StudyCohort, StudyInput
from app.services import direction_followup_data as data
from app.services import direction_followup_runner as runner
from app.services.direction_followup_models import execute_job
from app.services.direction_followup_protocol import plan
from app.services.direction_training_dataset import FUNDS
from app.services.direction_training_evaluation import metrics
from app.services.trading_calendar import load_calendar
from pydantic import ValidationError
from tests.test_cash_reinvestment_samples import rows_for
from tests.test_trading_nav_window import SOURCE


def cohort_metadata():
    rows = []
    for f in (*FUNDS, "007045"):
        rows.append(
            {
                "fund_code": f,
                "fund_master_id": f,
                "fund_type": "STOCK",
                "source_fund_type": "股票型",
                "status": "ACTIVE",
                "source_code": "TEST",
                "fund_name": f + "指数增强C",
                "established_date": "2020-01-01",
                "market": "O",
                "benchmark": "沪深300",
            }
        )
    return {
        "catalog": rows,
        "coverage": [
            {"fund_code": r["fund_code"], "n": 970, "first_date": "2021-01-04", "last_date": "2024-12-31"} for r in rows
        ],
        "sources": [
            {
                "source_code": "TEST",
                "enabled": True,
                "authorization_recorded": True,
                "authorized_api_names": ["fund_nav", "fund_div"],
            }
        ],
    }


def test_cohort_keeps_original_and_excludes_foreign_and_duplicate_without_using_scores():
    meta = cohort_metadata()
    duplicate = {**meta["catalog"][-1], "fund_code": "007046"}
    foreign = {**duplicate, "fund_code": "007047", "fund_master_id": "other", "fund_name": "越南QDII股票C"}
    meta["catalog"] += [duplicate, foreign]
    meta["coverage"] += [{**meta["coverage"][-1], "fund_code": r["fund_code"]} for r in (duplicate, foreign)]
    cohort, decisions = data.select_cohort(meta, plan())
    assert set(cohort.funds) == set((*FUNDS, "007045"))
    by_code = {r["fund_code"]: r for r in decisions}
    assert "DUPLICATE_SHARE_OR_PRODUCT" in by_code["007046"]["reasons"]
    assert "DIFFERENT_MARKET_CALENDAR" in by_code["007047"]["reasons"]
    with pytest.raises(ValidationError):
        StudyCohort(funds=("001632", "006730", "007045"))
    with pytest.raises(ValidationError):
        CashSampleRequest(fundCode="007045", cutoffDate="2023-06-20")


def test_extended_cash_input_stays_on_same_formula_and_protected_year_is_rejected(monkeypatch):
    cohort = StudyCohort(funds=(*FUNDS, "007045"))
    cutoff = date(2023, 6, 20)
    first, _ = data.historical_input("007045", cutoff, SOURCE, rows_for(), (), cohort)
    original, _ = data.historical_input("006730", cutoff, SOURCE, rows_for(), (), cohort)
    assert first.x == original.x and first.anchor_lag_sessions == 1

    def forbidden(*args, **kwargs):
        pytest.fail("2025 must be rejected before calculating values")

    import app.services.cash_reinvestment_samples as cash

    monkeypatch.setattr(cash, "build_cash_return_series", forbidden)
    with pytest.raises(ValueError, match="DATE"):
        data.answer("007045", date(2025, 1, 2), (), (), cohort, value=True)


def test_market_uses_previous_session_and_missing_stays_missing():
    cutoff = date(2023, 6, 20)
    item = StudyInput(
        fund="001632",
        cutoff=cutoff,
        anchor=date(2023, 6, 19),
        available_at=cutoff,
        anchor_lag_sessions=1,
        x=(0.1, 1, 0.2, 0.3, 0.4, 0.5, 0),
        input_hash="a" * 64,
    )
    calendar = load_calendar()
    idx = calendar.at_or_before_index(cutoff) - 1
    dates = calendar.sessions[idx - 20 : idx + 1]
    prices = {str(d): str(100 + i) for i, d in enumerate(dates)}
    output, reason = data.add_market(item, prices)
    assert reason is None and len(output.x) == 10
    assert output.x[7] == pytest.approx(0.2) and output.x[9] == pytest.approx(0.8)
    prices[str(cutoff)] = "999999999"
    assert data.add_market(item, prices)[0] == output
    mismatched = item.model_copy(update={"anchor": cutoff, "anchor_lag_sessions": 0})
    assert data.add_market(mismatched, prices) == (None, "MARKET_FUND_ANCHOR_MISMATCH")
    del prices[str(dates[3])]
    assert data.add_market(item, prices) == (None, "MARKET_MISSING_SESSION")


def test_unauthorized_market_is_not_requested(monkeypatch):
    def forbidden():
        pytest.fail("no credentials should be read")

    import app.core.config as config

    monkeypatch.setattr(config, "get_settings", forbidden)
    result = data.market_prices({"sources": []}, plan())
    assert result["reason"] == "SOURCE_NOT_AUTHORIZED"


def test_market_requests_four_declared_years_and_stops_on_failure(monkeypatch):
    import time

    import app.core.config as config
    import app.integrations.tushare_market_reference as integration

    calls = []

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def list_index_daily(self, code, *, start_date, end_date):
            from decimal import Decimal

            calls.append((code, start_date, end_date))
            return (SimpleNamespace(trade_date=start_date, close_price=Decimal(100)),)

    monkeypatch.setattr(integration, "TushareMarketReferenceClient", Client)
    monkeypatch.setattr(
        config,
        "get_settings",
        lambda: SimpleNamespace(
            tushare_token=SimpleNamespace(get_secret_value=lambda: "test-only"),
            tushare_api_url="https://example.invalid",
        ),
    )
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    meta = {
        "sources": [
            {
                "source_code": "TUSHARE_PRO_FUND",
                "enabled": True,
                "authorization_recorded": True,
                "authorized_api_names": ["index_daily"],
                "rate_limit_per_minute": 60,
            }
        ],
        "market_index": [{"display_name": "沪深300"}],
    }
    result = data.market_prices(meta, plan())
    assert result["status"] == "DOWNLOADED_RESEARCH_SNAPSHOT" and len(calls) == 4
    assert [c[1].year for c in calls] == [2021, 2022, 2023, 2024]


def matrix_rows(start, count, funds, *, inverse=False):
    calendar = load_calendar()
    dates = [d for d in calendar.sessions if d >= start][:count]
    rows = []
    for f in funds:
        for i, d in enumerate(dates):
            signal = math.sin(i / 5)
            y = int(signal < 0 if inverse else signal >= 0)
            end = calendar.future_sessions(d)[-1]
            item = {
                "fund": f,
                "cutoff": str(d),
                "anchor": str(d),
                "available_at": str(d),
                "anchor_lag_sessions": 0,
                "x": [signal, 0.1, 0.2, 0.3, 0.4, 0.5, 0],
                "input_hash": "a" * 64,
            }
            label = {
                "fund": f,
                "cutoff": str(d),
                "base": str(d),
                "end": str(end),
                "available_at": str(end + timedelta(days=1)),
                "y": y,
                "future_return": ".1" if y else "-.1",
            }
            rows.append({"input": item, "answer": label})
    return rows


@pytest.fixture(scope="module")
def prepared():
    cohort = StudyCohort(funds=(*FUNDS, "007045"))
    fits = matrix_rows(date(2021, 4, 1), 260, cohort.funds)
    cal = matrix_rows(date(2023, 4, 3), 70, cohort.funds, inverse=True)
    exam = [r["input"] for r in matrix_rows(date(2023, 10, 9), 40, cohort.funds)]
    data = {
        "cohort": cohort.model_dump(mode="json"),
        "window": {
            "name": "DEV_2023_Q4_V2",
            "fit_end": "2023-03-31",
            "cal_end": "2023-09-30",
            "exam_end": "2023-12-31",
        },
        "FIT": fits,
        "CAL": cal,
        "EXAM": exam,
        "market_inputs": {},
    }
    for item in [r["input"] for r in fits + cal] + exam:
        if item["fund"] in FUNDS:
            data["market_inputs"][f"{item['fund']}:{item['cutoff']}"] = {**item, "x": [*item["x"], 0.2, 0.3, 0.4]}
    return data


def test_jobs_change_only_the_intended_factor(prepared):
    jobs = runner.build_jobs(prepared)
    assert jobs["ORIGINAL_7"]["exam"] == jobs["EXPANDED_7"]["exam"]
    assert jobs["ORIGINAL_7"]["fit"] == [r for r in jobs["EXPANDED_7"]["fit"] if r["input"]["fund"] in FUNDS]
    small, large = jobs["MARKET_MATCHED_7"], jobs["MARKET_10"]
    assert len(small["fit"]) == len(large["fit"]) and len(small["exam"]) == len(large["exam"])
    assert all(
        a["input"]["x"] == b["input"]["x"][:7] and a["answer"] == b["answer"]
        for a, b in zip(small["fit"], large["fit"], strict=True)
    )
    assert not small["cal"] and not large["cal"]


def test_actual_7_and_10_column_training_and_calibration_reuses_base(prepared):
    jobs = runner.build_jobs(prepared)
    original = execute_job(jobs["ORIGINAL_7"])
    market = execute_job(jobs["MARKET_10"])
    assert original["status"] == market["status"] == "PREDICTED"
    assert len(original["artifact"]["mean"]) == 7 and len(market["artifact"]["mean"]) == 10
    cal = jobs["CALIBRATED_6M"]
    cal["base"] = original["artifact"]
    result = execute_job(cal)
    assert result["artifact"]["base"] == original["artifact"] and not result["base_fitted"]
    assert result["artifact"]["slope"] < 0 and result["calibration_state"] == "REVERSED_PENDING_VALIDATION"
    changed = deepcopy(jobs["ORIGINAL_7"])
    changed["exam"][0]["x"] = [999] * 7
    assert execute_job(changed)["artifact"] == original["artifact"]


def test_labels_in_exam_or_late_fit_label_are_rejected(prepared):
    job = runner.build_jobs(prepared)["ORIGINAL_7"]
    job["exam"] = deepcopy(job["exam"])
    job["exam"][0]["y"] = 1
    with pytest.raises(ValidationError):
        execute_job(job)
    job = deepcopy(runner.build_jobs(prepared)["ORIGINAL_7"])
    job["fit"][0]["answer"]["available_at"] = "2023-04-01"
    with pytest.raises(ValueError, match="STAGE_BOUNDARY"):
        execute_job(job)


def test_test_gate_reports_all_prerequisites_without_opening_values():
    rejected = runner.test_gate(None, False, False)
    assert len(rejected["reasons"]) == 3 and not rejected["test_values_read"]
    assert runner.test_gate("frozen-candidate", True, True)["status"] == "ELIGIBLE_FOR_SEPARATE_FROZEN_TEST"


def test_score_requires_sealed_predictions_before_answers(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "load_plan", lambda folder: ({}, {}))
    monkeypatch.setattr(runner, "answer", lambda *args, **kwargs: pytest.fail("answers read too early"))
    with pytest.raises(FileNotFoundError):
        runner.score(tmp_path)


def test_probability_and_direction_are_separate_metrics():
    before = metrics([(1, 0.99), (0, 0.6)])
    after = metrics([(1, 0.7), (0, 0.51)])
    assert before["accuracy"] == after["accuracy"] == 0.5
    assert after["brier_score"] < before["brier_score"]
