"""匹配市场实验的数据可得性、同题隔离、旧模型兼容和补数边界。"""

import json
import math
import sys
from copy import deepcopy
from datetime import date
from decimal import Decimal

import httpx
import pytest
from app.integrations.tushare import TushareIntegrationError
from app.integrations.tushare_market_reference import TushareIndexDaily, TushareMarketReferenceClient
from app.schemas.direction_training import DirectionInput
from app.services import direction_market_data as data
from app.services.direction_linear_analysis import evaluate
from app.services.direction_linear_models import execute_job, predict_model, validate_job
from app.services.direction_linear_protocol import (
    BASELINES,
    COVERAGE_VERSION,
    MARKET_BRANCHES,
    MARKET_VERSION,
    planned_dates,
    specification_for,
    study_windows,
)
from app.services.direction_market_runner import build_window, validate_mapping, window_predictions
from app.services.direction_training_artifacts import ROOT, digest, read_json, write_json
from app.services.direction_training_dataset import FUNDS
from app.services.direction_training_process import run_process
from app.services.trading_calendar import load_calendar


@pytest.fixture
def market_job():
    cal = load_calendar()
    days = [d for d in cal.sessions if date(2021, 1, 5) <= d <= date(2022, 3, 1)][:252]
    rows = []
    for fund in FUNDS:
        for i, day in enumerate(days):
            end = cal.future_sessions(day)[-1]
            rows.append(
                {
                    "input": {
                        "fund": fund,
                        "cutoff": str(day),
                        "anchor": str(cal.sessions[cal.at_or_before_index(day) - 1]),
                        "available_at": str(day),
                        "x": [math.sin(i + j) for j in range(7)],
                        "input_hash": "a" * 64,
                    },
                    "answer": {
                        "fund": fund,
                        "cutoff": str(day),
                        "end": str(end),
                        "available_at": str(end),
                        "y": i % 2,
                        "future_return": "0.01" if i % 2 else "-0.01",
                    },
                }
            )
    return {
        "version": MARKET_VERSION,
        "branch": "REFERENCE",
        "window": study_windows(MARKET_VERSION)[0],
        "fit": rows,
        "exam": [
            {
                "fund": f,
                "cutoff": "2023-01-03",
                "anchor": "2022-12-30",
                "available_at": "2023-01-03",
                "x": [0.1] * 7,
                "input_hash": "b" * 64,
            }
            for f in FUNDS
        ],
    }


def prices():
    return {
        str(d): str(100 + i / 10)
        for i, d in enumerate(load_calendar().sessions)
        if date(2021, 1, 1) <= d <= date(2024, 12, 31)
    }


def test_all_quarter_dates_and_gates_are_unchanged():
    assert study_windows(MARKET_VERSION) == study_windows(COVERAGE_VERSION)
    assert [len(planned_dates(MARKET_VERSION, w)) for w in study_windows(MARKET_VERSION)] == [
        59,
        59,
        64,
        60,
        58,
        59,
        64,
        40,
    ]
    with pytest.raises(ValueError, match="DEDICATED_FREEZE"):
        specification_for(MARKET_VERSION)


def test_reference_matches_existing_fitter_without_changing_old_contract(market_job):
    original = deepcopy(market_job)
    original["version"] = COVERAGE_VERSION
    expected, actual = execute_job(original), execute_job(market_job)
    assert expected["scores"] == actual["scores"]
    for k in ("mean", "scale", "coefficients", "intercept", "train_hash", "train_counts"):
        assert expected["models"]["POOLED"][k] == actual["models"]["POOLED"][k]
    with pytest.raises(ValueError):
        DirectionInput.model_validate({**market_job["exam"][0], "x": [1.0] * 10})


def test_ten_feature_worker_and_numeric_restore(market_job):
    market_job["branch"] = "MATCHED_MARKET"
    for row in market_job["fit"]:
        row["input"]["x"] += [row["input"]["x"][0] / 2, 0.03, row["input"]["x"][1] / 2]
    for row in market_job["exam"]:
        row["x"] += [0.05, 0.03, 0.05]
    output = run_process(market_job, command=[sys.executable, "-m", "scripts.direction_linear_worker"])
    assert output["status"] == "PREDICTED", output
    assert output["models"]["POOLED"]["indices"] == list(range(10))
    assert output["models"]["POOLED"]["C"] == 1
    _, items = validate_job(market_job)
    assert predict_model(output["models"]["POOLED"], items) == output["scores"]


@pytest.mark.parametrize(
    "change", ["answer_in_exam", "late_label", "new_year", "current_close", "dimensions", "branch", "window", "nan"]
)
def test_worker_rejects_leakage_and_unplanned_changes(market_job, change):
    if change == "answer_in_exam":
        market_job["exam"][0]["y"] = 1
    elif change == "late_label":
        market_job["fit"][0]["answer"]["available_at"] = "2022-07-01"
    elif change == "new_year":
        market_job["exam"][0]["cutoff"] = "2025-01-02"
    elif change == "current_close":
        market_job["exam"][0]["anchor"] = "2023-01-03"
    elif change == "dimensions":
        market_job["exam"][0]["x"].append(0)
    elif change == "branch":
        market_job["branch"] = "NEW_BEST_INDEX"
    elif change == "window":
        market_job["window"]["fit_end"] = "2022-12-31"
    else:
        market_job["exam"][0]["x"][0] = float("nan")
    with pytest.raises(ValueError):
        validate_job(market_job)


def test_market_formula_endpoint_and_future_values_are_isolated(market_job):
    raw = market_job["exam"][0]
    history = prices()
    actual, reason = data.augment(raw, history)
    assert reason is None
    cal = load_calendar()
    end = cal.at_or_before_index(date.fromisoformat(raw["anchor"]))
    vals = [float(history[str(d)]) for d in cal.sessions[end - 20 : end + 1]]
    daily = [vals[i + 1] / vals[i] - 1 for i in range(20)]
    mean = sum(daily) / 20
    ret = vals[-1] / vals[0] - 1
    assert actual["x"][:7] == raw["x"]
    assert actual["x"][7:] == pytest.approx([ret, (sum((r - mean) ** 2 for r in daily) / 20) ** 0.5, raw["x"][1] - ret])
    history[raw["cutoff"]] = "99999999"
    assert data.augment(raw, history)[0] == actual
    del history[raw["anchor"]]
    assert data.augment(raw, history) == (None, "MARKET_MISSING_SESSION")


@pytest.mark.parametrize(
    "rows",
    [
        [{"date": "2025-01-02", "close": "1"}],
        [{"date": "2021-01-02", "close": "1"}],
        [{"date": "2021-01-04", "close": "0"}],
        [{"date": "2021-01-04", "close": "NaN"}],
        [{"date": "2021-01-04", "close": "1"}] * 2,
    ],
)
def test_price_ingestion_rejects_invalid_rows(rows):
    with pytest.raises(ValueError):
        data.validate_prices(rows, "930653.CSI")


def test_missing_matched_history_removes_same_keys_from_all_groups(market_job):
    mapping = read_json(ROOT / "app/data/direction_market_mapping_v1.json")
    history = prices()
    market = {"prices": {code: dict(history) for code in ("000300.SH", "930653.CSI", "000905.SH", "980017.SZ")}}
    del market["prices"]["930653.CSI"]["2022-12-30"]
    old = {
        "window": market_job["window"],
        "planned": ["2023-01-03"],
        "complete": {"fit": market_job["fit"], "exam": {"CLEAN": market_job["exam"]}},
    }
    protocol = {"minimum": {"FIT": 252, "EXAM": 40, "CAL": 60}, "minimum_coverage": 0.9}
    bundle, report = build_window(
        old, mapping, market, protocol, {"source": {f: {"unused_gap_mature_count": 100} for f in FUNDS}}
    )
    assert report["status"] == "INSUFFICIENT_DATA"
    assert all(not any(r["fund"] == "001632" for r in job["exam"]) for job in bundle["jobs"].values())
    assert {digest([(r["fund"], r["cutoff"]) for r in job["exam"]]) for job in bundle["jobs"].values()}.__len__() == 1
    assert all("answer" not in item for job in bundle["jobs"].values() for item in job["exam"])
    assert report["excluded"]


def test_mapping_is_exact_and_cannot_pick_new_index():
    mapping = read_json(ROOT / "app/data/direction_market_mapping_v1.json")
    validate_mapping(mapping)
    mapping["funds"]["001632"]["index_code"] = "399997.SZ"
    with pytest.raises(ValueError, match="MAPPING_SCOPE"):
        validate_mapping(mapping)


def test_successful_job_does_not_label_missing_planned_input_as_predicted(market_job):
    bundle = {
        "window": market_job["window"],
        "planned": ["2023-01-03", "2023-01-04"],
        "jobs": {b: deepcopy(market_job) for b in MARKET_BRANCHES},
    }
    outputs = {b: {"status": "PREDICTED", "scores": [0.6] * 3} for b in MARKET_BRANCHES}
    rows = window_predictions(bundle, outputs)
    missing = [r for r in rows if r["cutoff"] == "2023-01-04"]
    assert len(missing) == 24
    assert all(r["score"] is None and r["predicted_up"] is None and r["status"] == "INPUT_UNAVAILABLE" for r in missing)


@pytest.mark.parametrize("codes", [[], ["930653.CSI"], ["000905.SH"], ["930653.CSI", "930653.CSI"]])
def test_exact_index_metadata_query(codes):
    def handler(request):
        payload = json.loads(request.content)
        assert payload["api_name"] == "index_basic" and payload["params"] == {"ts_code": "930653.CSI"}
        return httpx.Response(
            200, json={"code": 0, "data": {"fields": ["ts_code", "name"], "items": [[c, "CS食品饮"] for c in codes]}}
        )

    with TushareMarketReferenceClient(
        token="test",
        api_url="https://example.test",
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_retries=0,
        catalog_max_rows_per_query=8000,
        max_rows_per_query=367,
        transport=httpx.MockTransport(handler),
    ) as api:
        if codes == ["930653.CSI"]:
            assert api.get_index_basic("930653.CSI").index_code == "930653.CSI"
        elif not codes:
            assert api.get_index_basic("930653.CSI") is None
        else:
            with pytest.raises(TushareIntegrationError):
                api.get_index_basic("930653.CSI")


def test_acquisition_requests_only_fixed_development_years_and_resumes_without_network(tmp_path, monkeypatch):
    mapping = read_json(ROOT / "app/data/direction_market_mapping_v1.json")
    calls = []

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def list_index_daily(self, code, *, start_date, end_date):
            calls.append((code, start_date, end_date))
            assert 2021 <= start_date.year <= 2024 and start_date.year == end_date.year
            day = next(d for d in load_calendar().sessions if d.year == start_date.year)
            return (TushareIndexDaily(code, day, Decimal("100")),)

    monkeypatch.setattr(data, "metadata", lambda _: {"rate_limit_per_minute": 200})
    monkeypatch.setattr(data, "ensure_catalog", lambda *args: {})
    monkeypatch.setattr(data, "client", Client)
    monkeypatch.setattr(data, "sleep", lambda _: None)
    shared = tmp_path / "shared.json"
    write_json(shared, {"requests": [{"code": "000300.SH"}], "prices": [{"date": "2021-01-04", "close": "100"}]})
    result, _ = data.acquire(tmp_path, mapping, shared)
    assert len(calls) == 12 and result["database_price_rows_written"] == 0
    assert all(v["missing_dates"] for v in result["coverage"].values())
    # 重入只复用年度响应；输出汇总独占创建，保留原记录而不静默覆盖。
    with pytest.raises(FileExistsError):
        data.acquire(tmp_path, mapping, shared)
    assert len(calls) == 12


def test_missing_prediction_stays_in_plan_and_cannot_pass_gates(market_job):
    protocol = specification_for(COVERAGE_VERSION)
    protocol.update(version=MARKET_VERSION, branches=list(MARKET_BRANCHES))
    predictions, answers = [], []
    for w in protocol["windows"]:
        for d in planned_dates(MARKET_VERSION, w):
            for f in FUNDS:
                key = f"{f}:{d}"
                answers.append({"window": w["name"], "sample_key": key, "answer": None})
                for b in (*MARKET_BRANCHES, *BASELINES):
                    predictions.append(
                        {
                            "window": w["name"],
                            "sample_key": key,
                            "fund": f,
                            "cutoff": str(d),
                            "branch": b,
                            "score": None,
                            "predicted_up": None,
                            "status": "INSUFFICIENT_DATA",
                        }
                    )
    result = evaluate(protocol, predictions, answers)
    assert result["selected_candidate"] is None and result["same_question_count"] == 0
    assert result["candidate_status"]["MATCHED_MARKET"]["status"] == "INSUFFICIENT_DATA"
    bundle = {
        "window": market_job["window"],
        "planned": ["2023-01-03"],
        "jobs": {b: deepcopy(market_job) for b in MARKET_BRANCHES},
    }
    outputs = {b: {"status": "FAILED"} for b in MARKET_BRANCHES}
    rows = window_predictions(bundle, outputs)
    assert len(rows) == 24 and sum(r["status"] == "FAILED" for r in rows) == 9
