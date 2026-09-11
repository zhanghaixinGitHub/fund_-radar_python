"""量价公式、日期隔离、原模型兼容、受限来源和三组同题验证。"""

import sys
from copy import deepcopy
from datetime import date
from decimal import Decimal

import httpx
import pytest
from app.integrations.tushare_market_reference import TushareIndexActivity, TushareMarketReferenceClient
from app.schemas.direction_market import MarketDirectionInput
from app.schemas.direction_training import DirectionInput
from app.services import direction_volume_data as data
from app.services.direction_linear_models import execute_job, validate_job
from app.services.direction_linear_protocol import (
    COVERAGE_VERSION,
    VOLUME_BRANCHES,
    VOLUME_VERSION,
    planned_dates,
    specification_for,
    study_windows,
)
from app.services.direction_market_runner import build_window, window_predictions
from app.services.direction_training_process import run_process
from app.services.trading_calendar import load_calendar

from test_direction_market import market_job, prices  # noqa: F401


@pytest.fixture
def volume_job(market_job):  # noqa: F811
    market_job["version"] = VOLUME_VERSION
    return market_job


def test_fixed_windows_and_no_old_cli(volume_job):
    assert study_windows(VOLUME_VERSION) == study_windows(COVERAGE_VERSION)
    assert sum(len(planned_dates(VOLUME_VERSION, w)) for w in study_windows(VOLUME_VERSION)) == 463
    with pytest.raises(ValueError, match="DEDICATED_FREEZE"):
        specification_for(VOLUME_VERSION)
    original = deepcopy(volume_job)
    original["version"] = COVERAGE_VERSION
    a, b = execute_job(original), execute_job(volume_job)
    assert a["scores"] == b["scores"]
    for k in ("mean", "scale", "coefficients", "intercept", "train_hash"):
        assert a["models"]["POOLED"][k] == b["models"]["POOLED"][k]


@pytest.mark.parametrize("branch,size", [("AMOUNT_ACTIVITY", 8), ("AMOUNT_INTERACTION", 9)])
def test_real_worker_accepts_exact_new_dimensions(volume_job, branch, size):
    volume_job["branch"] = branch
    for row in volume_job["fit"]:
        row["input"]["x"] += [1 + row["input"]["x"][0] / 2, row["input"]["x"][1] / 2][: size - 7]
    for row in volume_job["exam"]:
        row["x"] += [1.2, 0.02][: size - 7]
    out = run_process(volume_job, command=[sys.executable, "-m", "scripts.direction_linear_worker"])
    assert out["status"] == "PREDICTED", out
    assert out["models"]["POOLED"]["indices"] == list(range(size))
    with pytest.raises(ValueError):
        DirectionInput.model_validate(volume_job["exam"][0])
    with pytest.raises(ValueError):
        MarketDirectionInput.model_validate(volume_job["exam"][0])


@pytest.mark.parametrize("change", ["answer", "late_label", "anchor", "future", "size", "nan"])
def test_isolation(volume_job, change):
    if change == "answer":
        volume_job["exam"][0]["y"] = 1
    if change == "late_label":
        volume_job["fit"][0]["answer"]["available_at"] = "2023-01-03"
    if change == "anchor":
        volume_job["exam"][0]["anchor"] = "2023-01-03"
    if change == "future":
        volume_job["exam"][0]["cutoff"] = "2025-01-02"
    if change == "size":
        volume_job["branch"] = "AMOUNT_ACTIVITY"
    if change == "nan":
        volume_job["exam"][0]["x"][0] = float("nan")
    with pytest.raises(ValueError):
        validate_job(volume_job)


def test_formula_uses_only_twenty_previous_sessions(volume_job):
    raw = volume_job["exam"][0]
    cal = load_calendar()
    end = cal.at_or_before_index(date.fromisoformat(raw["anchor"]))
    days = [str(d) for d in cal.sessions[end - 19 : end + 1]]
    amounts = dict(zip(days, ["10"] * 15 + ["20"] * 5, strict=True))
    b, c, reason = data.augment(raw, amounts)
    assert reason is None
    assert b["x"] == pytest.approx(raw["x"] + [1.6])
    assert c["x"] == pytest.approx(raw["x"] + [1.6, raw["x"][1] * 0.6])
    amounts[raw["cutoff"]] = "999999"
    assert data.augment(raw, amounts) == (b, c, None)
    del amounts[days[-1]]
    assert data.augment(raw, amounts)[2] == "VOLUME_MISSING_SESSION"


@pytest.mark.parametrize("value", [None, "0", "-1", "NaN", "Infinity"])
def test_invalid_amount_is_not_filled(value):
    amounts, cov = data.validate_rows([{"date": "2021-01-04", "close": "100", "amount": value}], {"2021-01-04": "100"})
    assert not amounts
    assert cov["invalid_dates"] == {"2021-01-04": "MISSING_NONPOSITIVE_OR_NONFINITE_AMOUNT"}


@pytest.mark.parametrize("change", ["duplicate", "future", "revised_price"])
def test_rejects_source_identity_changes(change):
    rows = [{"date": "2021-01-04", "close": "100", "amount": "12"}]
    if change == "duplicate":
        rows *= 2
    if change == "future":
        rows[0]["date"] = "2025-01-02"
    if change == "revised_price":
        rows[0]["close"] = "101"
    with pytest.raises(ValueError):
        data.validate_rows(rows, {"2021-01-04": "100"})


def test_activity_client_retains_amount_and_identity():
    def handler(request):
        import json

        payload = json.loads(request.content)
        assert payload["api_name"] == "index_daily"
        assert payload["fields"] == "ts_code,trade_date,close,amount"
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "fields": ["ts_code", "trade_date", "close", "amount"],
                    "items": [["930653.CSI", "20210104", 100, 123.4]],
                },
            },
        )

    with TushareMarketReferenceClient(
        token="test",
        api_url="https://example.invalid",
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_retries=0,
        catalog_max_rows_per_query=100,
        max_rows_per_query=367,
        transport=httpx.MockTransport(handler),
    ) as api:
        rows = api.list_index_activity("930653.CSI", start_date=date(2021, 1, 4), end_date=date(2021, 1, 4))
    assert rows[0].amount == Decimal("123.4")


def test_data_missing_keeps_common_keys_and_plan(volume_job):
    old = {
        "window": volume_job["window"],
        "planned": ["2023-01-03"],
        "complete": {"fit": volume_job["fit"], "exam": {"CLEAN": volume_job["exam"]}},
    }
    protocol = {
        "version": VOLUME_VERSION,
        "branches": list(VOLUME_BRANCHES),
        "minimum": {"FIT": 252, "EXAM": 40, "CAL": 60},
        "minimum_coverage": 0.9,
    }
    mapping = {"funds": {f: {"index_code": f} for f in ["001632", "006730", "008888"]}}
    oldcov = {"source": {f: {"unused_gap_mature_count": 60} for f in mapping["funds"]}}
    activity = {"amounts": {f: prices() for f in mapping["funds"]}}
    del activity["amounts"]["001632"]["2022-12-30"]
    bundle, cov = build_window(old, mapping, activity, protocol, oldcov, augment_inputs=data.augment_inputs)
    assert {tuple(i["fund"] for i in j["exam"]) for j in bundle["jobs"].values()} == {("006730", "008888")}
    assert cov["planned_per_fund"] == 1
    outputs = {b: {"status": "INSUFFICIENT_DATA"} for b in VOLUME_BRANCHES}
    predictions = window_predictions(bundle, outputs, branches=VOLUME_BRANCHES)
    assert len([p for p in predictions if p["branch"] in VOLUME_BRANCHES]) == 9


def test_annual_calls_are_bounded_and_cached(tmp_path, monkeypatch):
    calls = []

    class API:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def list_index_activity(self, code, *, start_date, end_date):
            calls.append((code, start_date.year))
            return tuple(
                TushareIndexActivity(code, d, Decimal("100"), Decimal("20"))
                for d in load_calendar().sessions
                if start_date <= d <= end_date
            )

    codes = ["a", "b", "c"]
    mapping = {"funds": {c: {"index_code": c} for c in codes}}
    monkeypatch.setattr(data, "client", API)
    monkeypatch.setattr(data, "metadata", lambda _: {"catalog": dict.fromkeys(codes, {}), "rate_limit_per_minute": 200})
    monkeypatch.setattr(data, "sleep", lambda _: None)
    oldprices = {c: dict.fromkeys(prices(), "100") for c in codes}
    result, _ = data.acquire(tmp_path, mapping, oldprices)
    assert len(calls) == 12 and all(v["usable"] == 969 for v in result["coverage"].values())
    # 另一个阶段入口若重入，年度文件可复用；汇总独占创建仍会阻止覆盖已完成结果。
    with pytest.raises(FileExistsError):
        data.acquire(tmp_path, mapping, oldprices)
    assert len(calls) == 12
