"""市场广度分母、日期、来源失败和两组同题比较的边界验证。"""

import json
from copy import deepcopy
from datetime import date

import httpx
import pytest
from app.integrations.tushare import TushareIntegrationError
from app.integrations.tushare_breadth import FIELDS, TushareBreadthClient
from app.services import direction_breadth_data as data
from app.services import direction_breadth_runner as runner
from app.services.direction_linear_models import execute_job, validate_job
from app.services.direction_linear_protocol import (
    BREADTH_VERSION,
    COVERAGE_VERSION,
    planned_dates,
    specification_for,
    study_windows,
)
from app.services.direction_training_artifacts import read_json

from test_direction_market import market_job  # noqa: F401


@pytest.fixture
def rows():
    return [
        {
            "ts_code": f"{600000 + i:06d}.SH" if i < 1500 else f"{i - 1500:06d}.SZ",
            "trade_date": "20210104",
            "close": 110 if i % 2 else 90,
            "pre_close": 100,
            "pct_chg": 10 if i % 2 else -10,
            "vol": 100,
        }
        for i in range(4000)
    ]


def test_up_fraction_weights_stocks_equally(rows):
    result = data.summarize(rows, "2021-01-04")
    assert result["up_fraction"] == 0.5 and result["counts"]["valid"] == 4000
    rows[1]["vol"] = 10000000
    assert data.summarize(rows, "2021-01-04")["up_fraction"] == 0.5


def test_flat_missing_suspension_and_other_markets(rows):
    rows[0].update(close=100, pct_chg=0)
    rows[1]["vol"] = 0
    rows[2]["pct_chg"] = None
    for code in ("900001.SH", "200001.SZ", "830001.BJ", "689001.SH"):
        rows.append({**rows[3], "ts_code": code})
    result = data.summarize(rows, "2021-01-04")
    assert result["counts"]["valid"] == 3998
    assert result["counts"]["flat"] == result["counts"]["no_trade"] == result["counts"]["invalid_quote"] == 1
    assert result["counts"]["out_of_scope"] == 4
    assert result["up_fraction"] == 1999 / 3998


@pytest.mark.parametrize("mutation", ("duplicate", "wrong_date", "missing_field", "wrong_code", "inconsistent_change"))
def test_source_corruption_rejected(rows, mutation):
    if mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "wrong_date":
        rows[0]["trade_date"] = "20210105"
    elif mutation == "missing_field":
        rows[0].pop("vol")
    elif mutation == "wrong_code":
        rows[0]["ts_code"] = "600000"
    else:
        rows[0]["pct_chg"] = 15
    with pytest.raises(ValueError):
        data.summarize(rows, "2021-01-04")


def test_partial_response_fails_coverage(rows):
    assert data.summarize(rows[:2500], "2021-01-04")["status"] == "INSUFFICIENT_SOURCE_COVERAGE"
    for row in rows[:100]:
        row["close"] = None
    assert data.summarize(rows, "2021-01-04")["up_fraction"] is None


def test_feature_date_boundary_and_no_fill(market_job):  # noqa: F811
    raw = market_job["exam"][0]
    days = ("2022-12-26", "2022-12-27", "2022-12-28", "2022-12-29", "2022-12-30")
    daily = {d: {"status": "READY", "up_fraction": v} for d, v in zip(days, (0.1, 0.2, 0.3, 0.4, 0.5), strict=True)}
    result, reason = data.augment(raw, daily)
    assert reason is None and result["x"][-1] == pytest.approx(0.3)
    daily["2023-01-03"] = {"status": "READY", "up_fraction": 1.0}
    assert data.augment(raw, daily)[0] == result
    daily.pop("2022-12-28")
    assert data.augment(raw, daily) == (None, "BREADTH_MISSING_SESSION")


def test_two_branches_and_original_numeric_parity(market_job):  # noqa: F811
    old = deepcopy(market_job)
    old["version"] = COVERAGE_VERSION
    baseline = execute_job(old)
    market_job["version"] = BREADTH_VERSION
    reference = execute_job(market_job)
    assert reference["scores"] == baseline["scores"]
    for k in ("coefficients", "mean", "scale", "intercept", "train_hash"):
        assert reference["models"]["POOLED"][k] == baseline["models"]["POOLED"][k]
    market_job["branch"] = "MARKET_BREADTH"
    for row in market_job["fit"]:
        row["input"]["x"].append(0.5)
    for row in market_job["exam"]:
        row["x"].append(0.5)
    assert execute_job(market_job)["status"] == "PREDICTED"
    market_job["exam"][0]["y"] = 1
    with pytest.raises(ValueError):
        validate_job(market_job)


def test_same_windows_and_dedicated_entry():
    assert study_windows(BREADTH_VERSION) == study_windows(COVERAGE_VERSION)
    assert sum(len(planned_dates(BREADTH_VERSION, w)) for w in study_windows(BREADTH_VERSION)) == 463
    with pytest.raises(ValueError, match="DEDICATED_FREEZE"):
        specification_for(BREADTH_VERSION)


def test_http_exact_fields_and_historical_scope(rows):
    def handle(request):
        body = json.loads(request.content)
        assert body["api_name"] == "daily" and body["params"] == {"trade_date": "20210104"}
        assert body["fields"] == ",".join(FIELDS)
        return httpx.Response(
            200, json={"code": 0, "data": {"fields": list(FIELDS), "items": [[r[f] for f in FIELDS] for r in rows]}}
        )

    with TushareBreadthClient(
        token="test-only",
        api_url="https://example.invalid",
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_retries=0,
        catalog_max_rows_per_query=8000,
        max_rows_per_query=6000,
        transport=httpx.MockTransport(handle),
    ) as api:
        assert api.list_daily_market(date(2021, 1, 4)) == tuple(rows)
        with pytest.raises(ValueError, match="DATE_SCOPE"):
            api.list_daily_market(date(2025, 1, 2))


def test_retry_preserves_evidence_and_cache_prevents_extra_requests(tmp_path, rows, monkeypatch):
    class FakeClient:
        count = 0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def list_daily_market(self, day):
            self.count += 1
            if self.count == 1:
                raise TushareIntegrationError("daily", "API code=-1; private token example", retryable=True)
            return [{**r, "trade_date": day.strftime("%Y%m%d")} for r in rows]

    api = FakeClient()
    monkeypatch.setattr(data, "client", lambda: api)
    monkeypatch.setattr(data, "metadata", lambda: {"source_id": "s", "rate_limit_per_minute": 200})
    monkeypatch.setattr(data, "days", lambda: ["2021-01-04", "2021-01-05"])
    monkeypatch.setattr(data, "sleep", lambda _: None)
    first, _ = data.acquire(tmp_path, {"source_id": "s"})
    assert first["retries"] == 1 and api.count == 3
    saved = read_json(tmp_path / "breadth-attempt-failed-2021-01-04.json")
    assert "private token" not in json.dumps(saved) and saved["provider_code"] == "-1"
    assert data.rebuild_daily(tmp_path) == first["daily"]
    with pytest.raises(FileExistsError):
        data.acquire(tmp_path, {"source_id": "s"})
    assert api.count == 3


def test_missing_input_excludes_same_population_from_both_arms(market_job, monkeypatch):  # noqa: F811
    old = {
        "window": market_job["window"],
        "planned": [r["cutoff"] for r in market_job["exam"][:1]],
        "complete": {"fit": market_job["fit"], "exam": {"CLEAN": market_job["exam"]}},
    }
    protocol = {"minimum": {"FIT": 252, "CAL": 60, "EXAM": 40}, "minimum_coverage": 0.9}
    source = {"source": {f: {"unused_gap_mature_count": 60} for f in runner.FUNDS}}
    monkeypatch.setattr(
        data,
        "augment",
        lambda raw, _: (
            (None, "BREADTH_MISSING_SESSION") if raw["fund"] == "001632" else ({**raw, "x": [*raw["x"], 0.5]}, None)
        ),
    )
    bundle, coverage = runner.build_window(old, {}, protocol, source)
    assert coverage["status"] == "INSUFFICIENT_DATA" and not coverage["fit_unchanged"]
    for phase in ("fit", "exam"):
        assert len(bundle["jobs"]["REFERENCE"][phase]) == len(bundle["jobs"]["MARKET_BREADTH"][phase])
