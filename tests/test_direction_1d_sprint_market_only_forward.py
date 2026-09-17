"""无NAV输入的提前答案、独立实际净值核对、修订和限频停止。"""

from datetime import datetime

import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_market_only_forward as f


@pytest.fixture
def ready(tmp_path, monkeypatch):
    calendar = b.calendar()
    monkeypatch.setattr(b, "ROOT", tmp_path)
    monkeypatch.setattr(b, "calendar", lambda: calendar)
    clock = [datetime(2026, 9, 16, 7, 10, tzinfo=b.ZONE)]
    monkeypatch.setattr(b, "now", lambda: clock[0])
    b.save(tmp_path / "protocol.json", {"deadline_at": "2026-09-17T12:11:38+08:00"})
    scope = [{"code": "001000", "family": "F", "group": "CN_EQUITY"}]
    monkeypatch.setattr(f.data, "scope", lambda: scope)
    source = {"at": clock[0].isoformat(), "base": "2026-09-15", "target": "2026-09-16", "market": {"synthetic": True}}
    manifest = {"at": "2026-09-16T02:00:00+08:00", "plan_hash": "PLAN", "model_sha256": "MODEL"}
    b.save(f.model.root() / "result.json", manifest)
    monkeypatch.setattr(f.model, "models", lambda: (manifest, {}))
    monkeypatch.setattr(
        f.model, "answers", lambda market, group, bundle: {"CANDIDATE": {"prediction": 1, "research_score": 0.6}}
    )
    monkeypatch.setattr(f.data, "load", lambda target: source)
    return clock, source, manifest


def forecast(ready):
    _, source, manifest = ready
    f.write_forecasts(source, manifest, {})
    return b.read(f.model.root() / "forward/2026-09-16/001000.json")


def observation():
    return {
        "at": "2026-09-16T19:00:00+08:00",
        "expires_at": "2026-10-01T00:00:00+08:00",
        "funds": [
            {
                "fund_code": "001000",
                "rows": [
                    {
                        "date": "2026-09-15",
                        "nav": "1.00",
                        "ann_date": "2026-09-16",
                        "received_at": "2026-09-16T19:00:01+08:00",
                    },
                    {
                        "date": "2026-09-16",
                        "nav": "1.01",
                        "ann_date": "2026-09-16",
                        "received_at": "2026-09-16T19:00:01+08:00",
                    },
                ],
            }
        ],
    }


def test_no_nav_parent_predicts_and_answers_are_immutable(ready):
    value = forecast(ready)
    path = f.model.root() / "forward/2026-09-16/001000.json"
    old = path.read_bytes()
    forecast(ready)
    assert path.read_bytes() == old and value["base_nav_at_forecast"] is None
    assert not (b.ROOT / "forward").exists()
    assert f.report()["verified_forecasts"] == 1


@pytest.mark.parametrize("fault", ["direction", "model", "input", "scope", "base", "nav_claim", "score"])
def test_rehashed_answer_or_bindings_fail_verification(ready, fault):
    value = forecast(ready)
    if fault == "direction":
        value["answers"]["CANDIDATE"]["prediction"] = 0
    elif fault == "score":
        value["answers"]["CANDIDATE"]["research_score"] = 0.7
    elif fault == "base":
        value["t"] = "2026-09-14"
    elif fault == "nav_claim":
        value["base_nav_at_forecast"] = "1.0"
    else:
        value[f"{fault}_hash"] = "CHANGED"
    b.save(f.model.root() / "forward/2026-09-16/001000.json", value, replace=True)
    path = f.model.root() / "receipts/2026-09-16/001000.json"
    b.save(path, b.read(path) | {"forecast_hash": b.digest(value)}, replace=True)
    result = f.report()
    assert result["verified_forecasts"] == 0 and len(result["invalid_forecasts"]) == 1


def test_readback_at_0830_is_invalid_even_if_creation_earlier(ready):
    forecast(ready)
    path = f.model.root() / "receipts/2026-09-16/001000.json"
    b.save(path, b.read(path) | {"readback_at": "2026-09-16T08:30:00+08:00"}, replace=True)
    assert f.report()["verified_forecasts"] == 0


@pytest.mark.parametrize(
    "fault",
    ["missing_t", "missing_u", "missing_ann", "future_ann", "historical_only", "early_target", "future_receipt"],
)
def test_only_two_real_published_observations_form_label(ready, fault):
    value, snapshot = forecast(ready), observation()
    rows = snapshot["funds"][0]["rows"]
    if fault == "missing_t":
        rows.pop(0)
    elif fault == "missing_u":
        rows.pop()
    elif fault == "missing_ann":
        rows[0]["ann_date"] = None
    elif fault == "future_ann":
        rows[0]["ann_date"] = "2026-09-17"
    elif fault == "historical_only":
        rows[0].pop("received_at")
    else:
        rows[1]["received_at"] = f"2026-09-16T{'17:59:00' if fault == 'early_target' else '23:00:00'}+08:00"
    assert f.observed_pair(value, snapshot, datetime(2026, 9, 16, 20, tzinfo=b.ZONE)) is None


def test_outcome_and_revision_keep_first_prediction_and_label(ready):
    clock, _, _ = ready
    value, snapshot = forecast(ready), observation()
    clock[0] = datetime(2026, 9, 16, 20, tzinfo=b.ZONE)
    b.save(b.ROOT / "observations/20260916T190010.json", snapshot)
    f.observe_outcomes([value], b.now())
    path = f.model.root() / "outcomes/2026-09-16/001000.json"
    initial = path.read_bytes()
    assert f.report()["matched_forward_metrics"]["CANDIDATE"]["accuracy"] == 1
    snapshot["funds"][0]["rows"][1]["nav"] = "0.99"
    b.save(b.ROOT / "observations/20260916T193010.json", snapshot)
    f.observe_outcomes([value], b.now())
    assert path.read_bytes() == initial
    assert f.report()["revised_questions"] == 1
    snapshot["funds"][0]["rows"][1]["received_at"] = "2026-09-16T19:40:00+08:00"
    b.save(b.ROOT / "observations/20260916T194010.json", snapshot)
    f.observe_outcomes([value], b.now())
    assert len(list((f.model.root() / "revisions/2026-09-16/001000").glob("*.json"))) == 1


def test_same_slot_existing_nav_refresh_and_sprint_deadline_make_no_requests(ready, monkeypatch):
    clock, _, _ = ready
    value = forecast(ready)
    clock[0] = datetime(2026, 9, 16, 20, 10, tzinfo=b.ZONE)
    b.save(b.ROOT / "latest-nav.json", {"at": "2026-09-16T20:01:00+08:00"})
    monkeypatch.setattr(b, "source", lambda: pytest.fail("unexpected source request"))
    f.refresh_if_needed([value], b.now())


def test_independent_refresh_is_per_fund_not_per_model_and_reuses_same_slot(ready, monkeypatch):
    from types import SimpleNamespace

    clock, _, _ = ready
    value = forecast(ready)
    clock[0] = datetime(2026, 9, 16, 20, 10, tzinfo=b.ZONE)
    b.save(
        b.ROOT / "history.json",
        {
            "expires_at": "2026-10-01T00:00:00+08:00",
            "funds": [{"fund_code": "001000", "source_fund_code": "001000.OF", "rows": []}],
        },
    )
    monkeypatch.setattr(b, "source", lambda: {"rate_limit_per_minute": 120})
    monkeypatch.setattr(
        b,
        "get_settings",
        lambda: SimpleNamespace(
            tushare_api_url="https://api.tushare.pro", tushare_token=SimpleNamespace(get_secret_value=lambda: "test")
        ),
    )
    monkeypatch.setattr(b.time, "sleep", lambda seconds: None)
    calls = []

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def list_nav_history(self, code, **kwargs):
            calls.append(code)
            return []

    monkeypatch.setattr(b, "TushareFundClient", Client)
    f.refresh_if_needed([value], b.now())
    f.refresh_if_needed([value], b.now())
    assert calls == ["001000.OF"]
    assert len(list((b.ROOT / "observations").glob("*.json"))) == 1


def test_outcome_rehashed_truth_cannot_replace_original_observation(ready):
    clock, _, _ = ready
    value = forecast(ready)
    clock[0] = datetime(2026, 9, 16, 20, tzinfo=b.ZONE)
    b.save(b.ROOT / "observations/20260916T190010.json", observation())
    f.observe_outcomes([value], b.now())
    path = f.model.root() / "outcomes/2026-09-16/001000.json"
    changed = b.read(path)
    changed["y"] ^= 1
    b.save(path, changed, replace=True)
    with pytest.raises(ValueError, match="OUTCOME_CHANGED"):
        f.report()
    clock[0] = datetime(2026, 9, 17, 13, tzinfo=b.ZONE)
    f.refresh_if_needed([value], b.now())
