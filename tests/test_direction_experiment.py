"""实验页面的数值一致性、数据失败关闭、固定模型及内部权限边界。"""

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest
from app.api.routes import watchlist_prediction as api
from app.repositories.cash_reinvestment_samples import CashDividend, CashNavPoint
from app.repositories.direction_experiment import read_experiment_history
from app.schemas.direction_experiment import DirectionExperiment
from app.schemas.direction_training import DirectionInput
from app.services import direction_experiment as service
from app.services.direction_nav_data import build_input
from app.services.direction_training_artifacts import digest
from app.services.trading_calendar import load_calendar, load_current_calendar
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql
from tests.test_historical_nav_http import HEADERS
from tests.test_historical_nav_http import client as client


def synthetic_models():
    result = []
    for branch, indices in (("DROP_60D_GROUP_L2", [0, 1, 3, 6]), ("REFERENCE", list(range(7)))):
        model = {
            "version": "DIRECTION_LINEAR_FULL_QUARTERS_V1",
            "branch": branch,
            "fund": "POOLED",
            "indices": indices,
            "mean": [0] * len(indices),
            "scale": [1] * len(indices),
            "coefficients": [0.2] * len(indices),
            "intercept": -0.1,
            "train_hash": "a" * 64,
            "train_counts": {f: 703 for f in service.FUNDS},
            "fit_end": "2024-03-31",
            "C": 0.01 if len(indices) == 4 else 1.0,
            "penalty": "L2",
            "solver_iterations": 9,
        }
        model["hash"] = digest(model)
        result.append(model)
    return result


def history(calendar, cutoff):
    index = calendar.at_or_before_index(cutoff)
    dates = calendar.sessions[index - 61 : index]
    # ann_date故意晚于cutoff，证明新版输入沿用V2规则；实际日期仍受交易日保护。
    nav = tuple(CashNavPoint(d, date(2026, 12, 31), Decimal(1) + Decimal(i % 17) / 100) for i, d in enumerate(dates))
    return dates, nav


def test_live_feature_and_score_exactly_match_frozen_research_formula():
    calendar, cutoff = load_calendar(), date(2024, 11, 15)
    dates, nav = history(calendar, cutoff)
    events = (CashDividend("cash", dates[5], dates[5], dates[10], dates[10], Decimal("0.03"), "实施"),)
    models = synthetic_models()
    original, _, issues = build_input("008888", cutoff, {p.nav_date: p for p in nav}, events)
    assert not issues
    scores, _, issues = service.build_experiment_scores("008888", cutoff, dates, nav, events, calendar, models)
    assert not issues
    for model, result in zip(models, scores, strict=True):
        assert result.score == service.predict_model(model, [original])[0]
    with pytest.raises(ValidationError):
        DirectionInput.model_validate({**original.model_dump(), "cutoff": date(2026, 9, 9)})


def test_2026_uses_current_calendar_and_rejects_missing_or_conflicting_input():
    calendar, cutoff = load_current_calendar(), date(2026, 9, 9)
    dates, nav = history(calendar, cutoff)
    scores, audit, issues = service.build_experiment_scores(
        "008888", cutoff, dates, nav, (), calendar, synthetic_models()
    )
    assert len(scores) == 2 and not issues and audit["dates"][-1] == "2026-09-08"
    bad_event = CashDividend("bad", dates[5], dates[5], dates[10], dates[11], Decimal("0.03"), "实施")
    for points, events, code in (
        (nav[:-1], (), "NAV_GAP_COUNT_EXCEEDED"),
        (nav, (bad_event,), "DIVIDEND_DATE_CONFLICT"),
    ):
        scores, audit, issues = service.build_experiment_scores(
            "008888", cutoff, dates, points, events, calendar, synthetic_models()
        )
        assert not scores and audit is None and code in issues


def test_model_loader_pins_content_and_rejects_silent_replacement(tmp_path, monkeypatch):
    models = synthetic_models()
    file = tmp_path / "models.json"
    payload = {m["branch"]: {"models": {"POOLED": m}} for m in models}
    file.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(service, "MODEL_FILE", file)
    monkeypatch.setattr(service, "MODEL_HASHES", {m["branch"]: m["hash"] for m in models})
    assert service.load_experiment_models() == models
    models[0]["intercept"] = 0.9
    models[0]["hash"] = digest({k: v for k, v in models[0].items() if k != "hash"})
    file.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="NOT_PINNED"):
        service.load_experiment_models()


def test_repository_only_reads_fixed_dates_and_keeps_dividend_visibility():
    dates, nav = history(load_current_calendar(), date(2026, 9, 9))
    statements = []
    rows = iter(([tuple((n.nav_date, n.ann_date, n.unit_nav)) for n in nav], []))

    class Session:
        def execute(self, statement):
            statements.append(statement.compile(dialect=postgresql.dialect()))
            return SimpleNamespace(all=lambda: next(rows))

    actual, _ = read_experiment_history(
        Session(), fund_code="008888", source_id=uuid4(), dates=dates, cutoff=date(2026, 9, 9)
    )
    assert actual == nav and len(statements) == 2
    assert all(str(s).startswith("SELECT") and "LIMIT" in str(s) for s in statements)
    assert dates[-1] == date(2026, 9, 8) and list(dates) in statements[0].params.values()
    assert "implementation_ann_date <=" in str(statements[1])
    assert "accumulated_nav" not in str(statements[0])


def test_http_requires_token_and_rejects_browser_and_bad_fund(client, monkeypatch):
    response = DirectionExperiment(
        fund_code="008888",
        status="UNAVAILABLE",
        read_at=datetime.now(UTC),
        reason_codes=("MISSING",),
        message="模型未就绪",
    )
    calls = []
    monkeypatch.setattr(api, "get_direction_experiment", lambda fund: calls.append(fund) or response)
    path = "/internal/v1/predictions/008888/experiment"
    assert client.get(path).status_code == 403
    assert client.get(path, headers={**HEADERS, "Origin": "http://localhost:5173"}).status_code == 403
    assert client.get(path.replace("008888", "bad"), headers=HEADERS).status_code == 422
    assert not calls
    result = client.get(path, headers=HEADERS)
    assert result.status_code == 200 and result.headers["cache-control"] == "no-store"
    assert result.json()["models"] == [] and result.json()["model_released"] is False
    assert "up_probability" not in result.json()


def test_missing_file_outside_scope_and_calendar_never_touch_database(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "MODEL_FILE", tmp_path / "absent.json")
    monkeypatch.setattr(service, "get_nav_preview_engine", lambda: pytest.fail("must not read database"))
    assert service.get_direction_experiment("000001").status == "NOT_APPLICABLE"
    assert service.get_direction_experiment("008888").reason_codes == ("EXPERIMENT_MODEL_UNAVAILABLE",)
    monkeypatch.setattr(service, "load_experiment_models", synthetic_models)
    monkeypatch.setattr(service, "utc_now", lambda: datetime(2027, 2, 10, tzinfo=UTC))
    assert service.get_direction_experiment("008888").reason_codes == ("CALENDAR_COVERAGE_INSUFFICIENT",)


@pytest.mark.parametrize("revision", [None, "ready"])
def test_service_read_only_no_training_and_stable_result(monkeypatch, revision):
    now = datetime(2026, 9, 10, 8, tzinfo=UTC)
    calendar = load_current_calendar()
    dates, nav = history(calendar, date(2026, 9, 9))
    executed = []

    class Session:
        def __init__(self, engine):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def begin(self):
            return self

        def execute(self, sql):
            executed.append(str(sql))

    source = SimpleNamespace(source_id=uuid4())
    revision_id = uuid4() if revision else None
    monkeypatch.setattr(service, "Session", Session)
    monkeypatch.setattr(service, "get_nav_preview_engine", lambda: None)
    monkeypatch.setattr(service, "utc_now", lambda: now)
    monkeypatch.setattr(service, "load_experiment_models", synthetic_models)
    monkeypatch.setattr(service, "read_historical_nav_source", lambda *a, **kw: source)
    monkeypatch.setattr(service, "current_cash_source_revision", lambda *a, **kw: revision_id)

    def read(*args, **kwargs):
        assert revision and kwargs["dates"] == dates and kwargs["cutoff"] == date(2026, 9, 9)
        return nav, ()

    monkeypatch.setattr(service, "read_experiment_history", read)
    first = service.get_direction_experiment("008888")
    second = service.get_direction_experiment("008888")
    assert first == second
    assert all(s == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY" for s in executed)
    if revision:
        assert first.status == "EXPERIMENTAL" and first.target_end_date == date(2026, 10, 15)
        assert first.latest_nav_date == date(2026, 9, 8)
        for update in ({"model_released": True}, {"status": "UNAVAILABLE"}, {"models": ()}):
            with pytest.raises(ValidationError):
                DirectionExperiment.model_validate({**first.model_dump(), **update})
    else:
        assert first.status == "DATA_INSUFFICIENT" and not first.models
