"""第二轮验证未来市场数据隔离、父答案绑定、越界写入拒绝和同题计分。"""

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_market as market


class ConstantModel:
    def predict_proba(self, x):
        return np.array([[0.2, 0.8]] * len(x))


def index_rows(anchor):
    wanted = list(map(str, base.input_days(anchor)))
    return {
        code: {d: {"close": str(100 + i), "amount": str(1000 + i)} for i, d in enumerate(wanted)}
        for code in market.INDICES
    }


def test_market_features_ignore_future_price_and_amount():
    points = index_rows(date(2026, 9, 14))
    before = market.market_vector([0.1] * 18, "2026-09-14", points)
    for rows in points.values():
        rows["2026-09-15"] = {"close": "1000000", "amount": "1000000000"}
    assert before == market.market_vector([0.1] * 18, "2026-09-14", points)
    assert len(before) == 30


def test_missing_market_day_is_not_forward_filled():
    points = index_rows(date(2026, 9, 14))
    del points[market.INDICES[0]]["2026-09-14"]
    with pytest.raises(ValueError, match="MARKET_INPUT_INCOMPLETE"):
        market.market_vector([0.1] * 18, "2026-09-14", points)


def test_fit_does_not_read_future_labels():
    days = base.calendar()[0][100:240]
    rows = [
        {"u": str(d), "mature": str(d + timedelta(days=2)), "family": "f", "y": i % 2, "x": [i / 100] + [0.1] * 29}
        for i, d in enumerate(days)
    ]
    cutoff = str(days[-1])
    first = market.fit(rows, "LR30_252", cutoff)
    for r in rows:
        if r["mature"] >= cutoff:
            r["y"] = 1 - r["y"]
    second = market.fit(rows, "LR30_252", cutoff)
    np.testing.assert_array_equal(first["model"][-1].coef_, second["model"][-1].coef_)


@pytest.mark.parametrize("bad", ["missing", "duplicate", "outside"])
def test_query_rejects_unusable_responses(bad):
    row = SimpleNamespace(trade_date=date(2026, 9, 11), amount=100, close_price=10)
    if bad == "missing":
        row.amount = None
    if bad == "outside":
        row.trade_date = date(2026, 9, 14)
    items = [row, row] if bad == "duplicate" else [row]
    client = SimpleNamespace(list_index_activity=lambda *a, **kw: items)
    with pytest.raises(ValueError):
        market.query(client, "000300.SH", date(2026, 9, 7), date(2026, 9, 11))


@pytest.fixture
def forward(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "ROOT", tmp_path)
    monkeypatch.setattr(base, "now", lambda: datetime(2026, 9, 14, 12, tzinfo=base.ZONE))
    base.initialize()
    base.save(tmp_path / "history.json", {"funds": [{"fund_code": "001000"}]})
    base.save(tmp_path / "scope.json", {"codes": ["001000", "002000"]})
    points = index_rows(date(2026, 9, 14))
    wanted = list(points["000300.SH"])
    parent = {
        "at": "2026-09-14T20:00:00+08:00",
        "base": "2026-09-14",
        "u": "2026-09-15",
        "code": "001000",
        "family": "a",
        "group": "CN_EQUITY",
        "base_nav": "1.06",
        "inputs": [{"date": d, "nav": str(1 + i * 0.001)} for i, d in enumerate(wanted)],
        "answers": {"ORIGINAL7": {"prediction": 0}},
    }
    base.save(tmp_path / "forward/2026-09-15/001000.json", parent)
    monkeypatch.setattr(base, "now", lambda: datetime(2026, 9, 14, 20, 5, tzinfo=base.ZONE))
    base.save(market.root() / "result.json", {"winner": "TREE30_252"})
    monkeypatch.setattr(market, "active", lambda: {})
    monkeypatch.setattr(market.time, "sleep", lambda _: None)
    monkeypatch.setattr(market, "client", lambda: SimpleClient())
    monkeypatch.setattr(market, "query", lambda c, i, start, end: {"rows": points[i]})
    bundle = {"TREE30_252": {"CN_EQUITY": {"model": ConstantModel()}}}
    monkeypatch.setattr(market, "models", lambda: ({"model_sha256": "abc"}, bundle))
    return tmp_path, parent


class SimpleClient:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def test_forecast_is_bound_to_original_and_never_overwritten(forward, monkeypatch):
    root, parent = forward
    result = market.tick()
    path = market.root() / "forward/2026-09-15/001000.json"
    original = path.read_bytes()
    assert base.read(path)["parent_hash"] == base.digest(parent)
    assert result["verified_forecasts"] == 1 and result["pending"] == 1
    market.tick()
    assert path.read_bytes() == original
    base.save(
        root / "outcomes/2026-09-15/001000.json",
        {"y": 1, "actual_direction": "UP", "forecast_hash": base.digest(parent)},
    )
    monkeypatch.setattr(base, "now", lambda: datetime(2026, 9, 15, 19, tzinfo=base.ZONE))
    measured = market.report()["matched_forward_metrics"]
    assert measured["TREE30_252"]["accuracy"] == 1
    assert measured["ORIGINAL7"]["accuracy"] == 0
    assert measured["TREE30_252"]["distinct_dates"] == 1


def test_write_crossing_deadline_is_retained_but_not_counted(forward, monkeypatch):
    saved = base.save

    def crossing(path, value, **kwargs):
        saved(path, value, **kwargs)
        if path.parent.parent == market.root() / "forward":
            monkeypatch.setattr(base, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=base.ZONE))

    monkeypatch.setattr(base, "save", crossing)
    result = market.tick()
    assert result["verified_forecasts"] == 0
    assert result["invalid_or_late"] == 1
    assert result["missing_due_predictions"] == 1


def test_existing_parent_hash_mismatch_is_rejected(forward):
    root, _ = forward
    market.tick()
    parent_path = root / "forward/2026-09-15/001000.json"
    changed = base.read(parent_path) | {"base_nav": "9"}
    base.save(parent_path, changed, replace=True)
    with pytest.raises(ValueError, match="PARENT_HASH_CHANGED"):
        market.report()
