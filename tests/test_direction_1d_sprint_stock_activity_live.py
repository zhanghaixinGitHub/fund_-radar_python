"""未来输入的20日基线不可夹带T，实际T缺失不可用历史补造；全程本地测试替身。"""

from copy import deepcopy
from datetime import date, timedelta

import pytest
from app.services import direction_1d_sprint_stock_activity_live as s


@pytest.fixture
def baseline_fixture(monkeypatch, tmp_path):
    root = tmp_path / "model"
    days = [str(date(2026, 8, 20) + timedelta(days=i)) for i in range(29)]
    t, u = "2026-09-16", "2026-09-17"
    dates = days[days.index(t) - 20 : days.index(t)]
    rows, raw = {}, {}
    for d in dates:
        path = tmp_path / (d + ".bin")
        path.write_bytes(d.encode())
        raw[d] = {"path": path.name, "sha256": s.model.core.sha(path)}
        rows[d] = {
            "date": d,
            "scope": s.model.data.features.SCOPE,
            "available": True,
            "total_amount": 100.0,
            "top20_share": 0.1,
        }
    value = {
        "at": "2026-09-16T18:30:00+08:00",
        "t": t,
        "u": u,
        "dates": dates,
        "rows": rows,
        "raw_sources": raw,
        "source_qualification_hash": "qualified",
        "calendar_hash": "calendar",
    }
    plan = {"source_qualification_hash": "qualified"}
    monkeypatch.setattr(s.model, "root", lambda: root)
    monkeypatch.setattr(s.b, "ROOT", tmp_path)
    monkeypatch.setattr(s.b, "calendar", lambda: (days, "calendar"))
    s.b.save(root / "future-baseline.json", value)
    s.b.save(root / "runtime-intent.json", {"baseline_hash": s.b.digest(value)})
    s.b.save(root / "plan.json", plan)
    s.b.save(tmp_path / "round-109/plan.json", {"parent": "source"})
    return root, value, t, u, s.b.digest(plan)


def test_frozen_exact_twenty_day_baseline_verifies(baseline_fixture):
    _, value, t, u, _ = baseline_fixture
    assert s.load_baseline(t, u) == value


@pytest.mark.parametrize("change", ["extra_t", "gap", "target", "source", "raw", "value"])
def test_rehashed_baseline_cannot_change_dates_source_or_raw(baseline_fixture, change):
    root, value, t, u, _ = baseline_fixture
    if change == "extra_t":
        value["rows"][t] = deepcopy(next(iter(value["rows"].values())))
    elif change == "gap":
        value["dates"][0] = "2026-01-01"
    elif change == "target":
        value["u"] = "2026-09-18"
    elif change == "source":
        value["source_qualification_hash"] = "other"
    elif change == "value":
        value["rows"][value["dates"][0]]["total_amount"] = -1
    else:
        (s.b.ROOT / value["raw_sources"][value["dates"][0]]["path"]).write_bytes(b"changed")
    s.b.save(root / "future-baseline.json", value, replace=True)
    s.b.save(root / "runtime-intent.json", {"baseline_hash": s.b.digest(value)}, replace=True)
    with pytest.raises(ValueError):
        s.load_baseline(t, u)


@pytest.fixture
def actual_source(baseline_fixture, monkeypatch, tmp_path):
    _, _, t, u, _ = baseline_fixture
    live_root = tmp_path / "actual"
    (live_root / u).mkdir(parents=True)
    (live_root / u / "raw.bin").write_bytes(b"actual-test-source")
    source = {
        "at": "2026-09-17T08:15:00+08:00",
        "available": True,
        "received_at": "2026-09-17T08:14:59+08:00",
        "receipt_hash": "receipt",
        "points": {t: {"included_set_sha256": "same-set"}},
    }
    row = {
        "date": t,
        "scope": s.model.data.features.SCOPE,
        "available": True,
        "total_amount": 200.0,
        "top20_share": 0.15,
        "included_set_sha256": "same-set",
    }
    monkeypatch.setattr(s.parent, "root", lambda: live_root)
    monkeypatch.setattr(s.parent, "load_live", lambda *args: deepcopy(source))
    monkeypatch.setattr(s.model.data.features, "parse", lambda *args: deepcopy(row))
    return baseline_fixture, source, row


def test_current_day_uses_actual_source_and_does_not_mutate_history(actual_source):
    fixture, source, _ = actual_source
    _, baseline, t, u, ph = fixture
    x = s.load_live(t, u, ph)
    assert x["available"] and x["feature"]["concentration_change"] == pytest.approx(0.05)
    assert x["feature"]["log_activity"] == pytest.approx(0.6931471805599453)
    assert x["parent_source_hash"] == s.b.digest(source) and x["baseline_hash"] == s.b.digest(baseline)
    assert x["new_source_requests"] == x["reserved_request_slots"] == 0
    assert t not in s.load_baseline(t, u)["rows"]


def test_missing_actual_t_does_not_fall_back_to_historical_t(actual_source):
    fixture, source, _ = actual_source
    _, _, t, u, ph = fixture
    source.update(available=False, points={}, received_at=None)
    x = s.load_live(t, u, ph)
    assert not x["available"] and t not in x["points"] and len(x["points"]) == 20


def test_baseline_frozen_after_actual_source_rejected(actual_source):
    fixture, _, _ = actual_source
    root, value, t, u, ph = fixture
    value["at"] = "2026-09-17T08:16:00+08:00"
    s.b.save(root / "future-baseline.json", value, replace=True)
    s.b.save(root / "runtime-intent.json", {"baseline_hash": s.b.digest(value)}, replace=True)
    with pytest.raises(ValueError, match="BASELINE_AFTER_ACTUAL_SOURCE"):
        s.load_live(t, u, ph)


def test_actual_universe_cannot_change(actual_source):
    fixture, _, row = actual_source
    _, _, t, u, ph = fixture
    row["included_set_sha256"] = "wrong"
    with pytest.raises(ValueError, match="ACTUAL_SOURCE_UNIVERSE_CHANGED"):
        s.load_live(t, u, ph)


def test_changed_model_plan_rejected_before_input(actual_source):
    fixture, _, _ = actual_source
    _, _, t, u, _ = fixture
    with pytest.raises(ValueError, match="MODEL_PLAN_CHANGED"):
        s.load_live(t, u, "different")
