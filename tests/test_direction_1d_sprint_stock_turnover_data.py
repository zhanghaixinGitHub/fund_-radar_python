"""核验成交额信息的日期、缺失、完整历史与实际父响应绑定，不访问供应商。"""

from copy import deepcopy

import pytest
from app.services import direction_1d_sprint_stock_turnover_data as s


def point():
    return {
        "date": "2026-09-16",
        "scope": "PROVIDER_RETURNED_SH_SZ_DAILY_QUOTES",
        "available": True,
        "total_amount": 100,
        "turnover_breadth": 0.6,
        "turnover_return_pct": 0.5,
    }


def test_exact_t_only_no_forward_fill():
    p = point()
    x = s.extend({}, "2026-09-16", "2026-09-17", {"2026-09-15": p})
    assert not x["stock_turnover"]["available"]
    assert s.extend({"original": 1}, "2026-09-16", "2026-09-17", {"2026-09-16": p}) == {
        "original": 1,
        "stock_turnover": p,
    }


@pytest.mark.parametrize(
    "field,bad",
    [
        ("date", "2026-09-15"),
        ("turnover_breadth", 1.1),
        ("turnover_return_pct", 21),
        ("total_amount", 0),
        ("turnover_breadth", True),
        ("total_amount", float("nan")),
    ],
)
def test_invalid_feature_rejected(field, bad):
    p = point() | {field: bad}
    with pytest.raises(ValueError):
        s.extend({}, "2026-09-16", "2026-09-17", {"2026-09-16": p})


def test_zero_amount_is_missing_not_a_neutral_available_feature():
    p = point() | {
        "available": False,
        "total_amount": 0,
        "unavailable_reason": "ZERO_TOTAL_RETURNED_AMOUNT",
        "turnover_breadth": None,
        "turnover_return_pct": None,
    }
    assert not s.extend({}, "2026-09-16", "2026-09-17", {"2026-09-16": p})["stock_turnover"]["available"]
    with pytest.raises(ValueError, match="UNAVAILABLE_FEATURE_CHANGED"):
        s.extend({}, "2026-09-16", "2026-09-17", {"2026-09-16": p | {"turnover_breadth": 0}})


def test_complete_history_guard_precedes_any_reconstruction(monkeypatch):
    monkeypatch.setattr(s.base, "read", lambda _: {"status": "RUNNING"})
    monkeypatch.setattr(s.old, "history", lambda: pytest.fail("must not read partial history"))
    with pytest.raises(ValueError, match="SOURCE_NOT_COMPLETE"):
        s.reconstruct()


def test_actual_source_unavailability_remains_unavailable(monkeypatch):
    parent = {"at": "2026-09-17T08:14:01+08:00", "available": False}
    monkeypatch.setattr(s.actual_source, "load_live", lambda *args: deepcopy(parent))
    monkeypatch.setattr(s.features, "parse", lambda *_: pytest.fail("failed source must not produce features"))
    result = s.live("2026-09-16", "2026-09-17", "parent-plan")
    assert not result["available"] and result["snapshot"]["points"] == {}
    assert result["at"] == parent["at"] and result["parent_source_hash"] == s.base.digest(parent)
    assert result["new_source_requests"] == 0


def test_actual_raw_derivation_binds_same_stock_distribution(monkeypatch, tmp_path):
    native = {"breadth": 0.4}
    parent = {"at": "2026-09-17T08:14:01+08:00", "available": True, "points": {"2026-09-16": native}}
    r = tmp_path / "2026-09-17"
    r.mkdir()
    (r / "raw.bin").write_bytes(b"already-verified-synthetic-bytes")
    monkeypatch.setattr(s.actual_source, "root", lambda: tmp_path)
    monkeypatch.setattr(s.actual_source, "load_live", lambda *args: deepcopy(parent))
    v = point() | {"source_distribution_hash": s.base.digest(native)}
    monkeypatch.setattr(s.features, "parse", lambda *_: deepcopy(v))
    assert s.live("2026-09-16", "2026-09-17", "p")["available"]
    v["source_distribution_hash"] = "other-raw"
    with pytest.raises(ValueError, match="LIVE_SOURCE_CHANGED"):
        s.live("2026-09-16", "2026-09-17", "p")
