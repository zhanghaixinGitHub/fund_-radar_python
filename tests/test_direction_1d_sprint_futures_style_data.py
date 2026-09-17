"""相对涨幅、日期、实时合约唯一对应及历史原样本保持测试，不访问供应商。"""

import json
from datetime import date

import pytest
from app.services import direction_1d_sprint_futures_style_data as s

T, U = "2026-09-16", "2026-09-17"


def quote(code, opened=100, closed=101):
    return {
        "ts_code": code,
        "trade_date": "20260916",
        "open": opened,
        "high": max(opened, closed) + 1,
        "low": min(opened, closed) - 1,
        "close": closed,
        "vol": 100,
        "oi": 200,
    }


def test_relative_changes_are_percent_points_and_bounded():
    points = s.relative(
        T, {"IC": quote("IC.CFX", closed=102), "IH": quote("IH.CFX", closed=99)}, s.intra.features(quote("IF.CFX"))
    )
    assert points["ic_relative_intraday_pct"] == pytest.approx(1)
    assert points["ih_relative_intraday_pct"] == pytest.approx(-2)
    assert (
        s.relative(
            T, {"IC": quote("IC.CFX", closed=200), "IH": quote("IH.CFX", closed=20)}, s.intra.features(quote("IF.CFX"))
        )["ic_relative_intraday_pct"]
        == 40
    )


def test_only_adjacent_t_features_used(monkeypatch):
    monkeypatch.setattr(s.base, "calendar", lambda: ([date(2026, 9, 15), date(2026, 9, 16), date(2026, 9, 17)], "c"))
    row = {
        "date": T,
        "intraday_return_pct": 1,
        "close_location": 0.2,
        "ic_relative_intraday_pct": 0.5,
        "ih_relative_intraday_pct": -1,
    }
    original = {"available": True, "old": 3}
    market = s.extend(original, T, U, {T: row})
    assert s.original_row({"y": 1, "market": market}) == {"y": 1, "market": original}
    assert not s.extend(original, "2026-09-15", T, {T: row})["futures_style"]["available"]
    with pytest.raises(ValueError, match="ADJACENCY_INVALID"):
        s.extend(original, "2026-09-15", U, {T: row})


@pytest.fixture
def actual():
    rows = [quote("IF.CFX"), quote("IC.CFX", closed=102), quote("IH.CFX", closed=99)]
    for r in list(rows)[1:]:
        rows.append(r | {"ts_code": r["ts_code"].replace(".CFX", "2609.CFX")})
    meta = {
        f: {
            f + "2609.CFX": {
                "ts_code": f + "2609.CFX",
                "list_date": "20260101",
                "delist_date": "20260918",
                "quote_unit": "指数点",
            }
        }
        for f in ("IC", "IH")
    }
    return rows, meta, s.intra.features(rows[0])


def encode(rows):
    fields = s.current.parser.FIELDS
    return json.dumps({"code": 0, "data": {"fields": fields, "items": [[r[k] for k in fields] for r in rows]}}).encode()


def test_unique_actual_contract_match(actual):
    rows, meta, reference = actual
    out = s.parse_live(encode(rows), T, meta, reference)
    assert out["inferred_contracts"] == {"IC": "IC2609.CFX", "IH": "IH2609.CFX"}
    assert out["ic_relative_intraday_pct"] == pytest.approx(1)


@pytest.mark.parametrize(
    "change",
    [
        "different_if",
        "different_concrete_close",
        "ambiguous_match",
        "missing_main",
        "old_day",
        "duplicate",
        "expired_contract",
        "zero_volume",
    ],
)
def test_live_source_changes_or_ambiguity_rejected(actual, change):
    rows, meta, reference = actual
    if change == "different_if":
        reference["intraday_return_pct"] += 0.1
    elif change == "different_concrete_close":
        rows[-1]["close"] += 0.01
    elif change == "ambiguous_match":
        rows.append(rows[-1] | {"ts_code": "IH2610.CFX"})
    elif change == "missing_main":
        rows.pop(1)
    elif change == "old_day":
        rows[0]["trade_date"] = "20260915"
    elif change == "duplicate":
        rows.append(dict(rows[0]))
    elif change == "expired_contract":
        meta["IH"]["IH2609.CFX"]["delist_date"] = "20260915"
    else:
        rows[1]["vol"] = 0
    with pytest.raises((ValueError, KeyError)):
        s.parse_live(encode(rows), T, meta, reference)


def test_source_feature_from_other_day_rejected():
    rows = {"IC": quote("IC.CFX"), "IH": quote("IH.CFX")}
    rows["IC"]["trade_date"] = "20260915"
    with pytest.raises(ValueError, match="FEATURE_DATE_CHANGED"):
        s.relative(T, rows, s.intra.features(quote("IF.CFX")))
