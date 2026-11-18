import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from app.services.fund_rating_huaan_qdii_relative import _independent_unit_curve


@pytest.mark.parametrize("failure", [None, "required_gap", "unit_conflict", "cash", "wrong_share", "wrong_url"])
def test_required_points_strict_but_unused_weekend_not_required(failure):
    code = "019524"
    days = ["2024-03-29", "2024-04-01"]

    def stamp(d):
        return int(datetime.fromisoformat(d).replace(tzinfo=timezone(timedelta(hours=8))).timestamp() * 1000)

    net = [{"x": stamp(d), "y": 1, "unitMoney": ""} for d in days]
    cum = [[stamp(d), 1] for d in days]
    if failure == "required_gap":
        net.pop()
    if failure == "unit_conflict":
        cum[0][1] = 2
    if failure == "cash":
        net[0]["unitMoney"] = "分红"
    raw = (
        'var fS_code="'
        + ("000000" if failure == "wrong_share" else code)
        + '";var Data_netWorthTrend='
        + json.dumps(net)
        + ";var Data_ACWorthTrend="
        + json.dumps(cum)
        + ";"
    ).encode()
    source = SimpleNamespace(
        refs={
            "curve": {
                "url": "https://wrong.invalid"
                if failure == "wrong_url"
                else "https://fund.eastmoney.com/pingzhongdata/019524.js",
                "kind": "ORIGINAL",
            }
        },
        raw={"curve": raw},
    )
    rows = [{"nav_date": d, "unit_nav": 1, "accum_nav": 1} for d in [*days, "2024-03-31"]]
    inputs = {"members": {code: {"independent_unit_curve_ref": "curve"}}}
    if failure:
        with pytest.raises(ValueError):
            _independent_unit_curve(source, inputs, code, rows, days)
    else:
        result = _independent_unit_curve(source, inputs, code, rows, days)
        assert result["required_points"] == 2
        assert result["unused_original_dates"] == ["2024-03-31"]
        assert not result["adjusted_nav_imputed"]


@pytest.mark.parametrize("failure", [None, "rewrite_old", "missing_new", "future", "wrong_previous"])
def test_monthly_admission_preserves_old_selection_time(failure):
    from app.services.fund_rating_huaan_qdii_relative import _preselection_times

    previous = {
        "target_exact_share": "014978.OF",
        "preselected_peer_shares": ["015299.OF"],
        "fixed_at": "2026-10-29T10:00:00+08:00",
    }
    addition = {
        "target_exact_share": "014978.OF",
        "previous_preselection_ref": "old",
        "additional_peer_shares": ["019524.OF"],
        "fixed_at": "2026-11-17T15:10:00+08:00",
    }
    if failure == "rewrite_old":
        addition["additional_peer_shares"].append("015299.OF")
    if failure == "missing_new":
        addition["additional_peer_shares"] = []
    if failure == "future":
        addition["fixed_at"] = "2026-11-18T00:00:00+08:00"
    if failure == "wrong_previous":
        addition["previous_preselection_ref"] = "other"
    source = SimpleNamespace(json=lambda ref: addition)
    inputs = {
        "fund_code": "014978",
        "members": {c: {"ts_code": c + ".OF"} for c in ["014978", "015299", "019524"]},
        "research_on": "2026-11-17T16:00:00+08:00",
    }
    cohort = {"preselection_record_ref": "old", "preselection_additions_ref": "addition"}
    if failure:
        with pytest.raises(ValueError):
            _preselection_times(source, inputs, cohort, previous, ["015299", "019524"])
    else:
        times = _preselection_times(source, inputs, cohort, previous, ["015299", "019524"])
        assert times["015299.OF"] == previous["fixed_at"]
        assert times["019524.OF"] == addition["fixed_at"]
