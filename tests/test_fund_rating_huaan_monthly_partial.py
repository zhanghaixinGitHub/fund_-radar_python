from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal

import pytest
from app.services import fund_rating_huaan_qdii_disclosure as disclosure
from app.services import fund_rating_huaan_qdii_relative as relative


def binding():
    inputs = {
        "version": disclosure.MONTHLY_VERSION,
        "monthly_disclosure_only": True,
        "fund_code": "014978",
        "share_class": "C",
        "as_of_date": "2026-10-30",
        "holding_date": "2026-06-30",
        "primary_ref": "primary",
        "legal_name": "legal",
        "manager_name": "manager",
    }
    bundle = {
        **{k: inputs[k] for k in ("fund_code", "share_class", "as_of_date", "holding_date")},
        "monthly_disclosure_only": True,
        "original_report_archive": {"sha256": "primary"},
        "identity": {"legal_name": "legal", "manager_name": "manager"},
        "family": "INDEX",
        "huaan_qdii_disclosure_calculation": {"inputs": inputs},
    }
    return bundle, inputs


def test_monthly_partial_is_explicit_and_legacy_remains_strict():
    bundle, inputs = binding()
    disclosure.check_binding(bundle, inputs)
    legacy = deepcopy(bundle)
    legacy.pop("monthly_disclosure_only")
    with pytest.raises(ValueError, match="EXISTING_REPLAY_REQUIRED"):
        disclosure.check_binding(legacy, inputs)
    bundle["huaan_qdii_relative_calculation"] = {"old": True}
    with pytest.raises(ValueError, match="DISCLOSURE_ONLY_BINDING"):
        disclosure.check_binding(bundle, inputs)


def test_partial_cannot_copy_old_management_or_performance(monkeypatch, tmp_path):
    bundle, _ = binding()
    dims = {
        key: {"lower": "0", "upper": "100", "candidate": None}
        for key in ["return", "risk", "execution", "stability", "management", "fee", "holdings", "liquidity"]
    }
    dims["holdings"] = {"lower": "100", "upper": "100", "candidate": "100"}
    dims["liquidity"] = {"lower": "75", "upper": "75", "candidate": "75"}
    bundle["reviewed_document"] = {"dimensions": {key: value["candidate"] for key, value in dims.items()}}
    monkeypatch.setattr(disclosure, "calculate", lambda *a, **k: {"dimensions": {"holdings": "100", "liquidity": "75"}})
    disclosure.replay_snapshot(bundle, dims, root=tmp_path)
    for key in ["management", "return", "fee"]:
        bad = deepcopy(dims)
        bad[key] = {"lower": "55", "upper": "55", "candidate": "55"}
        with pytest.raises(ValueError, match="UNOBSERVED_DIMENSION_BECAME_POINT"):
            disclosure.replay_snapshot(bundle, bad, root=tmp_path)


def test_production_partial_factory_discards_old_success_explanations_and_metrics(monkeypatch, tmp_path):
    from app.services import fund_rating_pipeline

    bundle, inputs = binding()
    inputs["sources"] = {"notice": {}}
    unknown = ["return", "risk", "execution", "stability", "management", "fee"]
    bundle["public_explanations"] = {key: "旧727日和九个窗口已经成功核验" for key in unknown}
    bundle["public_metrics"] = {key: ["上期成功指标"] for key in unknown}
    bundle["dimension_sources"] = {key: ["上期成功来源说明"] for key in unknown}
    saved = []

    def archive(root, value):
        saved.append(deepcopy(value))
        return {"file": "sources/test.json", "sha256": "test", "kind": "json"}

    monkeypatch.setattr(fund_rating_pipeline, "archive_json", archive)
    monkeypatch.setattr(disclosure, "calculate", lambda *a, **k: {"dimensions": {"holdings": "100", "liquidity": "75"}})
    period = {
        "fund_code": "014978",
        "as_of_date": "2026-10-30",
        "holding_date": "2026-06-30",
        "reason": "本期实际审阅原始报告、申赎公告并保留原完整持仓日期。" * 4,
        "evidence": [{"source_ref": "notice", "quote": "真实新期原文"}],
    }
    facts = {
        key: "本期实际缺件原因和失败原件另有记录。"
        for key in ["cohort", "publisher", "peer_adjusted_nav", "management", "fee"]
    }
    candidate = disclosure.prepare_monthly_disclosure_only(
        bundle,
        root=tmp_path,
        period_review=period,
        notice_reviews=[{"source_ref": "notice"}],
        missing_input_facts=facts,
    )
    for key in unknown:
        assert "727" not in candidate["public_explanations"][key]
        assert "已经成功核验" not in candidate["public_explanations"][key]
        assert "本期实际缺件原因" in candidate["public_explanations"][key]
        assert key not in candidate["public_metrics"]
        assert key not in candidate["dimension_sources"]
        assert saved[-1]["dimensions"][key] is None
    assert saved[-1]["dimensions"]["holdings"] == "100"
    with pytest.raises(ValueError, match="ACTUAL_REVIEW_AND_CAUSES_REQUIRED"):
        disclosure.prepare_monthly_disclosure_only(
            bundle,
            root=tmp_path,
            period_review=period,
            notice_reviews=[{"source_ref": "notice"}],
            missing_input_facts={},
        )


def test_independent_parts_do_not_consume_old_scalar_scores(tmp_path):
    from types import SimpleNamespace

    _, inputs = binding()
    inputs["management_inputs"] = {"score": "55"}
    inputs["monthly_fee_inputs"] = {"score": "75", "summary_ref": "summary", "prospectus_ref": "prospectus"}
    result = disclosure._independent_monthly_components(inputs, SimpleNamespace(sources={}), root=tmp_path)
    assert result["dimensions"] == {"management": None, "cost": None}
    assert set(result["facts"]["evidence_errors"]) == {"management", "cost"}


def test_extended_partial_replays_management_and_fee_but_keeps_performance_unknown(monkeypatch, tmp_path):
    bundle, inputs = binding()
    inputs["version"] = disclosure.MONTHLY_EXTENDED_VERSION
    dims = {
        key: {"lower": "0", "upper": "100", "candidate": None}
        for key in ["return", "risk", "execution", "stability", "management", "fee", "holdings", "liquidity"]
    }
    for key, value in {"management": "55", "fee": "75", "holdings": "100", "liquidity": "75"}.items():
        dims[key] = {"lower": value, "upper": value, "candidate": value}
    bundle["reviewed_document"] = {"dimensions": {key: point["candidate"] for key, point in dims.items()}}
    monkeypatch.setattr(
        disclosure,
        "calculate",
        lambda *args, **kwargs: {
            "dimensions": {"management": "55", "cost": "75", "holdings": "100", "liquidity": "75"}
        },
    )
    disclosure.replay_snapshot(bundle, dims, root=tmp_path)
    for key, value in [("management", "75"), ("fee", "55"), ("return", "75")]:
        wrong = deepcopy(dims)
        wrong[key] = {"lower": value, "upper": value, "candidate": value}
        with pytest.raises(ValueError):
            disclosure.replay_snapshot(bundle, wrong, root=tmp_path)


@pytest.mark.parametrize("failure", [None, "cash", "split", "unit_conflict", "publisher_conflict"])
def test_huaan_null_adjustment_needs_all_original_guards(monkeypatch, failure):
    days = []
    cursor = date(2023, 10, 31)
    while cursor <= date(2026, 10, 30):
        if cursor.weekday() < 5:
            days.append(str(cursor))
        cursor += timedelta(days=1)
    months = {}
    for day in days:
        months[day[:7]] = day
    native = [
        ["014978.OF", d.replace("-", ""), d.replace("-", ""), "2", "2", None if d == "2026-09-29" else "2"]
        for d in days
    ]
    normalized = [
        {
            "nav_date": row[2][:4] + "-" + row[2][4:6] + "-" + row[2][6:],
            "unit_nav": row[3],
            "accumulated_nav": row[4],
            "adjusted_nav": row[5],
        }
        for row in native
    ]
    if failure == "unit_conflict":
        normalized[-1]["unit_nav"] = normalized[-1]["accumulated_nav"] = "3"
    chart = [
        {
            "year": int(d[:4]),
            "month": int(d[5:7]) - 1,
            "day": int(d[-2:]),
            "value": "1" if failure == "publisher_conflict" else "0",
        }
        for d in days
    ]
    data = {
        "calendar": {
            "fund_code": "014978",
            "history": {"dates": days, "month_ends": list(months.values()), "rows": normalized},
        },
        "nav": {
            "code": 0,
            "data": {
                "has_more": False,
                "fields": ["ts_code", "ann_date", "nav_date", "unit_nav", "accum_nav", "adj_nav"],
                "items": native,
            },
        },
        "div": {
            "code": 0,
            "data": {
                "has_more": False,
                "count": 1 if failure == "cash" else 0,
                "fields": ["ts_code"],
                "items": [["014978.OF"]] if failure == "cash" else [],
            },
        },
        "publisher": [{"titleText": "收益率走势", "chartsDatas": chart}],
        "protocol": {"direct_formula": "NAV(previous_trading_day)", "no_imputation": True},
    }

    class Source:
        def json(self, ref):
            return data[ref]

    monkeypatch.setattr(
        relative,
        "_request_identity",
        lambda source, inputs, member, api: {
            "row_count": len(data["nav" if api == "fund_nav" else "div"]["data"]["items"])
        },
    )

    def split(*args):
        if failure == "split":
            raise ValueError("nonempty split original")
        return {"actual_empty_original": True}

    monkeypatch.setattr(relative, "_split_identity", split)
    inputs = {
        "fund_code": "014978",
        "window_start": days[0],
        "as_of_date": days[-1],
        "calendar_ref": "calendar",
        "own_publisher_ref": "publisher",
        "research_on": "2026-11-17T12:00:00+08:00",
        "dividend_statistic_protocol_ref": "protocol",
        "members": {
            "014978": {
                "ts_code": "014978.OF",
                "nav_ref": "nav",
                "dividend_ref": "div",
                "dividend_ref_request_ref": "request",
            }
        },
    }
    if failure:
        with pytest.raises(ValueError):
            relative._history(Source(), inputs, [])
    else:
        result = relative._history(Source(), inputs, [])
        assert result["values"]["014978"][-1] == Decimal(2)
        assert result["unit_total_return_reconciliation"]["original_adjusted_nav_null_dates"] == {
            "014978": ["2026-09-29"]
        }
        assert normalized[days.index("2026-09-29")]["adjusted_nav"] is None


def test_peer_null_adjustment_not_silently_reconstructed():
    with pytest.raises(ValueError, match="MISSING_UNRECONCILED"):
        relative._closed_nav_price({"unit_nav": "2", "accum_nav": "2", "adj_nav": None}, allow_unit=False)
