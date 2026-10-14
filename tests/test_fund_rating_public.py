"""公开资料规则的隔离边界测试；这些算例不是002112真实评级验收。"""

import hashlib
from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
from app.services.fund_rating_public_inputs import build_public_metrics, report_facts, validate_public_sources
from app.services.fund_rating_public_policy import BASIS, TERMS, calculate_public, policy
from app.services.fund_rating_public_validation import independent_score, sensitivity

from test_fund_rating_inputs import AS_OF, NOW, bundle


def sample_metrics():
    return dict(
        return_1y=".08",
        return_3y=".06",
        drawdown=".2",
        downside=".15",
        calmar=".3",
        rolling_positive=".7",
        monthly_positive=".6",
        team_months=24,
        team_changes=1,
        strategy_continuity=1,
        governance_state=2,
        issuer_hhi=".05",
        industry_hhi=".2",
        restricted_ratio=0,
        cash_ratio=".06",
        holder_risk_band=0,
        net_assets=1_000_000_000,
        cost_1y=".01",
    )


def public_bundle(monkeypatch):
    value = bundle()
    value["fund_code"] = "002112"
    h = value["history"]
    h.update(
        calendar_hash="isolated-calendar",
        events_complete=True,
        missing_dates=[],
        event_coverage={"dates_checked": len(h["dates"])},
    )
    calendar_days = [date.fromisoformat(d) for d in h["dates"]]
    calendar_days += [
        date(2026, 10, 1) + timedelta(days=i)
        for i in range(61)
        if (date(2026, 10, 1) + timedelta(days=i)).weekday() < 5
    ]
    from app.services import direction_1d_protocol

    monkeypatch.setattr(direction_1d_protocol, "calendar", lambda: (calendar_days, "isolated-calendar"))
    value.update(
        basis=BASIS,
        manager={"team_months": 24, "team_changes": 1},
        documents_checked_at=NOW.isoformat(),
        key_information={
            "fund_code": value["fund_code"],
            "currency": value["currency"],
            "published_on": str(AS_OF),
            "fees": {
                "management": ".007",
                "custody": ".0015",
                "sales": ".001",
                "subscription": 0,
                "redemption": 0,
                "subscription_mode": "EXTERNAL",
                "operating_cost": ".0096",
            },
            "checked_on": str(NOW.date()),
        },
        disclosure={
            "restricted_stock_amount": "0",
            "holder_risk_band": 0,
            "strategy_continuity": 1,
            "governance_state": 1,
        },
        sources=[],
        category_label="隔离公开资料测试",
    )
    value["holdings"]["balance_sheet"] = {"other_assets": {"货币资金": "20"}, "total_assets": "100", "liabilities": "0"}
    return value


def test_all_eight_dimensions_and_independent_calculation():
    m = sample_metrics()
    result = calculate_public(m)
    total, dims = independent_score(m, policy())
    assert set(result["dimensions"]) == set(TERMS)
    assert abs(total - float(result["score"])) < 1e-10
    assert all(abs(dims[k] - float(result["dimensions"][k])) < 1e-10 for k in TERMS)
    check = sensitivity(m, result)
    assert len(check["weight_20_percent"]) == 16


@pytest.mark.parametrize("dimension", list(TERMS))
def test_no_missing_dimension_is_dropped_or_zero_filled(dimension):
    m = sample_metrics()
    del m[TERMS[dimension][0][0]]
    with pytest.raises(KeyError):
        calculate_public(m)


@pytest.mark.parametrize("bad", [None, True, "NaN", "Infinity"])
def test_invalid_facts_cannot_generate_grade(bad):
    m = sample_metrics()
    m["cash_ratio"] = bad
    with pytest.raises(ValueError):
        calculate_public(m)


@pytest.mark.parametrize(
    "metric,worse",
    [
        ("drawdown", ".5"),
        ("industry_hhi", "1"),
        ("team_changes", 5),
        ("governance_state", 0),
        ("restricted_ratio", ".2"),
        ("holder_risk_band", 2),
        ("cost_1y", ".03"),
    ],
)
def test_worsening_risk_or_cost_cannot_improve_grade(metric, worse):
    m = sample_metrics()
    before = calculate_public(m)
    m[metric] = worse
    assert Decimal(calculate_public(m)["score"]) < Decimal(before["score"])


def test_public_input_expiry_calendar_and_actual_zero_drawdown(monkeypatch):
    b = public_bundle(monkeypatch)
    m, _, expiry = build_public_metrics(b, AS_OF, NOW)
    assert expiry > NOW and len(m["rolling"]) == 25
    b["history"]["unit_nav"] = ["1"] * len(b["history"]["unit_nav"])
    m, _, _ = build_public_metrics(b, AS_OF, NOW)
    assert m["drawdown"] == 0 and m["calmar"] == 0 and m["monthly_positive"] == 0
    with pytest.raises(ValueError, match="DOCUMENTS_EXPIRED"):
        build_public_metrics(b, AS_OF, NOW + timedelta(days=32))
    b["history"]["missing_dates"] = ["2026-09-25"]
    with pytest.raises(ValueError, match="EVENTS_INCOMPLETE"):
        build_public_metrics(b, AS_OF, NOW)


def test_public_sources_remain_local_bounded_and_hash_verified(tmp_path, monkeypatch):
    from app.services import fund_rating_public_inputs as inputs

    monkeypatch.setattr(inputs, "get_settings", lambda: SimpleNamespace(app_environment="local"))
    raw = b"isolated source"
    key = hashlib.sha256(raw).hexdigest()
    p = tmp_path / "sources" / f"{key}.pdf"
    p.parent.mkdir()
    p.write_bytes(raw)
    b = {
        "basis": BASIS,
        "fund_code": "002112",
        "source_scope": "PUBLIC_DISCLOSURE_LOCAL_ANALYSIS",
        "report_hash": key,
        "summary_hash": key,
        "classification": {"source_refs": [key]},
        "raw_archive": [
            {
                "sha256": key,
                "file": f"sources/{key}.pdf",
                "url": "https://www.dbfund.com.cn/report.pdf",
                "acquired_at": NOW.isoformat(),
            }
        ],
    }
    validate_public_sources(b, NOW, root=tmp_path)
    altered = deepcopy(b)
    altered["raw_archive"][0]["url"] = "https://example.com/report.pdf"
    with pytest.raises(ValueError, match="SOURCE_HOST"):
        validate_public_sources(altered, NOW, root=tmp_path)
    with pytest.raises(ValueError, match="SOURCE_FUTURE"):
        validate_public_sources(b, NOW - timedelta(days=1), root=tmp_path)
    p.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SOURCE_HASH"):
        validate_public_sources(b, NOW, root=tmp_path)
    monkeypatch.setattr(inputs, "get_settings", lambda: SimpleNamespace(app_environment="production"))
    with pytest.raises(ValueError, match="LOCAL_SCOPE"):
        validate_public_sources(b, NOW, root=tmp_path)


def test_missing_disclosure_cannot_mean_no_restrictions():
    with pytest.raises(ValueError, match="REPORT_PATTERN_UNSUPPORTED"):
        report_facts(["没有明确披露"])


@pytest.mark.parametrize("key,value", [("fund_code", "000001"), ("currency", "USD"), ("published_on", "2099-01-01")])
def test_public_document_identity_and_future_data_are_rejected(monkeypatch, key, value):
    b = public_bundle(monkeypatch)
    b["key_information"][key] = value
    with pytest.raises(ValueError, match="DOCUMENT_SCOPE"):
        build_public_metrics(b, AS_OF, NOW)


def test_public_explanation_keeps_facts_and_omits_implementation(monkeypatch):
    import json

    from app.services.fund_rating_public_view import public_evidence

    b = public_bundle(monkeypatch)
    metrics, facts, _ = build_public_metrics(b, AS_OF, NOW)
    public = public_evidence({"bundle": b, "metrics": metrics, "facts": facts}, calculate_public(metrics), AS_OF)
    text = json.dumps(public, ensure_ascii=False)
    assert len(public["dimensions"]) == 8
    assert not any(word in text for word in ("权重", "模型", "任务编号", "score", "methodology"))
    assert "报告披露受到监管措施，已整改" in text and "精确比例未公开" in text
