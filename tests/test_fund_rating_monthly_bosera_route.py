from contextlib import nullcontext
from datetime import date

import pytest
from app.services import fund_rating_bosera_qdii_performance as bosera
from app.services import fund_rating_monthly as monthly
from app.services import fund_rating_reviewed_snapshot as snapshot


def bundle():
    return {
        "fund_code": "018853",
        "identity": {"legal_name": "own"},
        "share_class": "C",
        "original_report_archive": {"sha256": "report"},
        "bosera_performance_calculation": {"inputs": {}},
    }


def review():
    return {
        "fund_code": "018853",
        "as_of_date": "2026-10-30",
        "legal_name": "own",
        "share_class": "C",
        "primary_report_hash": "report",
        "bosera_inputs": {"fund_code": "018853", "as_of_date": "2026-10-30"},
        "explanations": {"return": "actual review", "stability": "actual review"},
    }


def test_bosera_missing_originals_is_pending_not_unsupported(tmp_path):
    result = monthly.generate_monthly_inputs("018853", bundle(), date(2026, 10, 31), root=tmp_path, output=tmp_path)
    assert result["status"] == "MATERIALS_PENDING"
    assert result["missing"][0]["field"] == "bosera_inputs"


@pytest.mark.parametrize(
    "field,value",
    [("fund_code", "014978"), ("as_of_date", "2026-09-30"), ("share_class", "A"), ("primary_report_hash", "other")],
)
def test_bosera_foreign_or_old_review_rejected(tmp_path, field, value):
    r = review()
    r[field] = value
    with pytest.raises(ValueError, match="MONTHLY_PERIOD_REVIEW_FUND_PERIOD_ORIGINAL_MISMATCH"):
        monthly.generate_monthly_inputs(
            "018853", bundle(), date(2026, 10, 31), root=tmp_path, output=tmp_path, review=r
        )


def test_bosera_normal_generation_delegates_actual_original_factory(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(monthly, "Session", lambda *a: nullcontext(None))
    monkeypatch.setattr(monthly, "get_nav_preview_engine", lambda: None)
    monkeypatch.setattr(bosera, "check_binding", lambda b, r: calls.append(("binding", r)))
    monkeypatch.setattr(
        bosera,
        "prepare",
        lambda s, r, **k: (
            calls.append(("prepare", r)) or {"entries": [{"research_archive": {}, "research_format": "single"}]}
        ),
    )
    monkeypatch.setattr(snapshot, "_read", lambda *a: {"actual": True})
    monkeypatch.setattr(snapshot, "_case", lambda obj, *a: obj)
    result = monthly.generate_monthly_inputs(
        "018853", bundle(), date(2026, 10, 31), root=tmp_path, output=tmp_path, review=review()
    )
    assert result["status"] == "GENERATED_FOR_COMPILATION"
    assert [c[0] for c in calls] == ["binding", "prepare"]
    assert calls[1][1]["inputs"] == review()["bosera_inputs"]
    assert result["candidate_path"].endswith("candidate-bundle.json")
