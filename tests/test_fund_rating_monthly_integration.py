from contextlib import nullcontext
from datetime import date

import pytest

from app.services import fund_rating_monthly as monthly
from app.services import fund_rating_monthly_inputs as inputs
from app.services.fund_rating_reviewed_snapshot import KEYS


def test_partial_compile_keeps_missing_dimension_null(monkeypatch, tmp_path):
    from app.services import fund_rating_evidence_preview as evidence

    documents = []
    monkeypatch.setattr(monthly, "archive_json", lambda root, value: documents.append(value) or {"file": "f", "sha256": "h"})
    dimensions = {key: "75" for key in KEYS.values()}
    dimensions[KEYS["management"]] = None
    monkeypatch.setattr(evidence, "preview", lambda *a, **k: {"status": "PARTIAL", "dimensions": dimensions})
    result = monthly.compile_monthly_bundle({"fund_code": "015596", "family": "INDEX", "as_of_date": "2026-10-30",
                                             "public_explanations": {"management": "旧期经理职责已完成核验。", "return": "旧期尚缺收益原件，暂不评分。"},
                                             "monthly_missing_reasons": {"return": "旧期缺收益原件"},
                                             "evidence_v1_calculation": {"inputs": {}}}, root=tmp_path)
    assert result["entries"][0]["evidence_v1_calculation"]["partial_release_only"] is True
    unknown = documents[-1]["dimensions"]["management"]
    assert unknown["candidate"] is None and unknown["lower"] == "0" and unknown["upper"] == "100"
    assert documents[-1]["dimensions"]["return"] == "75"
    entry = result["entries"][0]
    assert "尚缺" in entry["public_explanations"]["management"]
    assert "已完成" not in entry["public_explanations"]["management"]
    assert entry["monthly_historical_explanations"]["management"] == ["旧期经理职责已完成核验。"]
    assert "已依据本期实际原件重新复算" in entry["public_explanations"]["return"]
    assert "return" not in entry["monthly_missing_reasons"]


@pytest.mark.parametrize("status", ["EVENT_REVIEW", "ERROR", "BLOCKED"])
def test_material_event_or_error_is_not_a_partial_release(monkeypatch, tmp_path, status):
    from app.services import fund_rating_evidence_preview as evidence

    monkeypatch.setattr(evidence, "preview", lambda *a, **k: {"status": status, "dimensions": {"ret": "75"}})
    with pytest.raises(ValueError, match="MONTHLY_EVIDENCE_INCOMPLETE"):
        monthly.compile_monthly_bundle({"family": "INDEX", "evidence_v1_calculation": {} or {"inputs": {}}}, root=tmp_path)


def test_new_nav_is_collected_but_missing_period_review_cannot_approve(monkeypatch, tmp_path):
    from app.services import direction_1d_protocol

    monkeypatch.setattr(direction_1d_protocol, "calendar", lambda: ([date(2026, 10, 30)], "calendar"))
    monkeypatch.setattr(monthly, "Session", lambda *a: nullcontext(None))
    monkeypatch.setattr(monthly, "get_nav_preview_engine", lambda: None)
    called = []
    monkeypatch.setattr(inputs, "provider_snapshot", lambda *a, **k: called.append(a) or {"history": {"dates": ["2026-10-30"]}})
    monkeypatch.setattr(inputs, "_derived", lambda *a, **k: {"sha256": "new-nav"})
    bundle = {"fund_code": "x", "family": "INDEX", "evidence_v1_calculation": {"inputs": {}}}
    result = monthly.generate_monthly_inputs("x", bundle, date(2026, 10, 31), root=tmp_path, output=tmp_path)
    assert called and result["provider_ref"]["sha256"] == "new-nav"
    assert result["status"] == "MATERIALS_PENDING" and result["missing"][0]["field"] == "period_review"


def test_period_review_cannot_be_borrowed_from_another_fund(monkeypatch, tmp_path):
    from app.services import direction_1d_protocol

    monkeypatch.setattr(direction_1d_protocol, "calendar", lambda: ([date(2026, 10, 30)], "calendar"))
    monkeypatch.setattr(monthly, "Session", lambda *a: nullcontext(None))
    monkeypatch.setattr(monthly, "get_nav_preview_engine", lambda: None)
    monkeypatch.setattr(inputs, "provider_snapshot", lambda *a, **k: {"history": {"dates": ["2026-10-30"]}})
    monkeypatch.setattr(inputs, "_derived", lambda *a, **k: {"sha256": "new-nav"})
    bundle = {"fund_code": "x", "family": "INDEX", "evidence_v1_calculation": {"inputs": {}},
              "identity": {"legal_name": "own"}, "share_class": "C", "original_report_archive": {"sha256": "own-report"}}
    with pytest.raises(ValueError, match="PERIOD_REVIEW_FUND_PERIOD_ORIGINAL_MISMATCH"):
        monthly.generate_monthly_inputs("x", bundle, date(2026, 10, 31), root=tmp_path, output=tmp_path,
                                        review={"fund_code": "other"})


def test_bosera_recomputes_only_two_dimensions_and_preserves_partial(monkeypatch, tmp_path):
    from app.services import fund_rating_bosera_qdii_performance as bosera
    from app.services import fund_rating_reviewed_snapshot as snapshot

    documents = []
    monkeypatch.setattr(monthly, "archive_json", lambda root, value: documents.append(value) or {"file": "f", "sha256": "h"})
    monkeypatch.setattr(bosera, "check_binding", lambda *a: None)
    monkeypatch.setattr(bosera, "calculate", lambda *a, **k: {"score": None, "grade": None,
                        "component_scores": {"return": "76.17037", "stability": "74.58333"}})
    observed = []
    monkeypatch.setattr(snapshot, "calculate", lambda b, **k: observed.append(b) or {"status": "PARTIAL", "score": None})
    unknown = {"lower": "0", "upper": "100", "candidate": None, "reason": "still missing"}
    bundle = {"fund_code": "018853", "as_of_date": "2026-10-30", "bosera_performance_calculation": {"inputs": {}},
              "reviewed_document": {"as_of": "2026-09-30", "dimensions": {"return": unknown, "stability": unknown, "risk": unknown}}}
    monthly.compile_monthly_bundle(bundle, root=tmp_path)
    result = observed[0]["reviewed_document"]
    assert result["as_of"] == "2026-10-30"
    assert result["dimensions"]["return"]["candidate"] == "76.17037"
    assert result["dimensions"]["risk"]["candidate"] is None
    assert observed[0]["bosera_performance_calculation"]["input_archive"]


def test_same_month_partial_retries_materials_and_consumes_changed_candidate_idempotently(monkeypatch, tmp_path):
    import json
    from datetime import datetime

    bundle = {"fund_code": "015596", "as_of_date": "2026-10-30", "reviewed_document": {
        "dimensions": {"management": {"candidate": None}}}}
    previous = {"015596": {"as_of_date": date(2026, 10, 30), "batch_id": "old", "input_snapshot": {
        "members": [{"fund_code": "015596", "bundle": bundle}]}}}
    monkeypatch.setattr(monthly, "now", lambda: datetime(2026, 11, 17, 12))
    monkeypatch.setattr(monthly, "archive_root", lambda: tmp_path)
    monkeypatch.setattr(monthly, "last_completed_month", lambda _: date(2026, 10, 31))
    monkeypatch.setattr(monthly, "Session", lambda *a: nullcontext(None))
    monkeypatch.setattr(monthly, "get_nav_preview_engine", lambda: None)
    monkeypatch.setattr(monthly, "current_inputs", lambda *a: previous)
    monkeypatch.setattr(monthly, "_period_covered", lambda *a: True)
    preparations, registrations = [], []
    monkeypatch.setattr(monthly, "generate_monthly_inputs", lambda *a, **k: preparations.append(1) or {
        "status": "MATERIALS_PENDING", "missing": [{"field": "new_original"}]})
    monkeypatch.setattr(monthly, "compile_monthly_bundle", lambda b, **k: {"new_source_version": b["new_source_version"]})
    monkeypatch.setattr(monthly, "register_update", lambda *a: registrations.append(1) or {"registered": True})
    path = tmp_path / "monthly/2026-10/015596/reviewed-bundle.json"
    path.parent.mkdir(parents=True)

    def write(version):
        path.write_text(json.dumps({"fund_code": "015596", "as_of_date": "2026-10-30", "new_source_version": version}))

    write(1)
    assert monthly.prepare_monthly(["015596"], tmp_path)["items"][0]["status"] == "REGISTERED_FOR_PUBLICATION"
    assert monthly.prepare_monthly(["015596"], tmp_path)["items"][0]["status"] == "UNCHANGED_PARTIAL_INPUT"
    write(2)
    assert monthly.prepare_monthly(["015596"], tmp_path)["items"][0]["status"] == "REGISTERED_FOR_PUBLICATION"
    next_run = tmp_path / "next-daily-run"
    next_candidate = next_run / "monthly/2026-10/015596/reviewed-bundle.json"
    next_candidate.parent.mkdir(parents=True)
    next_candidate.write_text(path.read_text())
    assert monthly.prepare_monthly(["015596"], next_run)["items"][0]["status"] == "UNCHANGED_PARTIAL_INPUT"
    assert len(preparations) == 4 and len(registrations) == 2


@pytest.mark.parametrize("method,field", [("typed_disclosure_calculation", "typed_inputs"), ("r5_calculation", "r5_inputs"),
                                          ("active_r5_partial_calculation", "r5_inputs")])
def test_typed_and_r5_collect_actual_nav_before_structured_missing_route(monkeypatch, tmp_path, method, field):
    from app.services import direction_1d_protocol

    monkeypatch.setattr(direction_1d_protocol, "calendar", lambda: ([date(2026, 10, 30)], "calendar"))
    monkeypatch.setattr(monthly, "Session", lambda *a: nullcontext(None))
    monkeypatch.setattr(monthly, "get_nav_preview_engine", lambda: None)
    fetched = []
    monkeypatch.setattr(inputs, "provider_snapshot", lambda *a, **k: fetched.append(1) or {"history": {"dates": ["2026-10-30"]}})
    monkeypatch.setattr(inputs, "_derived", lambda *a, **k: {"sha256": "actual-nav"})
    bundle = {"fund_code": "005284", method: {"profile": "CONVERTIBLE"}, "identity": {"legal_name": "own"},
              "share_class": "C", "original_report_archive": {"sha256": "report"}}
    review = {"fund_code": "005284", "as_of_date": "2026-10-30", "legal_name": "own", "share_class": "C", "primary_report_hash": "report"}
    result = monthly.generate_monthly_inputs("005284", bundle, date(2026, 10, 31), root=tmp_path, output=tmp_path, review=review)
    assert fetched and result["provider_ref"]["sha256"] == "actual-nav"
    assert result["missing"][0]["field"] == field


def test_identical_normalized_nav_and_review_reuse_generation(monkeypatch, tmp_path):
    from app.services import direction_1d_protocol

    monkeypatch.setattr(direction_1d_protocol, "calendar", lambda: ([date(2026, 10, 30)], "calendar"))
    monkeypatch.setattr(monthly, "Session", lambda *a: nullcontext(None))
    monkeypatch.setattr(monthly, "get_nav_preview_engine", lambda: None)
    version, calls = ["original"], []
    monkeypatch.setattr(inputs, "provider_snapshot", lambda *a, **k: {"normalized_records_observed_at": str(len(calls)),
        "history": {"dates": ["2026-10-30"], "rows": [{"content_hash": version[0]}]}})
    monkeypatch.setattr(inputs, "_derived", lambda *a, **k: {"sha256": "nav"})

    def build(*a, **k):
        calls.append(1)
        path = tmp_path / "candidate-bundle.json"
        path.write_text(version[0])
        return {"candidate_path": str(path)}

    monkeypatch.setattr(inputs, "build_monthly_inputs", build)
    bundle = {"fund_code": "x", "family": "INDEX", "evidence_v1_calculation": {"inputs": {}},
              "identity": {"legal_name": "own"}, "share_class": "C", "original_report_archive": {"sha256": "report"}}
    review = {"fund_code": "x", "as_of_date": "2026-10-30", "legal_name": "own", "share_class": "C",
              "primary_report_hash": "report", "management_review": {"source": "actually reviewed"}}
    args = dict(root=tmp_path, output=tmp_path, review=review)
    monthly.generate_monthly_inputs("x", bundle, date(2026, 10, 31), **args)
    review["reviewed_at"] = "later read of identical original"
    assert monthly.generate_monthly_inputs("x", bundle, date(2026, 10, 31), **args)["reused_identical_inputs"] is True
    assert len(calls) == 1
    version[0] = "corrected actual record"
    monthly.generate_monthly_inputs("x", bundle, date(2026, 10, 31), **args)
    assert len(calls) == 2


def test_incomplete_issuer_source_is_retried_with_identical_nav_and_review(monkeypatch, tmp_path):
    from app.services import direction_1d_protocol

    monkeypatch.setattr(direction_1d_protocol, "calendar", lambda: ([date(2026, 10, 30)], "calendar"))
    monkeypatch.setattr(monthly, "Session", lambda *a: nullcontext(None))
    monkeypatch.setattr(monthly, "get_nav_preview_engine", lambda: None)
    monkeypatch.setattr(inputs, "provider_snapshot", lambda *a, **k: {"history": {"dates": ["2026-10-30"]}})
    monkeypatch.setattr(inputs, "_derived", lambda *a, **k: {"sha256": "same-nav"})
    calls = []

    def build(*a, **k):
        calls.append(1)
        path = tmp_path / "candidate-bundle.json"
        path.write_text("same actual partial data")
        return {"candidate_path": str(path), "preview": {"status": "PARTIAL"},
                "attempt_errors": [{"step": "ISSUER_BENCHMARK", "reason": "111"}]}

    monkeypatch.setattr(inputs, "build_monthly_inputs", build)
    bundle = {"fund_code": "x", "family": "INDEX", "evidence_v1_calculation": {"inputs": {}},
              "identity": {"legal_name": "own"}, "share_class": "C", "original_report_archive": {"sha256": "report"}}
    review = {"fund_code": "x", "as_of_date": "2026-10-30", "legal_name": "own", "share_class": "C",
              "primary_report_hash": "report", "management_review": {"source": "actually reviewed"}}
    for _ in range(2):
        monthly.generate_monthly_inputs("x", bundle, date(2026, 10, 31), root=tmp_path, output=tmp_path, review=review)
    assert len(calls) == 2


def test_registration_identity_ignores_only_time_and_redundant_derived_envelopes():
    from copy import deepcopy
    from app.services.direction_1d_protocol import digest

    first = {"entries": [{"verified_at": "first-read", "evidence_v1_archive": {"sha256": "wrapper1"},
             "evidence_v1_calculation": {"inputs": {"verified_at": "first-read", "source_hash": "actual-original"}}}]}
    second = deepcopy(first)
    second["entries"][0]["verified_at"] = "second-read"
    second["entries"][0]["evidence_v1_archive"]["sha256"] = "wrapper2"
    second["entries"][0]["evidence_v1_calculation"]["inputs"]["verified_at"] = "second-read"
    assert digest(monthly._registration_identity(first)) == digest(monthly._registration_identity(second))
    second["entries"][0]["evidence_v1_calculation"]["inputs"]["source_hash"] = "corrected-actual-original"
    assert digest(monthly._registration_identity(first)) != digest(monthly._registration_identity(second))


def test_active_r5_partial_compile_delegates_to_actual_original_factory(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    calls = []
    monkeypatch.setitem(sys.modules, "app.services.fund_rating_active_r5_partial", SimpleNamespace(
        prepare=lambda session, raw, **kw: calls.append(raw) or {"entries": [{"source_factory": True}]}))
    monkeypatch.setattr(monthly, "Session", lambda *a: nullcontext(None))
    monkeypatch.setattr(monthly, "get_nav_preview_engine", lambda: None)
    originals = {"fund_code": "002112", "as_of_date": "2026-10-30", "version": "R5_ACTIVE_MONTHLY_PARTIAL_ORIGINALS_V1"}
    result = monthly.compile_monthly_bundle({"fund_code": "002112", "as_of_date": "2026-10-30",
                   "active_r5_partial_calculation": {"inputs": originals}}, root=tmp_path)
    assert calls == [originals] and result["entries"][0]["source_factory"] is True


def test_old_partial_cannot_replace_same_period_new_complete_registration(monkeypatch, tmp_path):
    import json
    from app.services import fund_rating_reviewed_snapshot as snapshot

    monkeypatch.setattr(monthly, "Session", lambda *a: nullcontext(None))
    monkeypatch.setattr(monthly, "get_nav_preview_engine", lambda: None)
    monkeypatch.setattr(monthly, "current_inputs", lambda *a: {})
    pointer = tmp_path / "pipeline/updates/015596.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(json.dumps({"fund_code": "015596", "as_of_date": "2026-10-30", "dossier": {"sha256": "new-full"}}))
    monkeypatch.setattr(snapshot, "_read", lambda *a: {"entries": [{"new_actual_full": True}]})
    monkeypatch.setattr(monthly, "_dossier_bundle", lambda *a: {"reviewed_document": {"dimensions": {"management": "55"}}})
    old = {"as_of_date": "2026-10-30", "evidence_v1_calculation": {"partial_release_only": True}}
    assert monthly._new_complete_prevents_partial(tmp_path, "015596", old) is True


def test_same_period_actual_review_survives_new_daily_run_directory(monkeypatch, tmp_path):
    from datetime import datetime
    from app.services.fund_materials_store import versioned_save

    bundle = {"fund_code": "015596", "reviewed_document": {"dimensions": {"management": None}}}
    previous = {"015596": {"as_of_date": date(2026, 10, 30), "batch_id": "old", "input_snapshot": {
        "members": [{"fund_code": "015596", "bundle": bundle}]}}}
    monkeypatch.setattr(monthly, "now", lambda: datetime(2026, 11, 18, 0, 30))
    monkeypatch.setattr(monthly, "archive_root", lambda: tmp_path)
    monkeypatch.setattr(monthly, "last_completed_month", lambda _: date(2026, 10, 31))
    monkeypatch.setattr(monthly, "Session", lambda *a: nullcontext(None))
    monkeypatch.setattr(monthly, "get_nav_preview_engine", lambda: None)
    monkeypatch.setattr(monthly, "current_inputs", lambda *a: previous)
    monkeypatch.setattr(monthly, "_period_covered", lambda *a: True)
    review = {"fund_code": "015596", "as_of_date": "2026-10-30", "source_hash": "same-actually-reviewed-original"}
    versioned_save(tmp_path / "pipeline/monthly-input-reviews/2026-10/015596.json", review)
    received = []
    monkeypatch.setattr(monthly, "generate_monthly_inputs", lambda *a, **kw: received.append(kw["review"]) or {
        "status": "MATERIALS_PENDING", "missing": [{"field": "still_missing_source"}]})
    monthly.prepare_monthly(["015596"], tmp_path / "daily-prediction-20261118")
    assert received == [review]


@pytest.mark.parametrize("accepted,candidate_fields", [
    (["management", "fee", "execution"], ["management", "fee"]),
    (["management", "fee", "execution", "holdings", "liquidity"], ["management", "fee", "execution"]),
])
def test_same_month_older_partial_cannot_replace_accepted_more(monkeypatch, tmp_path, accepted, candidate_fields):
    previous = {"fund_code": "002112", "as_of_date": "2026-10-30", "active_r5_partial_calculation": {"inputs": {}},
                "reviewed_document": {"dimensions": {key: "75" for key in accepted}}}
    candidate = {"fund_code": "002112", "as_of_date": "2026-10-30", "active_r5_partial_calculation": {"inputs": {}},
                 "reviewed_document": {"dimensions": {key: "75" if key in candidate_fields else None for key in accepted}}}
    monkeypatch.setattr(monthly, "Session", lambda *a: nullcontext(None))
    monkeypatch.setattr(monthly, "get_nav_preview_engine", lambda: None)
    monkeypatch.setattr(monthly, "current_inputs", lambda *a: {"002112": {"as_of_date": date(2026, 10, 30),
        "input_snapshot": {"members": [{"fund_code": "002112", "bundle": previous}]}}})
    assert monthly._new_complete_prevents_partial(tmp_path, "002112", candidate) is True


def test_actual_new_original_and_explicit_review_can_invalidate_an_old_dimension(monkeypatch, tmp_path):
    from app.services import fund_rating_management_inputs as management

    quote = "实际新增更正原文明确原先用于执行维度的金额或约束事实不再适用，需要撤回原有该维判断。"
    previous = {"research_archive": {"sha256": "prior-three-dimensional-research"},
                "reviewed_document": {"dimensions": {"management": "55", "fee": "76", "execution": "75"}}}
    source = {"sha256": "new-correction-original", "file": "new.json", "kind": "json"}
    candidate = {"fund_code": "002112", "as_of_date": "2026-10-30", "extra_archives": [source],
                 "reviewed_document": {"dimensions": {"management": "55", "fee": "76", "execution": None}},
                 "monthly_reassessment_review": {"fund_code": "002112", "as_of_date": "2026-10-30",
                     "reviewed_at": monthly.now().isoformat(), "invalidated_dimensions": ["execution"],
                     "supersedes_research_sha256": "prior-three-dimensional-research",
                     "reason": "已实际阅读新增更正原件，核对原金额和约束与本基金这一维的适用关系；原判断所依赖的事实已被更正，暂不能继续作为本期已验收维度。其他两维仍有独立原件支持，不因撤回执行一维而重新填旧分或抹去原历史。",
                     "evidence": [{"source": source, "quote": quote}]}}
    monkeypatch.setattr(management, "_source_text", lambda *a: management._plain(quote))
    assert monthly._actual_invalidation_review(candidate, previous, tmp_path) is True
    candidate["monthly_reassessment_review"]["evidence"][0]["source"] = {"sha256": "prior-three-dimensional-research"}
    assert monthly._actual_invalidation_review(candidate, previous, tmp_path) is False

@pytest.mark.parametrize('target,expected', [(date(2026, 10, 30), 'MATERIALS_PENDING'), (date(2026, 11, 30), 'MATERIALS_PENDING')])
def test_huaan_pending_originals_precede_old_typed_adapter(monkeypatch, tmp_path, target, expected):
    import json
    from app.services import fund_rating_monthly_huaan as huaan

    pending = {'file': 'original-graph.json', 'sha256': 'verified'}
    bundle = {'fund_code': '014978', 'as_of_date': '2026-10-30',
              'monthly_performance_followup': {'pending_performance_source_archive': pending}}
    monkeypatch.setattr(huaan, 'pending_performance_bundle', lambda value, root: {
        'fund_code': '014978', 'as_of_date': '2026-10-30', 'fresh_peer': '019524'})
    def resume(value, *, root, output):
        candidate = output / 'restored.json'
        candidate.write_text(json.dumps({'fresh_peer': '019524',
            'monthly_performance_followup': {'pending_performance_source_archive': pending}}))
        return {'status': 'MATERIALS_PENDING', 'candidate_path': str(candidate),
                'pending_performance_source_archive': pending, 'actual_refresh_called': True}
    monkeypatch.setattr(huaan, 'resume_monthly_performance', resume)
    result = monthly.generate_monthly_inputs('014978', bundle, target, root=tmp_path, output=tmp_path)
    assert result['status'] == expected
    assert result['pending_performance_source_archive'] == pending
    if target.month == 10:
        restored = json.loads(__import__('pathlib').Path(result['candidate_path']).read_text())
        assert restored['fresh_peer'] == '019524'
        assert restored['monthly_performance_followup']['pending_performance_source_archive'] == pending
    else:
        assert not list(tmp_path.glob('*candidate*'))



def test_huaan_dual_components_compile_before_old_typed(monkeypatch, tmp_path):
    from app.services import fund_rating_monthly_huaan as huaan
    calls = []
    def compile_actual(value, *, root):
        calls.append(value)
        return {'basis': 'strict', 'entries': [value]}
    monkeypatch.setattr(huaan, 'compile_monthly_performance', compile_actual)
    bundle = {'fund_code': '014978', 'huaan_qdii_relative_calculation': {'inputs': {}},
              'huaan_qdii_benchmark_calculation': {'inputs': {}},
              'typed_disclosure_calculation': {'profile': 'QDII_ACTIVE'}}
    result = monthly.compile_monthly_bundle(bundle, root=tmp_path)
    assert result['basis'] == 'strict' and len(calls) == 1
