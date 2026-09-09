"""候选规则只读审查：人工报告/分档验证规则，不作为真实模型发布证据。"""

import copy
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal, localcontext

import pytest
from app.api.routes import watchlist_prediction as api
from app.schemas.cash_release_review import CashReleaseCriterion, CashReleasePolicy, CashReleaseReview
from app.services import cash_prediction_check as precheck
from app.services import cash_release_policy as policy_source
from app.services import cash_release_review as review
from app.services.cash_reinvestment_research import restore_research
from app.services.cash_reinvestment_storage import cash_hash
from app.services.cash_release_evidence import validate_reliability, validate_window_evidence
from app.services.historical_nav_calibration import reliability_report
from app.services.historical_nav_evaluation import calculate_baseline_metrics
from app.services.historical_nav_storage import HistoricalNavStorageError
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from tests.test_cash_prediction_check import db, request_for
from tests.test_cash_prediction_check import evaluated as evaluated
from tests.test_cash_reinvestment_research import data as data
from tests.test_historical_nav_http import HEADERS
from tests.test_historical_nav_http import client as client
from tests.test_watchlist_prediction import run

NOW = datetime(2026, 9, 9, 5, tzinfo=UTC)
URL = "/internal/v1/predictions/release-review"


def distribution(*groups):
    """每组指定固定分数、题数和实际上涨数；不使用任何真实基金资料。"""
    labels = tuple(y for _, count, positives in groups for y in (1,) * positives + (0,) * (count - positives))
    scores = tuple(Decimal(score) for score, count, _ in groups for _ in range(count))
    return reliability_report(labels, scores), calculate_baseline_metrics(labels, scores)


def annual_checks(*groups):
    report, metrics = distribution(*groups)
    return review.evaluate_annual_reliability(
        report, metrics, review.load_release_policy(), window_id="VALIDATION_2024", scope="008888"
    )


def checked(row):
    return review.review_cash_research(
        restore_research(row), review.load_release_policy(), fund_code="006730", checked_at=NOW
    )


def wire_db(monkeypatch, row):
    sql = db(monkeypatch, row)
    monkeypatch.setattr(review, "Session", precheck.Session)
    monkeypatch.setattr(review, "get_nav_preview_engine", lambda: None)
    return sql


def rehash(row):
    row.report["report_hash"] = cash_hash({k: v for k, v in row.report.items() if k != "report_hash"})
    return row


def test_server_policy_remains_draft_and_has_fixed_scope():
    policy = review.load_release_policy()
    assert policy.approval_state == "DRAFT" and policy.approval_reference is None
    assert policy.fund_codes == ("001632", "006730", "008888")
    assert (policy.maximum_ece, policy.maximum_bin_gap, policy.minimum_coverage) == (
        Decimal(".10"),
        Decimal(".15"),
        Decimal(".80"),
    )
    assert str(policy.independent_test_start) == "2025-01-01" and str(policy.independent_test_end) == "2025-12-31"


@pytest.mark.parametrize(
    "update",
    [
        {"fund_codes": ["008888"]},
        {"minimum_fit_per_fund": 1},
        {"minimum_calibration_per_fund": 1},
        {"minimum_rolling_exam_per_fund": 1},
        {"minimum_annual_exam_per_fund": 1},
        {"minimum_bin_count": 1},
        {"required_market_states": ["UP"]},
        {"require_strict_brier_gain": False},
        {"independent_test_start": "2026-01-01"},
        {"approval_state": "APPROVED"},
        {"approval_reference": "pretend-approved"},
        {"maximum_ece": "NaN"},
        {"minimum_coverage": 0},
        {"maximum_ece": ".20", "maximum_bin_gap": ".15"},
        {"force": True},
    ],
)
def test_policy_rejects_scope_changes_and_invalid_approval(update):
    payload = {**review.load_release_policy().model_dump(mode="json"), **update}
    with pytest.raises(ValidationError):
        CashReleasePolicy.model_validate(payload)


@pytest.mark.parametrize("raw", [b"not-json", b"{}", b" " * 16385])
def test_policy_failure_has_no_permissive_fallback(monkeypatch, tmp_path, raw):
    path = tmp_path / "policy.json"
    path.write_bytes(raw)
    monkeypatch.setattr(policy_source, "POLICY_PATH", path)
    with pytest.raises(HistoricalNavStorageError) as error:
        review.load_release_policy()
    assert error.value.code == "CASH_RELEASE_POLICY_UNAVAILABLE" and error.value.status_code == 503


def test_missing_policy_rejected_without_database_read(monkeypatch, tmp_path):
    monkeypatch.setattr(policy_source, "POLICY_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(review, "get_nav_preview_engine", lambda: pytest.fail("no database fallback"))
    with pytest.raises(HistoricalNavStorageError):
        review.review_cash_release(request_for(run()))


def test_review_covers_every_window_fund_baseline_without_publication(evaluated):
    result = checked(evaluated)
    assert result.status == "BLOCKED" and result.policy.approval_state == "DRAFT"
    assert result.report_hash == evaluated.report["report_hash"]
    assert set(precheck.BLOCKERS).issubset(result.blocking_codes)
    assert {"POLICY_NOT_FROZEN", "EX_ANTE_COVERAGE_EVIDENCE_MISSING", "MARKET_STATE_EVIDENCE_MISSING"}.issubset(
        result.blocking_codes
    )
    for flag in (
        "publication_allowed",
        "policy_persisted",
        "inference_executed",
        "independent_test_read",
        "database_written",
    ):
        assert getattr(result, flag) is False
    assert sum(result.check_counts.values()) == len(result.checks)
    gains = [c for c in result.checks if c.baseline_id]
    assert len(gains) == 3 * 4 * 4 * 2
    assert {c.scope for c in gains} == {"ALL", "001632", "006730", "008888"}
    assert {c.baseline_id for c in gains} == precheck.BASELINES
    assert all(c.window_id == "VALIDATION_2024" for c in result.checks if c.code.startswith("ANNUAL_"))
    coverage = [c for c in result.checks if c.code == "EXAM_COVERAGE"]
    assert len(coverage) == 9 and all(
        c.status == "MISSING" and c.actual is None and c.required == Decimal(".80") for c in coverage
    )
    serialized = result.model_dump_json()
    assert not any(
        f'"{key}"' in serialized
        for key in ("base_model", "coefficients", "prediction_preview", "up_probability", "offline_label")
    )


def test_missing_windows_do_not_invent_calibration_or_baseline_scores():
    result = checked(run())
    assert all(
        c.status == "MISSING" for c in result.checks if c.code in ("WINDOW_EVALUATED", "ANNUAL_CALIBRATION_EVIDENCE")
    )
    assert not any(c.baseline_id for c in result.checks)
    assert result.check_counts["FAIL"] > 0  # 数量实测不足，与“根本没有考试成绩”分开。


def test_approved_policy_is_not_frozen_and_cannot_publish(evaluated):
    policy = review.load_release_policy().model_copy(
        update={"approval_state": "APPROVED", "approval_reference": "synthetic-test-only"}
    )
    result = review.review_cash_research(restore_research(evaluated), policy, fund_code="006730", checked_at=NOW)
    assert next(c.status for c in result.checks if c.code == "POLICY_APPROVAL") == "PASS"
    assert next(c.status for c in result.checks if c.code == "POLICY_FREEZE") == "MISSING"
    assert result.status == "BLOCKED" and not result.publication_allowed
    assert result.policy_hash != checked(evaluated).policy_hash


def test_deterministic_under_caller_decimal_context_and_no_report_mutation(evaluated):
    original = copy.deepcopy(evaluated.report)
    expected = checked(evaluated)
    with localcontext() as context:
        context.prec = 9
        actual = checked(evaluated)
    assert expected == actual and evaluated.report == original


@pytest.mark.parametrize(
    ("groups", "code", "expected"),
    [
        (((".1", 100, 20), (".7", 100, 80)), "ANNUAL_ECE", "PASS"),
        (((".09999999", 100, 20), (".69999999", 100, 80)), "ANNUAL_ECE", "FAIL"),
        (((".1", 100, 25), (".7", 100, 70)), "ANNUAL_BIN_GAP", "PASS"),
        (((".09999999", 100, 25), (".7", 100, 70)), "ANNUAL_BIN_GAP", "FAIL"),
        (((".3", 30, 9), (".7", 30, 21)), "ANNUAL_BIN_COUNT", "PASS"),
        (((".3", 29, 9), (".7", 30, 21)), "ANNUAL_BIN_COUNT", "FAIL"),
        (((".3", 100, 45), (".5", 100, 40)), "ANNUAL_OBSERVED_RATE_ORDER", "FAIL"),
        (((".3", 100, 30),), "ANNUAL_POPULATED_BINS", "FAIL"),
        (((".3", 100, 30),), "ANNUAL_OBSERVED_RATE_ORDER", "MISSING"),
        ((("0", 30, 0), ("1", 30, 30)), "ANNUAL_ECE", "PASS"),
    ],
)
def test_annual_thresholds_are_inclusive_but_gain_requires_strict_improvement(groups, code, expected):
    checks = [c for c in annual_checks(*groups) if c.code == code]
    assert checks and (
        any(c.status == expected for c in checks) if expected != "PASS" else all(c.status == "PASS" for c in checks)
    )


@pytest.mark.parametrize(
    ("operator", "actual", "expected"),
    [("GT", 0, "FAIL"), ("LT", 0, "FAIL"), ("GT", ".00000001", "PASS"), ("LT", "-.00000001", "PASS")],
)
def test_baseline_ties_do_not_pass(operator, actual, expected):
    assert review._numeric("GAIN", actual, 0, operator, "人工边界", window_id="test", scope="ALL").status == expected


def test_rounded_ece_cannot_hide_slightly_failed_threshold():
    report, metrics = distribution((".0999999999", 100, 20), (".6999999999", 100, 80))
    assert report.ece == Decimal(".10")  # 原报告8位小数看似刚好满足，但原分档重算结果略超限。
    checks = review.evaluate_annual_reliability(
        report, metrics, review.load_release_policy(), window_id="VALIDATION_2024", scope="ALL"
    )
    ece = next(c for c in checks if c.code == "ANNUAL_ECE")
    assert ece.actual == Decimal(".1000000001") and ece.status == "FAIL"


@pytest.mark.parametrize(
    "mutation", ["edges", "count", "nan", "empty_value", "gap", "ece", "sufficiency", "rate", "population"]
)
def test_inconsistent_bins_are_corruption_not_failed_quality(mutation):
    report, metrics = distribution((".1", 100, 20), (".7", 100, 80))
    report = report.model_copy(deep=True)
    if mutation == "edges":
        report.bins[0].lower = Decimal(".01")
    elif mutation == "count":
        report.bins[0].count = -1
    elif mutation == "nan":
        report.bins[0].mean_score = Decimal("NaN")
    elif mutation == "empty_value":
        report.bins[1].mean_score = Decimal(0)
    elif mutation == "gap":
        report.bins[0].absolute_gap = Decimal(0)
    elif mutation == "ece":
        report.ece = Decimal(0)
    elif mutation == "sufficiency":
        report.bins[0].enough_samples = False
    elif mutation == "rate":
        report.bins[0].observed_up_rate = Decimal(".2001")
        report.bins[0].absolute_gap = Decimal(".1001")
    else:
        report.sample_count += 1
    with pytest.raises((ValueError, ArithmeticError)):
        validate_reliability(report, metrics)


@pytest.mark.parametrize(
    "mutation",
    [
        "bin_totals",
        "metric_totals",
        "fund_missing",
        "deficit",
        "exam_count",
        "before_id",
        "delta",
        "unevaluated_scores",
    ],
)
@pytest.mark.parametrize("path", [URL, "/internal/v1/predictions/generation-check"])
def test_corrupt_stored_evidence_rejected_after_rehash(client, monkeypatch, evaluated, mutation, path):
    row = copy.deepcopy(evaluated)
    window = row.report["windows"][-1]
    if mutation == "bin_totals":
        window["reliability_after"]["ece"] = "0"
    elif mutation == "metric_totals":
        for comparison in (window["before"], window["after"], *window["baselines"]):
            comparison["validation"]["brier_score"] = "0.12345678"
    elif mutation == "fund_missing":
        window["reliability_per_fund"].pop()
    elif mutation == "deficit":
        window["funds"][0]["missing"]["FIT"] = 1
    elif mutation == "exam_count":
        window["funds"][0]["counts"]["EXAM"] += 1
    elif mutation == "before_id":
        window["before"]["baseline_id"] = "A_DIFFERENT_MODEL"
    elif mutation == "delta":
        window["ece_delta"] = "123"
    else:
        window["status"] = "INSUFFICIENT_DATA"
    rehash(row)
    wire_db(monkeypatch, row)
    response = client.post(path, json=request_for(row).model_dump(mode="json", by_alias=True), headers=HEADERS)
    assert response.status_code == 503 and response.json()["detail"]["code"] == "CASH_RESEARCH_CORRUPTED"


def test_http_readonly_identity_no_training_and_no_numeric_input_reads(client, monkeypatch, evaluated):
    from app.services import cash_prediction_inference, historical_nav_calibration, historical_nav_training

    def forbidden(*args, **kwargs):
        pytest.fail("review must not train or infer")

    monkeypatch.setattr(historical_nav_training, "fit_logistic_artifact", forbidden)
    monkeypatch.setattr(historical_nav_training, "predict_artifact_logits", forbidden)
    monkeypatch.setattr(historical_nav_calibration, "predict_calibrated_scores", forbidden)
    monkeypatch.setattr(cash_prediction_inference, "calculate_cash_inference", forbidden)
    sql = wire_db(monkeypatch, evaluated)
    response = client.post(URL, json=request_for(evaluated).model_dump(mode="json", by_alias=True), headers=HEADERS)
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    result = CashReleaseReview.model_validate(response.json())
    assert result.status == "BLOCKED" and result.fund_code == "006730" and not result.database_written
    assert len(sql) == 5 and "READ ONLY" in sql[0] and "REPEATABLE READ" in sql[0]
    assert "WHERE cash_research_run.run_id =" in sql[-1]
    assert all(
        not any(
            t in s for t in ("unit_nav", "cash_sample", "forecast_result", "analysis_model_release", "INSERT", "UPDATE")
        )
        for s in sql
    )


@pytest.mark.parametrize(
    "extra",
    [{"force": True}, {"includeTest": True}, {"policy": {}}, {"minimumCoverage": 0}, {"coverage": []}, {"model": {}}],
)
def test_http_does_not_accept_policy_evidence_or_bypass(client, monkeypatch, extra):
    monkeypatch.setattr(api, "review_cash_release", lambda _: pytest.fail("must not read"))
    payload = {**request_for(run()).model_dump(mode="json", by_alias=True), **extra}
    assert client.post(URL, json=payload, headers=HEADERS).status_code == 422


@pytest.mark.parametrize("headers", [{}, {"X-Service-Token": "bad"}, {**HEADERS, "Origin": "http://localhost"}])
def test_http_internal_auth_precedes_read(client, monkeypatch, headers):
    monkeypatch.setattr(api, "review_cash_release", lambda _: pytest.fail("must not read"))
    assert (
        client.post(URL, json=request_for(run()).model_dump(mode="json", by_alias=True), headers=headers).status_code
        == 403
    )


def test_http_hash_conflict_and_database_failure_redaction(client, monkeypatch, evaluated):
    wire_db(monkeypatch, evaluated)
    payload = request_for(evaluated).model_dump(mode="json", by_alias=True)
    assert client.post(URL, json={**payload, "expectedReportHash": "f" * 64}, headers=HEADERS).status_code == 409

    def fail(_):
        raise SQLAlchemyError("synthetic-private-db-value")

    monkeypatch.setattr(api, "review_cash_release", fail)
    result = client.post(URL, json=payload, headers=HEADERS)
    assert result.status_code == 503 and "synthetic-private" not in result.text


def test_response_rejects_nan_and_false_summary(evaluated):
    with pytest.raises(ValidationError):
        CashReleaseCriterion(
            code="X", window_id="GLOBAL", scope="ALL", status="FAIL", operator="LE", actual="NaN", message="bad"
        )
    payload = checked(evaluated).model_dump(mode="json")
    for update in (
        {"publication_allowed": True},
        {"database_written": True},
        {"blocking_codes": []},
        {"check_counts": {"PASS": 999}},
    ):
        with pytest.raises(ValidationError):
            CashReleaseReview.model_validate({**payload, **update})


@pytest.mark.parametrize(
    "update",
    [
        {"actual": "0.5", "required": "0.8", "status": "PASS"},
        {"actual": None, "required": "0.8", "status": "PASS"},
        {"actual": "0.8", "required": None, "status": "PASS"},
        {"actual": "0.8", "required": "0.8", "status": "MISSING"},
    ],
)
def test_numeric_status_cannot_contradict_evidence(update):
    with pytest.raises(ValidationError):
        CashReleaseCriterion(
            code="EXAM_COVERAGE", window_id="W", scope="F", operator="GE", message="人工约束", **update
        )


def test_individually_consistent_fund_bins_cannot_conflict_with_aggregate(evaluated):
    window = restore_research(evaluated).report.windows[-1].model_copy(deep=True)
    report = window.reliability_per_fund[0].after
    bucket = next(b for b in report.bins if b.count)
    shift = Decimal(".0001") if bucket.mean_score < (bucket.lower + bucket.upper) / 2 else Decimal("-.0001")
    bucket.mean_score += shift
    bucket.absolute_gap = abs(bucket.mean_score - bucket.observed_up_rate)
    report.ece = sum(b.absolute_gap * b.count / report.sample_count for b in report.bins if b.count).quantize(
        Decimal(".00000001"), rounding=ROUND_HALF_UP
    )
    # 该基金自己的分箱数学已自洽，但总体对应档的均值没有一起变化，仍必须拒绝。
    validate_reliability(report, window.after.per_fund[0].metrics)
    with pytest.raises(ValueError, match="aggregate bin value mismatch"):
        validate_window_evidence(window)


@pytest.mark.parametrize("mutation", ["fit_count", "calibration_count", "model_date", "negative_slope"])
def test_model_binding_and_slope_are_checked(evaluated, mutation):
    window = restore_research(evaluated).report.windows[-1].model_copy(deep=True)
    if mutation == "negative_slope":
        # 负斜率是实测失败，而不是“没有模型”；不靠翻转答案或模型方向修饰成绩。
        policy = review.load_release_policy()
        criterion = review._numeric(
            "CALIBRATION_SLOPE", -1, 0, "GT", "人工负斜率", window_id=window.window.window_id, scope="ALL"
        )
        assert criterion.status == "FAIL" and policy.require_monotone_observed_rates
        return
    if mutation == "fit_count":
        window.model.base_model.train_counts_per_fund["006730"] += 1
    elif mutation == "calibration_count":
        window.model.calibrator.counts_per_fund["006730"] += 1
    else:
        window.model = window.model.model_copy(
            update={"calibrator": window.model.calibrator.model_copy(update={"end_date": NOW.date()})}
        )
    with pytest.raises(ValueError, match="mismatch"):
        validate_window_evidence(window)
