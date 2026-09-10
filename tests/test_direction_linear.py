"""单因素训练边界、序列重放与开发候选门槛验证。"""

import sys
from copy import deepcopy
from datetime import date

import pytest
from app.services import direction_linear_runner as runner
from app.services.direction_linear_analysis import accuracy_delta, evaluate
from app.services.direction_linear_models import execute_job, restore, select_fit, validate_job
from app.services.direction_linear_protocol import (
    ABLATION_BRANCHES,
    ABLATION_DROPS,
    ABLATION_VERSION,
    BASELINES,
    BRANCHES,
    RECENCY_BRANCHES,
    RECENCY_VERSION,
    VERSION,
    fit_boundary,
    specification_for,
    specification_v1,
)
from app.services.direction_training_artifacts import digest
from app.services.direction_training_dataset import FUNDS, WINDOWS, exam_dates
from app.services.direction_training_process import run_process
from app.services.trading_calendar import load_calendar


@pytest.fixture
def job():
    calendar = load_calendar()
    days = [d for d in calendar.sessions if date(2021, 1, 4) <= d <= date(2022, 11, 1)][:252]
    fit = []
    for fund in FUNDS:
        for i, day in enumerate(days):
            end = calendar.future_sessions(day)[-1]
            fit.append(
                {
                    "input": {
                        "fund": fund,
                        "cutoff": str(day),
                        "anchor": str(day),
                        "available_at": str(day),
                        "x": [(i * j + 3) % 19 / 19 for j in range(1, 8)],
                        "input_hash": "a" * 64,
                    },
                    "answer": {
                        "fund": fund,
                        "cutoff": str(day),
                        "end": str(end),
                        "available_at": str(end),
                        "y": i % 2,
                        "future_return": "0.01" if i % 2 else "-0.01",
                    },
                }
            )
    window = dict(zip(("name", "fit_end", "cal_end", "exam_end"), WINDOWS[0], strict=True))
    return {
        "version": VERSION,
        "branch": "REFERENCE",
        "window": window,
        "fit": fit,
        "exam": [
            {
                "fund": f,
                "cutoff": "2023-07-03",
                "anchor": "2023-06-30",
                "available_at": "2023-07-03",
                "x": [0.1] * 7,
                "input_hash": "b" * 64,
            }
            for f in FUNDS
        ],
    }


def test_actual_worker_returns_restorable_json_and_same_scores(job):
    output = run_process(job, command=[sys.executable, "-m", "scripts.direction_linear_worker"])
    assert output["status"] == "PREDICTED" and output["model_fit_count"] == 1
    expected = execute_job(job)
    assert expected["models"] == output["models"] and expected["scores"] == output["scores"]
    assert restore(output["models"]["POOLED"])["indices"] == list(range(7))


@pytest.mark.parametrize(
    "mutation",
    [
        "exam_answer",
        "duplicate_fit",
        "fit_future",
        "short_horizon",
        "test_year",
        "new_window",
        "new_branch",
        "extra_path",
    ],
)
def test_rejects_leakage_unplanned_jobs_and_labels(job, mutation):
    if mutation == "exam_answer":
        job["exam"][0]["y"] = 1
    elif mutation == "duplicate_fit":
        job["fit"].append(deepcopy(job["fit"][0]))
    elif mutation == "fit_future":
        job["fit"][0]["answer"]["available_at"] = "2023-01-03"
    elif mutation == "short_horizon":
        job["fit"][0]["answer"]["end"] = "2021-01-06"
    elif mutation == "test_year":
        job["exam"][0]["cutoff"] = "2025-01-02"
    elif mutation == "new_window":
        job["window"]["fit_end"] = "2023-01-31"
    elif mutation == "new_branch":
        job["branch"] = "DEEP_NETWORK"
    else:
        job["source_path"] = "unplanned.json"
    with pytest.raises(ValueError):
        validate_job(job)


def test_recent_fit_changes_only_history_and_retains_minimum(job):
    rows = [{"input": {"cutoff": d}} for d in ("2021-06-29", "2021-06-30", "2021-07-01")]
    assert select_fit(rows, "RECENT_18M", date(2022, 12, 31)) == rows[2:]
    assert select_fit(rows, "REFERENCE", date(2022, 12, 31)) == rows
    job["branch"] = "RECENT_18M"
    with pytest.raises(ValueError, match="FIT_INSUFFICIENT"):
        validate_job(job)


def test_feature_group_removal_preserves_keys_and_ignores_only_removed_columns(job):
    job["branch"] = "DROP_60D_GROUP"
    expected = execute_job(job)
    for row in job["fit"]:
        for index in (2, 4, 5):
            row["input"]["x"][index] = 999.0
    for row in job["exam"]:
        for index in (2, 4, 5):
            row["x"][index] = -999.0
    actual = execute_job(job)
    assert actual["scores"] == expected["scores"]
    assert actual["models"]["POOLED"]["indices"] == [0, 1, 3, 6]
    assert actual["train_counts"] == expected["train_counts"] == dict.fromkeys(FUNDS, 252)


def test_per_fund_labels_cannot_change_other_fund_model(job):
    job["branch"] = "PER_FUND"
    expected = execute_job(job)
    for row in job["fit"]:
        if row["input"]["fund"] == "001632":
            row["answer"]["y"] = 1 - row["answer"]["y"]
            row["answer"]["future_return"] = "0.01" if row["answer"]["y"] else "-0.01"
    actual = execute_job(job)
    assert actual["model_fit_count"] == 3
    assert actual["models"]["001632"] != expected["models"]["001632"]
    for fund in ("006730", "008888"):
        assert actual["models"][fund] == expected["models"][fund]


def test_exam_values_do_not_fit_scaler_or_weights_and_bad_artifact_rejected(job):
    before = execute_job(job)
    job["exam"][0]["x"] = [999.0] * 7
    after = execute_job(job)
    assert before["models"] == after["models"]
    model = deepcopy(before["models"]["POOLED"])
    model["scale"][0] = 0.0
    model["hash"] = digest({k: v for k, v in model.items() if k != "hash"})
    with pytest.raises(ValueError, match="NUMBERS"):
        restore(model)


def exam_records(winner=False):
    protocol = specification_v1()
    predictions, answers = [], []
    for window in protocol["windows"]:
        for index, day in enumerate(
            exam_dates(date.fromisoformat(window["cal_end"]), date.fromisoformat(window["exam_end"]))[1]
        ):
            for fund in FUNDS:
                key, y = f"{fund}:{day}", index % 2
                end = load_calendar().future_sessions(day)[-1]
                answers.append(
                    {
                        "window": window["name"],
                        "sample_key": key,
                        "answer": {
                            "fund": fund,
                            "cutoff": str(day),
                            "end": str(end),
                            "available_at": str(end),
                            "y": y,
                            "future_return": ".01" if y else "-.01",
                        },
                    }
                )
                for branch in (*BRANCHES, *BASELINES):
                    value = 0.9 if branch == "ALWAYS_UP" else 0.1
                    if winner and branch == "RECENT_18M":
                        value = 0.9 if y else 0.1
                    predictions.append(
                        {
                            "window": window["name"],
                            "branch": branch,
                            "sample_key": key,
                            "fund": fund,
                            "cutoff": str(day),
                            "score": value,
                            "predicted_up": int(value > 0.5),
                            "status": "PREDICTED",
                        }
                    )
    return protocol, predictions, answers


@pytest.mark.parametrize("winner", [False, True])
def test_frozen_selection_and_independent_gate_do_not_claim_test_execution(winner):
    protocol, predictions, answers = exam_records(winner)
    result = evaluate(protocol, predictions, answers)
    assert result["selected_candidate"] == ("RECENT_18M" if winner else None)
    assert len(result["candidate_status"]) == 3
    decision = runner.decision(protocol, result, "a" * 64)
    assert not decision["independent_test_run"] and not decision["model_released"]
    assert decision["candidate_rule_frozen"] == winner
    assert not decision["historical_2025_values_read"]


def test_failed_variant_retained_prevents_selective_best_run_reporting():
    protocol, predictions, answers = exam_records(True)
    for row in predictions:
        if row["branch"] == "PER_FUND":
            row.update(score=None, predicted_up=None, status="FAILED")
    result = evaluate(protocol, predictions, answers)
    assert result["same_question_count"] == 0 and result["selected_candidate"] is None
    assert result["failures"]["PER_FUND"]["FAILED"] == len(answers)
    assert all(s["status"] == "INSUFFICIENT_DATA" for s in result["candidate_status"].values())
    assert result["available_descriptive"]["RECENT_18M"]["equal_fund_macro"]["accuracy"] == 1.0


@pytest.mark.parametrize("mutation", ["missing_prediction", "duplicate_answer", "non_finite", "wrong_identity"])
def test_evaluation_rejects_missing_or_invalid_records(mutation):
    protocol, predictions, answers = exam_records()
    if mutation == "missing_prediction":
        predictions.pop()
    elif mutation == "duplicate_answer":
        answers.append(answers[0])
    elif mutation == "non_finite":
        predictions[0]["score"] = float("nan")
        answers[0]["answer"] = None
    else:
        predictions[0]["fund"] = "006730"
    with pytest.raises(ValueError):
        evaluate(protocol, predictions, answers)


def test_score_cannot_export_answers_before_prediction_seal(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "load_plan", lambda _: (specification_v1(), {}))
    monkeypatch.setattr(runner, "source", lambda: pytest.fail("must not read answers before prediction seal"))
    with pytest.raises(FileNotFoundError):
        runner.score(tmp_path)
    assert not list(tmp_path.iterdir())


def test_equal_correct_counts_are_ties_despite_float_summation_order():
    def result(counts):
        return {"per_fund": {f: {"correct_count": n, "sample_count": 43} for f, n in zip(FUNDS, counts, strict=True)}}

    assert sum(n / 43 for n in (10, 32, 25)) > sum(n / 43 for n in (10, 26, 31))
    assert accuracy_delta(result((10, 32, 25)), result((10, 26, 31))) == 0
    assert accuracy_delta(result((10, 33, 25)), result((10, 26, 31))) == pytest.approx(1 / 129)


@pytest.mark.parametrize("branch,index", ABLATION_DROPS.items())
def test_single_feature_ablation_ignores_exactly_one_column_and_preserves_rows(job, branch, index):
    job.update(version=ABLATION_VERSION, branch=branch)
    before = execute_job(job)
    for row in job["fit"]:
        row["input"]["x"][index] = 999.0
    for row in job["exam"]:
        row["x"][index] = -999.0
    after = execute_job(job)
    model = after["models"]["POOLED"]
    assert model["indices"] == [i for i in range(7) if i != index]
    assert len(model["coefficients"]) == 6 and after["model_fit_count"] == 1
    assert after["scores"] == before["scores"]
    assert after["train_counts"] == before["train_counts"] == dict.fromkeys(FUNDS, 252)
    for field in ("mean", "scale", "coefficients", "intercept"):
        assert model[field] == before["models"]["POOLED"][field]
    model["indices"][0] = index
    model["hash"] = digest({k: v for k, v in model.items() if k != "hash"})
    with pytest.raises(ValueError, match="FEATURES"):
        restore(model)


def test_ablation_reference_worker_matches_previous_training_math(job):
    old = execute_job(job)
    job["version"] = ABLATION_VERSION
    actual = run_process(job, command=[sys.executable, "-m", "scripts.direction_linear_worker"])
    assert actual["status"] == "PREDICTED" and actual["scores"] == old["scores"]
    for field in ("mean", "scale", "coefficients", "intercept", "train_hash", "train_counts"):
        assert actual["models"]["POOLED"][field] == old["models"]["POOLED"][field]


@pytest.mark.parametrize(
    "version,branch",
    [
        (ABLATION_VERSION, "RECENT_18M"),
        (ABLATION_VERSION, "DROP_60D_GROUP"),
        (VERSION, "DROP_RETURN_60D"),
        ("UNKNOWN", "REFERENCE"),
    ],
)
def test_branch_names_cannot_leak_between_frozen_studies(job, version, branch):
    job.update(version=version, branch=branch)
    with pytest.raises(ValueError):
        validate_job(job)


@pytest.mark.parametrize("branch", ABLATION_BRANCHES)
@pytest.mark.parametrize("mutation", ["exam_answer", "fit_future", "test_year"])
def test_ablation_retains_answer_isolation_and_chronology(job, branch, mutation):
    job.update(version=ABLATION_VERSION, branch=branch)
    if mutation == "exam_answer":
        job["exam"][0]["y"] = 1
    elif mutation == "fit_future":
        job["fit"][0]["answer"]["available_at"] = "2023-01-03"
    else:
        job["exam"][0]["cutoff"] = "2025-01-02"
    with pytest.raises(ValueError):
        validate_job(job)


@pytest.mark.parametrize("winner", [False, True])
def test_ablation_selection_keeps_all_original_gates_and_reports_new_candidates(winner):
    old, predictions, answers = exam_records(winner)
    protocol = specification_for(ABLATION_VERSION)
    names = dict(zip(BRANCHES, ABLATION_BRANCHES, strict=True))
    for row in predictions:
        row["branch"] = names.get(row["branch"], row["branch"])
    assert protocol["selection"] == old["selection"]
    assert all(protocol[k] == old[k] for k in ("windows", "minimum", "bootstrap", "logistic", "threshold"))
    result = evaluate(protocol, predictions, answers)
    assert set(result["candidate_status"]) == set(ABLATION_BRANCHES[1:])
    assert result["selected_candidate"] == ("DROP_RETURN_60D" if winner else None)
    assert result["version"] == ABLATION_VERSION and not result["independent_test"]
    if winner:
        for row in predictions:
            if row["branch"] == "DROP_POSITION_60D":
                row.update(score=None, predicted_up=None, status="FAILED")
        failed = evaluate(protocol, predictions, answers)
        assert failed["selected_candidate"] is None and failed["same_question_count"] == 0
        assert failed["failures"]["DROP_POSITION_60D"]["FAILED"] == len(answers)


def test_independence_review_reads_only_named_metadata_and_does_not_claim_attestation(monkeypatch):
    from app.services import direction_linear_evidence as evidence

    seen = []

    def read_plan(path):
        seen.append(path.name)
        assert path.name == "independent-test-plan.json"
        return {
            "planned_period": ["2025-01-01", "2025-12-31"],
            "independence_evidence": "NO_GLOBAL_ACCESS_ATTESTATION",
            "source_versions_verified": False,
        }

    monkeypatch.setattr(evidence, "read_json", read_plan)
    monkeypatch.setattr(evidence, "read_seal", lambda folder, name: {"manifest_hash": evidence.P1_HASH})
    result = evidence.independence_review(
        {"independent_plan": {"historical_independence_attested": False, "source_first_versions_verified": False}}
    )
    assert seen == ["independent-test-plan.json"]
    assert not result["independence_passed"] and not result["read_2025_values"]
    assert result["future_start_date"] is None


def recent_job(job):
    payload = deepcopy(job)
    payload.update(version=RECENCY_VERSION, branch="MATURE_EXPANDING")
    for fund in FUNDS:
        row = deepcopy(next(r for r in job["fit"] if r["input"]["fund"] == fund))
        cutoff = date(2023, 5, 31)
        row["input"].update(cutoff=str(cutoff), anchor="2023-05-30", available_at=str(cutoff))
        row["answer"].update(
            cutoff=str(cutoff), end=str(load_calendar().future_sessions(cutoff)[-1]), available_at="2023-06-30"
        )
        payload["fit"].append(row)
    return payload


@pytest.mark.parametrize("branch", RECENCY_BRANCHES[1:])
def test_recency_accepts_only_labels_mature_before_exam_without_changing_old_reference(job, branch):
    payload = recent_job(job)
    payload["branch"] = branch
    fit, exam = validate_job(payload)
    assert len(fit) == len(job["fit"]) + 3
    assert max(a.available_at for _, a in fit) < min(i.cutoff for i in exam)
    assert fit_boundary(RECENCY_VERSION, branch, payload["window"]) == "2023-06-30"
    payload["branch"] = "REFERENCE"
    with pytest.raises(ValueError, match="FIT_BOUNDARY"):
        validate_job(payload)


@pytest.mark.parametrize("branch", RECENCY_BRANCHES[1:])
@pytest.mark.parametrize("mutation", ["immature", "short_target", "exam_answer", "protected_year", "wrong_version"])
def test_recency_worker_rejects_temporal_leakage_and_protocol_mixing(job, branch, mutation):
    payload = recent_job(job)
    payload["branch"] = branch
    if mutation == "immature":
        payload["fit"][-1]["answer"]["available_at"] = "2023-07-03"
    elif mutation == "short_target":
        payload["fit"][-1]["answer"]["end"] = "2023-06-01"
    elif mutation == "exam_answer":
        payload["exam"][0]["y"] = 1
    elif mutation == "protected_year":
        payload["exam"][0]["cutoff"] = "2025-01-02"
    else:
        payload["version"] = ABLATION_VERSION
    with pytest.raises(ValueError):
        validate_job(payload)


def test_recency_actual_worker_keeps_model_boundary_and_exam_out_of_fit(job):
    payload = recent_job(job)
    before = run_process(payload, command=[sys.executable, "-m", "scripts.direction_linear_worker"])
    assert before["status"] == "PREDICTED"
    model = before["models"]["POOLED"]
    assert model["fit_end"] == "2023-06-30" and model["indices"] == list(range(7))
    payload["exam"][0]["x"] = [1000.0] * 7
    after = execute_job(payload)
    assert model == after["models"]["POOLED"]
    reference = execute_job(job)
    job["version"] = RECENCY_VERSION
    actual = execute_job(job)
    assert actual["scores"] == reference["scores"]


def test_recency_count_control_uses_dates_not_labels():
    from app.services.direction_linear_recency import latest_rows

    mature = [
        {"input": {"fund": f, "cutoff": f"2023-01-0{i}"}, "answer": {"y": i % 2}} for f in FUNDS for i in range(1, 5)
    ]
    reference = [r for r in mature if r["input"]["cutoff"] <= "2023-01-02"]
    selected = latest_rows(reference, list(reversed(mature)))
    assert len(selected) == len(reference)
    assert all(r["input"]["cutoff"] in ("2023-01-03", "2023-01-04") for r in selected)
    changed = deepcopy(mature)
    for row in changed:
        row["answer"]["y"] = 1 - row["answer"]["y"]
    assert [r["input"] for r in latest_rows(reference, changed)] == [r["input"] for r in selected]
    with pytest.raises(ValueError, match="DUPLICATE"):
        latest_rows(reference, mature + [mature[0]])
    with pytest.raises(ValueError, match="SHORTAGE"):
        latest_rows(reference, mature[:1])


def test_recency_make_job_keeps_exact_exam_inputs_and_selects_only_declared_training(job):
    bundle = {
        "window": job["window"],
        "complete": {"fit": job["fit"], "exam": {"CLEAN": job["exam"]}},
        "recent_fit": {b: recent_job(job)["fit"] for b in RECENCY_BRANCHES[1:]},
    }
    for branch in RECENCY_BRANCHES:
        result = runner.make_job(bundle, branch, RECENCY_VERSION)
        assert result["exam"] is job["exam"]
        assert result["fit"] is (job["fit"] if branch == "REFERENCE" else bundle["recent_fit"][branch])


@pytest.mark.parametrize("winner", [False, True])
def test_recency_has_unchanged_gates_fixed_budget_and_no_automatic_independent_pass(winner):
    old, predictions, answers = exam_records(winner)
    protocol = specification_for(RECENCY_VERSION)
    predictions = [r for r in predictions if r["branch"] != "PER_FUND"]
    for row in predictions:
        row["branch"] = {"RECENT_18M": "MATURE_EXPANDING", "DROP_60D_GROUP": "MATURE_SAME_COUNT"}.get(
            row["branch"], row["branch"]
        )
    assert all(protocol[k] == old[k] for k in ("windows", "selection", "bootstrap", "minimum", "threshold", "logistic"))
    assert protocol["research_fit_budget"]["maximum_models"] == 9
    result = evaluate(protocol, predictions, answers)
    assert result["selected_candidate"] == ("MATURE_EXPANDING" if winner else None)
    assert not runner.decision(protocol, result, "a" * 64)["independent_test_run"]


def audit_fixture():
    from decimal import Decimal
    from types import SimpleNamespace

    from app.services import direction_nav_data as nav_data

    cutoff = date(2023, 6, 30)
    days = nav_data.history_dates(cutoff)
    nav = {d: SimpleNamespace(unit_nav=Decimal(100 + i) / 100, nav_date=d) for i, d in enumerate(days)}
    item, audit, issues = nav_data.build_input("001632", cutoff, nav, ())
    assert not issues
    return item, {"audit": {"CLEAN": audit}}, nav


def test_independent_feature_audit_detects_changed_numbers_and_ignores_future_nav():
    from decimal import Decimal
    from types import SimpleNamespace

    from app.services.direction_linear_recency import audit_input

    item, record, nav = audit_fixture()
    audit_input(item, record, nav, ())
    nav[item.cutoff] = SimpleNamespace(unit_nav=Decimal("9999"))
    audit_input(item, record, nav, ())
    bad = item.model_dump(mode="json")
    bad["x"][1] += 0.01
    with pytest.raises(ValueError, match="FEATURE_MISMATCH"):
        audit_input(bad, record, nav, ())
    record["audit"]["CLEAN"]["dates"][-1] = str(item.cutoff)
    with pytest.raises(ValueError, match="INPUT_HISTORY"):
        audit_input(item, record, nav, ())


def test_independent_label_audit_checks_exact_target_and_cash_reinvestment():
    from decimal import Decimal
    from types import SimpleNamespace

    from app.services import direction_nav_data as nav_data
    from app.services.direction_linear_recency import audit_answer, independent_series

    cutoff = date(2023, 6, 30)
    future = load_calendar().future_sessions(cutoff)
    dates = (cutoff, *future)
    nav = {d: SimpleNamespace(unit_nav=Decimal("1.0"), nav_date=d) for d in dates}
    event = SimpleNamespace(
        ex_date=future[2],
        nav_ex_date=future[2],
        ann_date=cutoff,
        implementation_ann_date=cutoff,
        cash_dividend=Decimal("0.1"),
        process_status="实施",
        event_key="audit-cash",
    )
    series, available = independent_series(dates, nav, (event,), date(2024, 12, 31))
    assert series[-1] == 110 and available == nav_data.assumed_available(future[-1])
    answer, issues = nav_data.build_answer("001632", cutoff, nav, (event,), include_value=True)
    assert not issues and Decimal(answer.future_return) == Decimal("0.1")
    audit_answer(answer, nav, (event,))
    bad = answer.model_dump(mode="json")
    bad["future_return"] = "0.2"
    with pytest.raises(ValueError, match="LABEL_MISMATCH"):
        audit_answer(bad, nav, (event,))
    bad["end"] = str(future[-2])
    with pytest.raises(ValueError, match="LABEL_DATES"):
        audit_answer(bad, nav, (event,))
