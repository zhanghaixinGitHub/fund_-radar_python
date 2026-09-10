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
    COMBINATION_BRANCHES,
    COMBINATION_C,
    COMBINATION_VERSION,
    COVERAGE_BRANCHES,
    COVERAGE_VERSION,
    RECENCY_BRANCHES,
    RECENCY_VERSION,
    REGULARIZATION_BRANCHES,
    REGULARIZATION_C,
    REGULARIZATION_VERSION,
    VERSION,
    fit_boundary,
    planned_dates,
    specification_for,
    specification_v1,
    study_windows,
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


def test_regularization_shrinks_weights_while_preserving_training_and_scaler(job):
    old = execute_job(job)
    job["version"] = REGULARIZATION_VERSION
    outputs = {}
    for branch in REGULARIZATION_BRANCHES:
        job["branch"] = branch
        outputs[branch] = execute_job(job)
    reference = outputs["REFERENCE"]["models"]["POOLED"]
    assert outputs["REFERENCE"]["scores"] == old["scores"]
    norms = []
    for branch, output in outputs.items():
        model = restore(output["models"]["POOLED"])
        assert model["C"] == REGULARIZATION_C[branch] and model["penalty"] == "L2"
        assert 0 < model["solver_iterations"] < 1000
        for name in ("indices", "mean", "scale", "train_hash", "train_counts", "fit_end"):
            assert model[name] == reference[name] == old["models"]["POOLED"][name]
        norms.append(sum(c * c for c in model["coefficients"]))
    assert norms[0] > norms[1] > norms[2]


@pytest.mark.parametrize("branch", REGULARIZATION_BRANCHES[1:])
def test_regularization_actual_worker_is_reproducible_and_exam_does_not_fit(job, branch):
    job.update(version=REGULARIZATION_VERSION, branch=branch)
    worker = run_process(job, command=[sys.executable, "-m", "scripts.direction_linear_worker"])
    assert worker["status"] == "PREDICTED"
    expected = execute_job(job)
    assert worker["models"] == expected["models"] and worker["scores"] == expected["scores"]
    job["exam"][0]["x"] = [999.0] * 7
    assert execute_job(job)["models"] == worker["models"]


@pytest.mark.parametrize("branch", REGULARIZATION_BRANCHES)
@pytest.mark.parametrize("mutation", ["C_override", "fit_future", "exam_answer", "new_branch", "new_year"])
def test_regularization_only_accepts_fixed_jobs_and_original_time_boundaries(job, branch, mutation):
    job.update(version=REGULARIZATION_VERSION, branch=branch)
    if mutation == "C_override":
        job["C"] = 0.3
    elif mutation == "fit_future":
        job["fit"][0]["answer"]["available_at"] = "2023-01-03"
    elif mutation == "exam_answer":
        job["exam"][0]["y"] = 1
    elif mutation == "new_branch":
        job["branch"] = "MATURE_EXPANDING"
    else:
        job["exam"][0]["cutoff"] = "2025-01-02"
    with pytest.raises(ValueError):
        validate_job(job)


def test_regularization_artifact_rejects_changed_hyperparameters_and_legacy_version(job):
    job.update(version=REGULARIZATION_VERSION, branch="L2_STRONG")
    original = execute_job(job)["models"]["POOLED"]
    for key, value in (("C", 1.0), ("penalty", "L1"), ("solver_iterations", 1000), ("version", VERSION)):
        model = {**original, key: value}
        model["hash"] = digest({k: v for k, v in model.items() if k != "hash"})
        with pytest.raises(ValueError):
            restore(model)


@pytest.mark.parametrize("winner", [False, True])
def test_regularization_preserves_gates_and_retains_failed_variants(winner):
    original, predictions, answers = exam_records(winner)
    protocol = specification_for(REGULARIZATION_VERSION)
    assert all(
        protocol[k] == original[k] for k in ("windows", "minimum", "features", "selection", "bootstrap", "threshold")
    )
    assert protocol["parameter_search"] and protocol["C_by_branch"] == REGULARIZATION_C
    assert protocol["research_fit_budget"]["maximum_jobs"] == 9
    predictions = [r for r in predictions if r["branch"] != "PER_FUND"]
    for row in predictions:
        row["branch"] = {"RECENT_18M": "L2_STRONG", "DROP_60D_GROUP": "L2_STRONGER"}.get(row["branch"], row["branch"])
    result = evaluate(protocol, predictions, answers)
    assert result["selected_candidate"] == ("L2_STRONG" if winner else None)
    assert not runner.decision(protocol, result, "a" * 64)["independent_test_run"]
    for row in predictions:
        if row["branch"] == "L2_STRONGER":
            row.update(score=None, predicted_up=None, status="FAILED")
    failed = evaluate(protocol, predictions, answers)
    assert failed["selected_candidate"] is None and failed["same_question_count"] == 0


def test_regularization_bundle_does_not_use_previous_recency_variants(job):
    bundle = {
        "window": job["window"],
        "complete": {"fit": job["fit"], "exam": {"CLEAN": job["exam"]}},
        "recent_fit": {b: [{"invalid": True}] for b in RECENCY_BRANCHES[1:]},
    }
    for branch in REGULARIZATION_BRANCHES:
        payload = runner.make_job(bundle, branch, REGULARIZATION_VERSION)
        assert payload["fit"] is job["fit"] and payload["exam"] is job["exam"]
        validate_job(payload)


@pytest.mark.parametrize(
    "branch,old_version",
    [("REFERENCE", REGULARIZATION_VERSION), ("DROP_60D_GROUP", VERSION), ("L2_STRONGER", REGULARIZATION_VERSION)],
)
def test_combination_controls_preserve_prior_training_math(job, branch, old_version):
    job.update(version=old_version, branch=branch)
    old = execute_job(job)
    job["version"] = COMBINATION_VERSION
    current = execute_job(job)
    assert current["scores"] == old["scores"]
    model, previous = current["models"]["POOLED"], old["models"]["POOLED"]
    assert all(model[k] == v for k, v in previous.items() if k not in ("hash", "version"))


def test_combination_worker_uses_four_columns_strong_penalty_and_no_exam_fit(job):
    job.update(version=COMBINATION_VERSION, branch="DROP_60D_GROUP_L2")
    actual = run_process(job, command=[sys.executable, "-m", "scripts.direction_linear_worker"])
    expected = execute_job(job)
    assert actual["status"] == "PREDICTED"
    assert actual["scores"] == expected["scores"] and actual["models"] == expected["models"]
    model = restore(actual["models"]["POOLED"])
    assert model["indices"] == [0, 1, 3, 6] and model["C"] == 0.01
    job["exam"][0]["x"] = [999.0] * 7
    assert execute_job(job)["models"] == actual["models"]
    job["exam"][0]["x"] = [0.1] * 7
    for row in job["fit"]:
        for index in (2, 4, 5):
            row["input"]["x"][index] = 999.0
    assert execute_job(job)["scores"] == actual["scores"]
    job["branch"] = "DROP_60D_GROUP"
    weak = execute_job(job)["models"]["POOLED"]
    assert weak["mean"] == model["mean"] and weak["scale"] == model["scale"]
    assert sum(c * c for c in model["coefficients"]) < sum(c * c for c in weak["coefficients"])


@pytest.mark.parametrize("branch", COMBINATION_BRANCHES)
@pytest.mark.parametrize("mutation", ["C", "future_label", "exam_answer", "year_2025", "branch"])
def test_combination_rejects_unplanned_changes_and_leakage(job, branch, mutation):
    job.update(version=COMBINATION_VERSION, branch=branch)
    if mutation == "C":
        job["C"] = 0.001
    elif mutation == "future_label":
        job["fit"][0]["answer"]["available_at"] = "2023-01-03"
    elif mutation == "exam_answer":
        job["exam"][0]["y"] = 1
    elif mutation == "year_2025":
        job["exam"][0]["cutoff"] = "2025-01-02"
    else:
        job["branch"] = "MATURE_EXPANDING"
    with pytest.raises(ValueError):
        validate_job(job)


def test_combination_saved_model_cannot_change_penalty_or_columns(job):
    job.update(version=COMBINATION_VERSION, branch="DROP_60D_GROUP_L2")
    original = execute_job(job)["models"]["POOLED"]
    for key, value in (("C", 1.0), ("indices", [0, 1, 2, 6]), ("version", REGULARIZATION_VERSION)):
        changed = {**original, key: value}
        changed["hash"] = digest({k: v for k, v in changed.items() if k != "hash"})
        with pytest.raises(ValueError):
            restore(changed)


@pytest.mark.parametrize("winner", [False, True])
def test_combination_preserves_selection_and_closes_search_even_with_a_winner(winner):
    original, predictions, answers = exam_records(winner)
    protocol = specification_for(COMBINATION_VERSION)
    assert all(protocol[k] == original[k] for k in ("windows", "minimum", "selection", "bootstrap", "threshold"))
    assert protocol["C_by_branch"] == COMBINATION_C
    assert protocol["new_hypotheses"] == ["DROP_60D_GROUP_L2"]
    assert protocol["research_fit_budget"]["maximum_jobs"] == 12
    for row in predictions:
        row["branch"] = {"RECENT_18M": "DROP_60D_GROUP_L2", "PER_FUND": "L2_STRONGER"}.get(row["branch"], row["branch"])
    result = evaluate(protocol, predictions, answers)
    assert result["selected_candidate"] == ("DROP_60D_GROUP_L2" if winner else None)
    value = runner.decision(protocol, result, "a" * 64)
    assert value["current_snapshot_combination_search"] == "CLOSED"
    assert not value["automatic_followup_training"] and not value["independent_test_run"]
    assert not value["historical_2025_values_read"] and not value["model_released"]
    for row in predictions:
        if row["branch"] == "DROP_60D_GROUP_L2":
            row.update(score=None, predicted_up=None, status="FAILED")
    failed = evaluate(protocol, predictions, answers)
    assert failed["selected_candidate"] is None and failed["same_question_count"] == 0
    assert failed["failures"]["DROP_60D_GROUP_L2"]["FAILED"] == len(answers)


@pytest.mark.parametrize("mutation", ["none", "sample", "model", "score"])
def test_combination_control_parity_detects_changed_data_models_and_predictions(job, tmp_path, monkeypatch, mutation):
    from app.services import direction_linear_combination as combination
    from app.services.direction_training_artifacts import write_json

    job.update(version=REGULARIZATION_VERSION)
    old = execute_job(job)
    job["version"] = COMBINATION_VERSION
    current = execute_job(job)
    if mutation == "model":
        current["models"]["POOLED"]["coefficients"][0] += 0.01
    if mutation == "score":
        current["scores"][0] += 0.01
    prior, folder = tmp_path / "prior", tmp_path / "new"
    prior.mkdir()
    folder.mkdir()
    name = job["window"]["name"]
    for location, output in ((prior, old), (folder, current)):
        write_json(location / f"linear-models-{name}.json", {"REFERENCE": output})
        write_json(
            location / f"linear-prepared-{name}.json", {"data": int(mutation == "sample" and location == folder)}
        )
    monkeypatch.setattr(combination, "CONTROLS", {"REFERENCE": "f5905e47-097f-4df7-bf85-3c3fdbc64c16"})
    monkeypatch.setattr(combination, "run_folder", lambda _: prior)
    protocol = {"version": COMBINATION_VERSION, "windows": [job["window"]]}
    if mutation != "none":
        with pytest.raises(ValueError, match="COMBINATION"):
            combination.control_parity(folder, protocol)
    else:
        result = combination.control_parity(folder, protocol)
        assert result[0]["model_values_identical"] and result[0]["max_difference"] == 0


def test_combination_effects_do_not_assume_additive_improvements():
    from app.services.direction_linear_combination import effect_summary

    groups = {
        b: {"per_fund": {"001632": {"sample_count": 100, "correct_count": n}}}
        for b, n in zip(COMBINATION_BRANCHES, (50, 60, 65, 55), strict=True)
    }
    result = effect_summary(groups)
    assert result["feature_effect_at_C1"] == 0.1
    assert result["feature_effect_at_C001"] == -0.1
    assert result["penalty_effect_with_four_features"] == -0.05
    assert result["accuracy_interaction"] == -0.2


@pytest.mark.parametrize("failed", [False, True])
def test_combination_diagnosis_handles_four_column_models_and_failed_branch(job, tmp_path, failed):
    from app.services.direction_linear_combination import combination_diagnostics
    from app.services.direction_training_artifacts import write_json, write_jsonl
    from app.services.direction_training_evaluation import grouped_metrics

    protocol = specification_for(COMBINATION_VERSION)
    protocol["windows"] = [job["window"]]
    name = job["window"]["name"]
    outputs, predictions, answers = {}, [], []
    for item in job["exam"]:
        cutoff = date.fromisoformat(item["cutoff"])
        end = str(load_calendar().future_sessions(cutoff)[-1])
        answers.append(
            {
                "window": name,
                "sample_key": f"{item['fund']}:{cutoff}",
                "answer": {
                    "fund": item["fund"],
                    "cutoff": str(cutoff),
                    "end": end,
                    "available_at": end,
                    "y": 1,
                    "future_return": "0.01",
                },
            }
        )
    groups = {}
    for branch in COMBINATION_BRANCHES:
        job.update(version=COMBINATION_VERSION, branch=branch)
        output = execute_job(job)
        outputs[branch] = {"status": "FAILED"} if failed and branch == "DROP_60D_GROUP_L2" else output
        rows = [
            {
                "window": name,
                "sample_key": f"{item['fund']}:{item['cutoff']}",
                "fund": item["fund"],
                "cutoff": item["cutoff"],
                "branch": branch,
                "y": 1,
                "score": score,
                "predicted_up": int(score > 0.5),
                "status": outputs[branch]["status"],
            }
            for item, score in zip(job["exam"], output["scores"], strict=True)
        ]
        predictions.extend(rows)
        groups[branch] = grouped_metrics([] if failed else rows, FUNDS)
    write_json(
        tmp_path / f"linear-prepared-{name}.json", {"complete": {"fit": job["fit"], "exam": {"CLEAN": job["exam"]}}}
    )
    write_json(tmp_path / f"linear-models-{name}.json", outputs)
    write_jsonl(tmp_path / "linear-predictions.jsonl", predictions)
    result = {"same_question_count": 0 if failed else 3, "common": groups, "per_window": {name: groups}}
    report = combination_diagnostics(tmp_path, protocol, answers, result)
    assert report["same_question_count"] == result["same_question_count"] and report["search_closed"]
    assert len(report["windows"][name]["branches"]["DROP_60D_GROUP"]["coefficients"]) == 4
    assert sum(r["count"] for r in report["error_transitions"]["REFERENCE"]) == (0 if failed else 3)
    assert not report["independent_test"]


def test_full_quarter_plan_includes_cross_quarter_targets_and_preserves_2025():
    windows = study_windows(COVERAGE_VERSION)
    calendar = load_calendar()
    assert len(windows) == 8
    dates = [planned_dates(COVERAGE_VERSION, w) for w in windows]
    assert [len(ds) for ds in dates] == [59, 59, 64, 60, 58, 59, 64, 40]
    assert sum(map(len, dates)) == len(set(d for ds in dates for d in ds)) == 463
    assert dates[-1][-1] == date(2024, 12, 2)
    assert calendar.future_sessions(dates[0][-1])[-1] > date(2023, 3, 31)
    assert all(
        str(calendar.future_sessions(ds[-1], 21)[-1]) == w["label_asof"] <= "2024-12-31"
        for w, ds in zip(windows, dates, strict=True)
    )
    old = {w["cal_end"]: w for w in study_windows(VERSION)}
    assert all(w["fit_end"] == old[w["cal_end"]]["fit_end"] for w in windows if w["cal_end"] in old)
    assert len(planned_dates(VERSION, old["2023-06-30"])) == 44


@pytest.mark.parametrize("branch", COVERAGE_BRANCHES)
def test_full_quarter_worker_reuses_locked_model_and_can_predict_quarter_end(job, branch):
    job.update(version=COMBINATION_VERSION, branch=branch)
    old = execute_job(job)
    job.update(version=COVERAGE_VERSION, window=study_windows(COVERAGE_VERSION)[2])
    current = run_process(job, command=[sys.executable, "-m", "scripts.direction_linear_worker"])
    assert current["status"] == "PREDICTED" and current["scores"] == old["scores"]
    model, previous = current["models"]["POOLED"], old["models"]["POOLED"]
    assert all(model[k] == v for k, v in previous.items() if k not in ("version", "hash"))
    for item in job["exam"]:
        item.update(cutoff="2023-09-28", anchor="2023-09-27", available_at="2023-09-28")
    fit, exam = validate_job(job)
    assert len(exam) == 3 and len(fit) == 756
    assert execute_job(job)["models"] == current["models"]


@pytest.mark.parametrize("branch", COVERAGE_BRANCHES)
@pytest.mark.parametrize(
    "mutation", ["C", "threshold", "late_fit", "year_2025", "december_tail", "label_asof", "new_branch"]
)
def test_full_quarter_worker_rejects_unfrozen_changes_and_future_periods(job, branch, mutation):
    job.update(version=COVERAGE_VERSION, branch=branch, window=study_windows(COVERAGE_VERSION)[2])
    if mutation in ("C", "threshold"):
        job[mutation] = 0.3
    elif mutation == "late_fit":
        job["fit"][0]["answer"]["available_at"] = "2023-01-03"
    elif mutation == "year_2025":
        job["exam"][0]["cutoff"] = "2025-01-02"
    elif mutation == "december_tail":
        job["window"] = study_windows(COVERAGE_VERSION)[-1]
        for item in job["exam"]:
            item.update(cutoff="2024-12-03", anchor="2024-12-02", available_at="2024-12-03")
    elif mutation == "label_asof":
        job["window"]["label_asof"] = "2025-01-01"
    else:
        job["branch"] = "L2_STRONGER"
    with pytest.raises(ValueError):
        validate_job(job)


def test_full_quarter_selection_uses_maturity_not_label_direction():
    from app.services.direction_linear_coverage import select_records

    window = study_windows(COVERAGE_VERSION)[2]
    records = {}
    for cutoff, available in (("2022-11-30", "2022-12-30"), ("2022-12-01", "2023-01-03"), ("2023-09-28", "2023-11-06")):
        records[cutoff] = {
            "cutoff": cutoff,
            "inputs": {"CLEAN": {"cutoff": cutoff}},
            "label": {"available_at": available},
            "input_issues": {"CLEAN": []},
            "label_issues": [],
        }
    fit, exam, counts = select_records(records, window)
    assert [r["cutoff"] for r in fit] == ["2022-11-30"]
    assert [r["cutoff"] for r in exam] == ["2023-09-28"]
    assert counts["scorable_count"] == 1
    records["2023-09-28"]["label"]["available_at"] = "2023-11-07"
    assert select_records(records, window)[2]["scorable_count"] == 0


def coverage_exam_records(winner):
    protocol = specification_for(COVERAGE_VERSION)
    predictions, answers = [], []
    for window in protocol["windows"]:
        for i, cutoff in enumerate(planned_dates(COVERAGE_VERSION, window)):
            future = load_calendar().future_sessions(cutoff, 21)
            for fund in FUNDS:
                y = i % 2
                answers.append(
                    {
                        "window": window["name"],
                        "sample_key": f"{fund}:{cutoff}",
                        "answer": {
                            "fund": fund,
                            "cutoff": str(cutoff),
                            "end": str(future[19]),
                            "available_at": str(future[20]),
                            "y": y,
                            "future_return": "0.01" if y else "-0.01",
                        },
                    }
                )
                for branch in (*COVERAGE_BRANCHES, *BASELINES):
                    score = 0.9 if (branch == "ALWAYS_UP" or winner and branch == "DROP_60D_GROUP_L2" and y) else 0.1
                    predictions.append(
                        {
                            "window": window["name"],
                            "sample_key": f"{fund}:{cutoff}",
                            "fund": fund,
                            "cutoff": str(cutoff),
                            "branch": branch,
                            "score": score,
                            "predicted_up": int(score > 0.5),
                            "status": "PREDICTED",
                        }
                    )
    return protocol, predictions, answers


@pytest.mark.parametrize("winner", [False, True])
def test_full_quarter_evaluation_counts_all_mature_targets_and_keeps_original_gates(winner):
    protocol, predictions, answers = coverage_exam_records(winner)
    assert protocol["selection"] == specification_v1()["selection"]
    assert not protocol["parameter_search"] and protocol["research_fit_budget"]["maximum_models"] == 16
    result = evaluate(protocol, predictions, answers)
    assert result["same_question_count"] == 1389 and len(result["valid_windows"]) == 8
    assert result["time_blocks"]["count"] == 19
    assert result["selected_candidate"] == ("DROP_60D_GROUP_L2" if winner else None)
    decision = runner.decision(protocol, result, "a" * 64)
    assert not decision["independent_test_run"] and not decision["model_released"]
    assert decision["current_snapshot_combination_search"] == "CLOSED"
    for row in predictions:
        if row["branch"] == "DROP_60D_GROUP_L2" and row["window"] == protocol["windows"][0]["name"]:
            row.update(status="FAILED", score=None, predicted_up=None)
    failed = evaluate(protocol, predictions, answers)
    assert failed["selected_candidate"] is None and failed["failures"]["DROP_60D_GROUP_L2"]["FAILED"] == 177


def test_full_quarter_blocks_never_compress_a_missing_date():
    from app.services.direction_training_evaluation import complete_blocks

    protocol, _, answers = coverage_exam_records(False)
    common = {(r["window"], r["sample_key"]) for r in answers}
    plans = {w["name"]: planned_dates(COVERAGE_VERSION, w) for w in protocol["windows"]}
    blocks, _ = complete_blocks(protocol, common, planned_by_window=plans)
    first = blocks[0]
    common.remove(first["keys"][0])
    reduced, excluded = complete_blocks(protocol, common, planned_by_window=plans)
    assert len(reduced) == 18 and excluded[first["window"]]["incomplete_blocks"] == 1
    assert reduced[0] == blocks[1]


def test_full_quarter_answers_cannot_be_exported_before_prediction_seal(tmp_path, monkeypatch):
    from app.services import direction_linear_coverage as coverage

    monkeypatch.setattr(runner, "load_plan", lambda _: (specification_for(COVERAGE_VERSION), {}))
    monkeypatch.setattr(
        coverage, "export_answers", lambda *_: pytest.fail("answers exported before predictions sealed")
    )
    with pytest.raises(FileNotFoundError):
        runner.score(tmp_path)


@pytest.mark.parametrize("mutation", ["none", "score", "answer"])
def test_full_quarter_old_cohort_requires_same_answers_and_predictions(tmp_path, monkeypatch, mutation):
    from app.services import direction_linear_coverage as coverage
    from app.services.direction_training_artifacts import write_json, write_jsonl

    prior = tmp_path / "prior"
    current = tmp_path / "current"
    prior.mkdir()
    current.mkdir()
    window = study_windows(COVERAGE_VERSION)[2]
    protocol = {"version": COVERAGE_VERSION, "branches": ["REFERENCE"], "windows": [window]}
    answer = {"y": 1, "available_at": "2023-07-31", "future_return": "0.01"}
    old = {
        "window": "OLD",
        "sample_key": "001632:2023-07-03",
        "fund": "001632",
        "branch": "REFERENCE",
        "score": 0.6,
        "predicted_up": 1,
        "status": "PREDICTED",
    }
    write_json(prior / "linear-plan.json", {"windows": [{"name": "OLD", "exam_end": "2023-09-30"}]})
    write_jsonl(prior / "linear-answers.jsonl", [{"window": "OLD", "sample_key": old["sample_key"], "answer": answer}])
    write_jsonl(prior / "linear-predictions.jsonl", [old])
    row = {**old, "window": window["name"], "score": 0.7 if mutation == "score" else 0.6}
    write_jsonl(current / "linear-predictions.jsonl", [row])
    answers = [
        {
            "window": window["name"],
            "sample_key": row["sample_key"],
            "answer": {**answer, "future_return": "0.02"} if mutation == "answer" else answer,
        }
    ]
    monkeypatch.setattr(coverage, "prior_study", lambda: prior)
    monkeypatch.setattr(coverage, "regularization_diagnostics", lambda *_: {})
    if mutation != "none":
        with pytest.raises(ValueError, match="PREVIOUS_SCORED_ROW_CHANGED"):
            coverage.diagnostics(current, protocol, answers)
    else:
        report = coverage.diagnostics(current, protocol, answers)
        assert report["previous_scored_parity"]["REFERENCE"]["verified_count"] == 1
        assert report["previous_scored_parity"]["REFERENCE"]["complete"]
        assert report["cohorts"]["ADDED_DEVELOPMENT_ROWS"]["REFERENCE"]["sample_weighted"] is None
