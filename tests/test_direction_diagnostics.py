"""样本量诊断的日期选择、原输入边界与结果完整性检查。"""

from copy import deepcopy

import pytest
from app.services import direction_diagnostics as d
from app.services.direction_linear_models import execute_job, predict_model, validate_job
from tests.test_direction_linear import job  # noqa: F401


def synthetic_rows(n=703):
    """日期索引测试不读取真实净值，也不使用标签来选择训练行。"""
    return [
        {"input": {"fund": fund, "cutoff": f"date-{i:04d}"}, "answer": {"y": i % 2}}
        for fund in d.FUNDS
        for i in range(n)
    ]


@pytest.mark.parametrize("n", [278, 343, 403, 462, 521, 585, 645, 703])
def test_nested_date_selection_same_span_and_count(n):
    small = d.spread_indices(n, 252)
    middle = d.spread_indices(n, (252 + n) // 2)
    full = d.spread_indices(n, n)
    assert len(small) == 252 and len(middle) == (252 + n) // 2
    assert {0, n - 1} <= small <= middle <= full == set(range(n))
    assert d.spread_indices(n, 252) == small


def test_date_selection_ignores_answers_and_preserves_all_funds():
    rows = synthetic_rows()
    altered = deepcopy(rows)
    for row in altered:
        row["answer"]["y"] = 1 - row["answer"]["y"]
    for mode in d.ALL_MODES:
        selected = d.select_rows(rows, mode)
        other = d.select_rows(altered, mode)
        assert [r["input"] for r in selected] == [r["input"] for r in other]
        dates = [{r["input"]["cutoff"] for r in selected if r["input"]["fund"] == f} for f in d.FUNDS]
        assert dates[0] == dates[1] == dates[2]
        assert all(r in rows for r in selected)
    recent = d.select_rows(rows, "RECENT_252")
    assert {r["input"]["cutoff"] for r in recent} == {f"date-{i:04d}" for i in range(451, 703)}


@pytest.mark.parametrize("mutation", ["short", "mismatch", "duplicate", "unknown_fund"])
def test_population_mismatch_is_rejected(mutation):
    rows = synthetic_rows(252 if mutation == "short" else 278)
    if mutation in ("short", "mismatch"):
        rows.pop()
    elif mutation == "duplicate":
        rows[1]["input"]["cutoff"] = rows[0]["input"]["cutoff"]
    else:
        rows[0]["input"]["fund"] = "999999"
    with pytest.raises(ValueError, match="DIAGNOSTIC_"):
        d.select_rows(rows, "SPREAD_252")


def test_jobs_keep_maturity_and_exam_answer_boundaries(job):  # noqa: F811
    source = {"window": job["window"], "complete": {"fit": job["fit"], "exam": {"CLEAN": job["exam"]}}}
    jobs = d.make_jobs(source, job["version"])
    for payload in jobs.values():
        fit, exam = validate_job(payload)
        assert len(fit) == 756 and len(exam) == 3
    broken = deepcopy(jobs["SPREAD_252"])
    broken["exam"][0]["y"] = 1
    with pytest.raises(ValueError):
        validate_job(broken)
    broken = deepcopy(jobs["SPREAD_252"])
    broken["fit"][0]["answer"]["available_at"] = "2024-12-31"
    with pytest.raises(ValueError, match="LINEAR_FIT_BOUNDARY"):
        validate_job(broken)


def test_identical_selected_population_keeps_original_linear_numbers(job):  # noqa: F811
    source = {"window": job["window"], "complete": {"fit": job["fit"], "exam": {"CLEAN": job["exam"]}}}
    jobs = d.make_jobs(source, job["version"])
    original = execute_job(job)
    selected = execute_job(jobs["SPREAD_252"])
    assert original["models"] == selected["models"]
    _, exam = validate_job(job)
    assert predict_model(selected["models"]["POOLED"], exam) == original["scores"]


def test_correlation_preserves_unknown_constant_and_sign():
    assert d.correlation([1, 1, 1], [0, 1, 0]) is None
    assert d.correlation([1, 2, 3], [0, 0, 0]) is None
    assert d.correlation([1, 2, 3], [0, 0, 1]) > 0
    assert d.correlation([1, 2, 3], [1, 1, 0]) < 0


def test_second_replay_stops_before_creating_new_folder(tmp_path, monkeypatch):
    (tmp_path / "diagnostic-replay-started.json").write_text("{}")
    monkeypatch.setattr(d, "verify", lambda folder: None)
    monkeypatch.setattr(d, "new_folder", lambda: pytest.fail("must not create another run"))
    with pytest.raises(ValueError, match="DIAGNOSTIC_REPLAY_ALREADY_STARTED"):
        d.replay(tmp_path)


def test_replay_cannot_be_replayed(tmp_path, monkeypatch):
    (tmp_path / "diagnostic-replay-origin.json").write_text("{}")
    monkeypatch.setattr(d, "verify", lambda folder: None)
    with pytest.raises(ValueError, match="DIAGNOSTIC_REPLAY_OF_REPLAY"):
        d.replay(tmp_path)
