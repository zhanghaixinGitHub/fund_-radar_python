"""实际拟合的输入隔离、可读产物、有限分支和进程回收。"""

import json
import math
import sys
from copy import deepcopy
from datetime import date, timedelta

import pytest
from app.services.direction_training_models import execute_job, prepared, recent_lower
from app.services.direction_training_process import run_process
from app.services.direction_training_protocol import TREE
from app.services.direction_training_runner import jobs_for_window
from app.services.trading_calendar import load_calendar


def rows(start, end, count, *, reverse=False):
    dates = [d for d in load_calendar().sessions if start <= d <= end][:count]
    result = []
    for f in ("001632", "006730", "008888"):
        for i, d in enumerate(dates):
            signal = math.sin(i / 7)
            y = int(signal < 0 if reverse else signal >= 0)
            item = {
                "fund": f,
                "cutoff": str(d),
                "anchor": str(d),
                "available_at": str(d),
                "x": [signal, math.cos(i / 7), i / 100, 0.2, 0.1, 0.5, 0],
                "input_hash": f.zfill(64),
            }
            answer = {
                "fund": f,
                "cutoff": str(d),
                "end": str(d + timedelta(days=30)),
                "available_at": str(d + timedelta(days=31)),
                "y": y,
                "future_return": "0.1" if y else "-0.1",
            }
            result.append({"input": item, "answer": answer})
    return result


@pytest.fixture(scope="module")
def data():
    return {
        "window": {"name": "FIXTURE", "fit_end": "2022-12-31", "cal_end": "2023-06-30", "exam_end": "2023-09-30"},
        "FIT": rows(date(2021, 7, 1), date(2022, 11, 1), 270),
        "CAL": rows(date(2023, 1, 3), date(2023, 5, 15), 70, reverse=True),
        "EXAM": [r["input"] for r in rows(date(2023, 7, 3), date(2023, 8, 1), 10)],
    }


def test_job_routes_do_not_give_calibration_answers_to_base_training(data):
    jobs = jobs_for_window(data)
    assert all(not jobs[b]["cal"] for b in ("A", "B", "C"))
    assert jobs["A_CAL"]["cal"] == data["CAL"]
    assert all("answer" not in r for job in jobs.values() for r in job["exam"])
    wrong = deepcopy(jobs["A"])
    wrong["exam"][0]["answer"] = {"y": 1}
    with pytest.raises(ValueError):
        execute_job(wrong)


def test_late_training_answer_and_wrong_cohort_are_rejected(data):
    wrong = deepcopy(data["FIT"])
    wrong[0]["answer"]["available_at"] = "2023-01-01"
    with pytest.raises(ValueError, match="BOUNDARY"):
        prepared(wrong, date(2020, 12, 31), date(2022, 12, 31))
    with pytest.raises(ValueError, match="COHORT"):
        prepared(data["FIT"][:270], date(2020, 12, 31), date(2022, 12, 31))


def test_a_fit_is_unchanged_by_exam_values_and_calibration_reuses_exact_base(data):
    jobs = jobs_for_window(data)
    a = execute_job(jobs["A"])
    changed = deepcopy(jobs["A"])
    changed["exam"][0]["x"] = [1000] * 7
    second = execute_job(changed)
    assert a["artifact"] == second["artifact"]
    expected_mean = sum(r["input"]["x"][0] for r in data["FIT"]) / len(data["FIT"])
    assert a["artifact"]["mean"][0] == pytest.approx(expected_mean, abs=1e-12)
    jobs["A_CAL"]["base"] = json.dumps(a["artifact"])
    calibrated = execute_job(jobs["A_CAL"])
    assert calibrated["artifact"]["base_model"] == a["artifact"]
    assert calibrated["artifact"]["calibrator"]["slope"] < 0
    assert calibrated["status"] == "REVERSED_PENDING_VALIDATION"
    assert calibrated["scores"]["A_CAL"] != a["scores"]["A"]


def test_b_same_history_is_not_an_extra_fit_and_tree_configuration_is_fixed(data):
    jobs = jobs_for_window(data)
    assert execute_job(jobs["B"])["status"] == "NOT_DISTINCT"
    tree = execute_job(jobs["C"])
    assert tree["artifact"]["parameters"] == TREE
    assert tree["artifact"]["parameters"]["early_stopping"] is False
    assert tree["artifact"]["fit_count"] == len(data["FIT"])
    assert recent_lower(date(2024, 8, 31)) == date(2023, 2, 28)


def test_worker_timeout_terminates_process():
    result = run_process(
        {}, seconds=0.1, command=[sys.executable, "-c", "import sys,time;sys.stdin.read();time.sleep(30)"]
    )
    assert result["status"] == "FAILED" and result["reason"] == "CANDIDATE_TIMEOUT"
    assert result["elapsed_seconds"] < 5


def test_actual_worker_runs_with_memory_limit(data):
    job = jobs_for_window(data)["BASELINES"]
    result = run_process(job)
    assert result["status"] == "PREDICTED"
    assert len(result["scores"]["ALWAYS_UP"]) == len(data["EXAM"])
