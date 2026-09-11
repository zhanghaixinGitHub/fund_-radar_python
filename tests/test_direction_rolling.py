"""滚动更新的标签成熟、历史选择、模型复用与旧算法数值一致性。"""

import sys
from copy import deepcopy
from datetime import date

import pytest
from app.services import direction_rolling_models as m
from app.services import direction_rolling_runner as r
from app.services.direction_linear_models import execute_job as execute_original
from app.services.direction_linear_models import restore as restore_original
from app.services.direction_linear_models import validate_job as validate_original
from app.services.direction_linear_protocol import COVERAGE_VERSION, specification_for, study_windows
from app.services.direction_training_artifacts import digest
from app.services.direction_training_process import run_process
from app.services.trading_calendar import load_calendar


def input_row(fund, day, i=0):
    calendar = load_calendar()
    return {
        "fund": fund,
        "cutoff": str(day),
        "anchor": str(calendar.sessions[calendar.at_or_before_index(day) - 1]),
        "available_at": str(day),
        "x": [(i * j + 3) % 19 / 19 for j in range(1, 8)],
        "input_hash": "a" * 64,
    }


def labeled_row(fund, day, i=0):
    future = load_calendar().future_sessions(day, 21)
    return {
        "input": input_row(fund, day, i),
        "answer": {
            "fund": fund,
            "cutoff": str(day),
            "end": str(future[-2]),
            "available_at": str(future[-1]),
            "y": i % 2,
            "future_return": "0.01" if i % 2 else "-0.01",
        },
    }


@pytest.fixture
def payload():
    days = [d for d in load_calendar().sessions if d >= date(2021, 4, 7)][:270]
    return {
        "version": m.VERSION,
        "month": "2023-01",
        "recipe": "ALL",
        "fit": [labeled_row(f, d, i) for f in r.FUNDS for i, d in enumerate(days)],
        "exam": [input_row(f, date(2023, 1, 3)) for f in r.FUNDS],
    }


@pytest.mark.parametrize(
    "branch,cutoff,expected",
    [
        ("QUARTER_ALL", "2023-03-30", "2023-01_ALL"),
        ("MONTH_ALL", "2023-03-30", "2023-03_ALL"),
        ("QUARTER_252", "2024-12-02", "2024-10_252"),
        ("MONTH_252", "2024-12-02", "2024-12_252"),
        ("QUARTER_252", "2024-10-08", "2024-10_252"),
    ],
)
def test_routing_and_first_month_reuse(branch, cutoff, expected):
    assert m.model_key(branch, cutoff) == expected
    assert m.fit_end("2024-03") == date(2024, 2, 29)
    assert m.fit_end("2023-01") == date(2022, 12, 31)


def test_pool_deduplicates_identical_rows_and_rejects_conflicts(payload):
    row = payload["fit"][0]
    assert r.merge_rows([row, row]) == [row]
    other = deepcopy(row)
    other["input"]["x"][0] += 1
    with pytest.raises(ValueError, match="ROLLING_POOL_CONFLICT"):
        r.merge_rows([row, other])
    other = deepcopy(row)
    other["answer"]["fund"] = r.FUNDS[1]
    with pytest.raises(ValueError, match="ROLLING_POOL_IDENTITY"):
        r.merge_rows([other])


def test_mature_earlier_exam_enters_only_later_training(payload):
    early = [labeled_row(f, date(2023, 1, 3)) for f in r.FUNDS]
    pool = r.merge_rows([*payload["fit"], *early])
    assert r.select_fit(pool, "2023-01", "ALL") == payload["fit"]
    assert r.select_fit(pool, "2023-02", "ALL") == payload["fit"]
    later = r.select_fit(pool, "2023-03", "ALL")
    assert all(row in later for row in early)


def test_recent_selection_is_latest_shared_dates_and_ignores_direction(payload):
    selected = r.select_fit(payload["fit"], "2023-01", "252")
    assert len(selected) == 756
    for fund in r.FUNDS:
        assert [x for x in selected if x["input"]["fund"] == fund] == [
            x for x in payload["fit"] if x["input"]["fund"] == fund
        ][-252:]
    altered = deepcopy(payload["fit"])
    for row in altered:
        row["answer"]["y"] = 1 - row["answer"]["y"]
    assert [x["input"] for x in r.select_fit(altered, "2023-01", "252")] == [x["input"] for x in selected]
    with pytest.raises(ValueError, match="ROLLING_COMMON_FIT_DATES"):
        r.select_fit(payload["fit"][:-1], "2023-01", "ALL")


@pytest.mark.parametrize(
    "mutation",
    [
        "future_label",
        "publication",
        "horizon",
        "exam_label",
        "exam_month",
        "protected",
        "anchor",
        "duplicate",
        "unbalanced",
        "recent_count",
        "extra_field",
        "unknown_month",
    ],
)
def test_worker_rejects_leakage_and_unplanned_input(payload, mutation):
    if mutation == "future_label":
        payload["fit"][0]["answer"]["available_at"] = "2023-01-03"
    elif mutation == "publication":
        payload["fit"][0]["answer"]["available_at"] = payload["fit"][0]["answer"]["end"]
    elif mutation == "horizon":
        payload["fit"][0]["answer"]["end"] = payload["fit"][0]["answer"]["available_at"]
    elif mutation == "exam_label":
        payload["exam"][0]["y"] = 1
    elif mutation == "exam_month":
        payload["exam"][0] = input_row(r.FUNDS[0], date(2023, 2, 1))
    elif mutation == "protected":
        payload["month"] = "2024-12"
        payload["exam"] = [input_row(f, date(2024, 12, 31)) for f in r.FUNDS]
    elif mutation == "anchor":
        payload["exam"][0]["anchor"] = payload["exam"][0]["cutoff"]
    elif mutation == "duplicate":
        payload["fit"][1] = payload["fit"][0]
    elif mutation == "unbalanced":
        payload["fit"].pop()
    elif mutation == "recent_count":
        payload["recipe"] = "252"
    elif mutation == "extra_field":
        payload["future_answers"] = []
    else:
        payload["month"] = "2025-01"
    with pytest.raises(ValueError):
        m.validate_job(payload)


def test_real_worker_and_original_algorithm_have_identical_numbers(payload):
    output = run_process(payload, command=[sys.executable, "-m", "scripts.direction_rolling_worker"])
    assert output["status"] == "PREDICTED"
    original_job = {
        "version": COVERAGE_VERSION,
        "branch": "REFERENCE",
        "window": study_windows(COVERAGE_VERSION)[2],
        "fit": payload["fit"],
        "exam": [input_row(f, date(2023, 7, 3)) for f in r.FUNDS],
    }
    old = execute_original(original_job)["models"]["POOLED"]
    for field in r.NUMERIC_FIELDS:
        assert output["model"][field] == old[field]
    direct = m.execute_job(payload)
    assert output["model"] == direct["model"] and output["scores"] == direct["scores"]
    assert output["model"]["train_hash"] == digest(payload["fit"])
    broken = deepcopy(output["model"])
    broken["intercept"] += 1
    with pytest.raises(ValueError, match="ROLLING_MODEL_HASH"):
        m.restore(broken)
    broken["hash"] = digest({k: v for k, v in broken.items() if k != "hash"})
    broken["fit_end"] = "2023-01-03"
    with pytest.raises(ValueError, match="ROLLING_MODEL_PROTOCOL"):
        m.restore(broken)


def test_legacy_entrypoints_reject_new_time_contract(payload):
    with pytest.raises(ValueError, match="ROLLING_REQUIRES_DEDICATED_WORKER"):
        validate_original(payload)
    with pytest.raises(ValueError, match="ROLLING_REQUIRES_DEDICATED_RESTORE"):
        restore_original({"version": m.VERSION})
    with pytest.raises(ValueError):
        specification_for(m.VERSION)


@pytest.mark.parametrize("marker", ["rolling-replay-started.json", "rolling-replay-origin.json"])
def test_only_one_replay_and_no_replay_of_replay(tmp_path, monkeypatch, marker):
    (tmp_path / marker).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(r, "new_folder", lambda: pytest.fail("must not create another run"))
    with pytest.raises(ValueError, match="ROLLING_REPLAY_"):
        r.replay(tmp_path)
