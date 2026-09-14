"""固定120日统计、运行槽及截止收尾；合成数据验证，不生成真实历史回填。"""

from copy import deepcopy
from datetime import timedelta

import pytest
from app.services import direction_1d_independent as p
from app.services import direction_1d_independent_audit as audit
from app.services import direction_1d_independent_report as report
from app.services import direction_1d_independent_runtime as runtime
from app.services.direction_1d_protocol import calendar


@pytest.fixture
def cohort():
    sessions = [str(d) for d in calendar()[0]]
    schedule = p.schedule(sessions, "2026-01-05")
    members = {str(i): {"fund_code": str(i), "family": str(i)} for i in range(3)}
    contract = {"sessions": sessions, "schedule": schedule, "members": members, "coverage": list(members.values())}
    rows = []
    for i, day in enumerate(schedule["targets"]):
        for code in members:
            y = i % 2
            fixed = 1 - y if i % 4 in (0, 1) else y
            rows.append(
                {
                    "fund_code": code,
                    "family": code,
                    "u": day,
                    "t": p.bounds(sessions, day)[0],
                    "y": y,
                    "scores": {"ACTIVITY12": 0.8 if y else 0.2, "FIXED7": 0.6 if fixed else 0.4},
                    "directions": {
                        "ACTIVITY12": y,
                        "FIXED7": fixed,
                        "ALWAYS_UP": 1,
                        "ALWAYS_NON_UP": 0,
                        "INITIAL_MAJORITY": 0,
                        "MOMENTUM": 1 - y,
                    },
                }
            )
    at = p.instant(schedule["grace_targets"][-1] + "T23:59:59+08:00") + timedelta(seconds=1)
    return contract, rows, at


def test_complete_synthetic_stable_improvement_requires_every_gate(cohort):
    contract, rows, at = cohort
    result = report.evaluate(contract, rows, at)
    assert result["verdict"] == "STABLE_GAIN_EVIDENCE"
    assert all(result["checks"].values())
    assert result["coverage"]["planned_total"] == 360
    assert result["common_count"] == 360
    assert result["paired_interval_95"][0] > 0
    assert result["weighted_accuracy_gain"] == pytest.approx(0.5)
    assert result["model_released"] is False


def test_missing_predictions_stay_in_expected_denominator(cohort):
    contract, rows, at = cohort
    for row in rows[:40]:
        row["directions"] = {}
        row["scores"] = {}
    result = report.evaluate(contract, rows, at)
    assert result["coverage"]["due_total"] == 360
    assert result["coverage"]["common"] == pytest.approx(320 / 360)
    assert result["coverage"]["mature"] == 1
    assert result["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_unknown_results_never_become_wrong_answers(cohort):
    contract, rows, at = cohort
    for row in rows[:40]:
        row["y"] = None
    result = report.evaluate(contract, rows, at)
    assert result["coverage"]["unknown_answers"] == 40
    assert result["metrics"]["ACTIVITY12"]["correct"] == 320
    assert result["metrics"]["ACTIVITY12"]["count"] == 320
    assert result["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_interim_report_never_selects_a_model(cohort):
    contract, rows, _ = cohort
    at = p.instant(contract["schedule"]["targets"][59] + "T22:00:00+08:00")
    result = report.evaluate(contract, rows, at)
    assert result["due_target_days"] == 60
    assert not result["stage_finished"]
    assert result["verdict"] == "PENDING_FINAL_REVIEW"


def test_one_direction_does_not_prove_both_regimes(cohort):
    contract, rows, at = cohort
    for row in rows:
        row["y"] = 1
        row["directions"]["ACTIVITY12"] = 1
    result = report.evaluate(contract, rows, at)
    assert not result["checks"]["direction_dates_sufficient"]
    assert not result["checks"]["beats_every_simple_baseline"]
    assert result["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_worse_fixed_candidate_ends_without_new_search(cohort):
    contract, rows, at = cohort
    for row in rows:
        row["directions"]["ACTIVITY12"] = 1 - row["y"]
        row["directions"]["FIXED7"] = row["y"]
    result = report.evaluate(contract, rows, at)
    assert result["verdict"] == "NO_IMPROVEMENT"
    assert result["stage_finished"] and result["new_fits"] == 0


def test_insufficient_fund_is_not_removed_from_required_fraction(cohort):
    contract, rows, at = cohort
    rows = [r for r in rows if r["fund_code"] != "0"]
    result = report.evaluate(contract, rows, at)
    assert set(result["funds"]) == {"0", "1", "2"}
    assert result["funds"]["0"]["status"] == "INSUFFICIENT"
    assert result["coverage"]["planned_total"] == 360


def test_fund_share_duplicates_keep_original_family_weight(cohort):
    _, rows, _ = cohort
    simple = [r for r in rows if r["fund_code"] in ("0", "1")]
    expected = report.metric(simple, "ACTIVITY12")
    duplicates = [deepcopy(r) for r in simple if r["fund_code"] == "0"]
    for row in duplicates:
        row["fund_code"] = "share-c"
    actual = report.metric([*simple, *duplicates], "ACTIVITY12")
    assert actual["weighted_accuracy"] == expected["weighted_accuracy"]
    assert actual["count"] > expected["count"]


def test_bootstrap_reproducible_and_same_date_rows_not_independent(cohort):
    contract, rows, _ = cohort
    first = report.paired_interval(rows, contract["schedule"]["targets"])
    assert first == report.paired_interval(rows, contract["schedule"]["targets"])
    assert first == report.paired_interval([r for r in rows if r["fund_code"] == "0"], contract["schedule"]["targets"])


def test_out_of_scope_duplicate_questions_rejected(cohort):
    contract, rows, at = cohort
    with pytest.raises(ValueError, match="QUESTION_SCOPE_INVALID"):
        report.evaluate(contract, [*rows, rows[0]], at)
    rows[0]["fund_code"] = "stranger"
    with pytest.raises(ValueError, match="QUESTION_SCOPE_INVALID"):
        report.evaluate(contract, rows, at)


def test_clock_slots_never_catch_up_a_missed_deadline():
    sessions = [str(d) for d in calendar()[0]]
    contract = {"sessions": sessions, "schedule": {"targets": ["2026-09-14"]}}
    assert runtime.slot_for(contract, p.instant("2026-09-14T07:31:00+08:00")) == ("2026-09-14", "07:30")
    for stamp in ("2026-09-14T07:36:00+08:00", "2026-09-14T08:30:00+08:00", "2026-09-13T07:30:00+08:00"):
        assert runtime.slot_for(contract, p.instant(stamp)) is None


def test_finalization_does_not_depend_on_database_being_online(cohort, tmp_path, monkeypatch):
    contract, _, at = cohort
    monkeypatch.setattr(runtime, "load", lambda root: (contract, {}))
    monkeypatch.setattr(p, "now", lambda: at)

    def unavailable(*args):
        raise AssertionError("must not query database after fixed final deadline")

    monkeypatch.setattr(runtime, "fresh_scope", unavailable)
    result = runtime.tick(tmp_path)
    assert result == {"status": "STAGE_FINISHED", "verdict": "INSUFFICIENT_EVIDENCE"}
    before = (tmp_path / "final-report.json").read_bytes()
    assert runtime.tick(tmp_path) == result
    assert before == (tmp_path / "final-report.json").read_bytes()


def test_incomplete_calendar_blocks_bulk_input_collection_before_network(tmp_path, monkeypatch):
    monkeypatch.setattr(
        audit, "verify", lambda root: {"draft": {"blockers": ["CALENDAR_MISSING"], "schedule": {"complete": False}}}
    )
    with pytest.raises(ValueError, match="START_BLOCKED"):
        runtime.input_acceptance(tmp_path)
    assert not list(tmp_path.iterdir())


def test_failures_keep_stack_but_not_sensitive_message():
    try:
        raise RuntimeError("password=do-not-emit token=not-for-output")
    except RuntimeError as exc:
        result = runtime.failure(exc)
    assert result["code"] == "RuntimeError"
    assert result["frames"]
    assert "password" not in p.canonical(result)
