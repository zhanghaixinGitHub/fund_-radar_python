"""独立预算和门槛的无拟合测试；真实分类器由受控执行槽位调用。"""

import copy
import json
from pathlib import Path

import pytest
from app.services import fund_002112_peer_experiment as e


@pytest.fixture
def ledger_root(tmp_path):
    e.exclusive(tmp_path / "protocol.json", {"test": "synthetic_no_model_fit"})
    return tmp_path


def fill_ledger(path, count, monkeypatch):
    monkeypatch.setattr(e, "checkpoint", lambda *args: {})
    if count > 6:
        e.exclusive(path / "historical-decision.json", {"passed": True})
    for slot in e.SLOTS[:count]:
        e.reserve(path, slot, "test-permit")


def test_exclusive_cannot_overwrite(tmp_path):
    p = tmp_path / "a.json"
    e.exclusive(p, {"a": 1})
    with pytest.raises(FileExistsError):
        e.exclusive(p, {"a": 2})
    assert e.read(p) == {"a": 1}


def test_once_only_reuses_identical_content(tmp_path):
    p = tmp_path / "a.json"
    e.once(p, {"a": 1})
    e.once(p, {"a": 1})
    with pytest.raises(ValueError, match="IMMUTABLE"):
        e.once(p, {"a": 2})


def test_single_consume_and_no_refund(ledger_root):
    e.reserve(ledger_root, e.SLOTS[0], "token")
    assert e.counts(ledger_root)["consumed_slots"] == 1
    assert e.counts(ledger_root)["actual_new_fits"] == 0
    with pytest.raises(ValueError, match="CONSUMED"):
        e.reserve(ledger_root, e.SLOTS[0], "another-token")


def test_budget_exhaustion(ledger_root, monkeypatch):
    fill_ledger(ledger_root, 12, monkeypatch)
    with pytest.raises(ValueError, match="BUDGET_EXHAUSTED"):
        e.reserve(ledger_root, e.SLOTS[-1], "token")
    assert e.counts(ledger_root)["remaining_slots"] == 0


def test_out_of_order_rejected(ledger_root):
    with pytest.raises(ValueError, match="OUT_OF_ORDER"):
        e.reserve(ledger_root, e.SLOTS[1], "token")


def test_full_requires_historical_pass(ledger_root, monkeypatch):
    fill_ledger(ledger_root, 6, monkeypatch)
    e.exclusive(ledger_root / "historical-decision.json", {"passed": False})
    with pytest.raises(ValueError, match="HISTORICAL_GATE"):
        e.reserve(ledger_root, e.SLOTS[6], "token")
    assert e.ledger(ledger_root) == 6


def test_preceding_checkpoint_required(ledger_root):
    e.reserve(ledger_root, e.SLOTS[0], "token")
    with pytest.raises(FileNotFoundError):
        e.reserve(ledger_root, e.SLOTS[1], "token")


@pytest.mark.parametrize("field,value", [("ordinal", 2), ("previous_sha256", "bad"), ("protocol_sha256", "bad")])
def test_ledger_tampering(ledger_root, field, value):
    e.reserve(ledger_root, e.SLOTS[0], "token")
    p = ledger_root / "attempts" / (e.SLOTS[0] + ".json")
    item = e.read(p)
    item[field] = value
    p.write_text(json.dumps(item), encoding="utf-8")
    with pytest.raises(ValueError, match="LEDGER"):
        e.ledger(ledger_root)


def test_deleted_attempt_with_execution_evidence_rejected(ledger_root):
    e.exclusive(ledger_root / "started" / (e.SLOTS[0] + ".json"), {})
    with pytest.raises(ValueError, match="ORPHAN"):
        e.ledger(ledger_root)


def test_incomplete_consumed_slot_never_retries(ledger_root, monkeypatch):
    e.reserve(ledger_root, e.SLOTS[0], "token")
    monkeypatch.setattr(e, "child", lambda *a: pytest.fail("must not refit"))
    with pytest.raises(ValueError, match="NO_REFIT"):
        e.ensure_slot(ledger_root, e.SLOTS[0])
    assert e.ledger(ledger_root) == 1


def test_failed_slot_never_retries(ledger_root, monkeypatch):
    e.reserve(ledger_root, e.SLOTS[0], "token")
    e.exclusive(ledger_root / "failures" / (e.SLOTS[0] + ".json"), {})
    monkeypatch.setattr(e, "child", lambda *a: pytest.fail("must not refit"))
    with pytest.raises(ValueError, match="CANNOT_RETRY"):
        e.ensure_slot(ledger_root, e.SLOTS[0])


def test_checkpoint_reuse_without_child(ledger_root, monkeypatch):
    slot = e.SLOTS[0]
    e.reserve(ledger_root, slot, "token")
    e.exclusive(ledger_root / "result.json", {"test": True})
    cp = {
        "slot": slot,
        "attempt_sha256": e.file_hash(ledger_root / "attempts" / (slot + ".json")),
        "files": {"result.json": e.file_hash(ledger_root / "result.json")},
    }
    e.exclusive(ledger_root / "checkpoints" / (slot + ".json"), cp)
    monkeypatch.setattr(e, "child", lambda *a: pytest.fail("must not run child"))
    assert e.ensure_slot(ledger_root, slot) == cp
    (ledger_root / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="FROZEN_FILE"):
        e.ensure_slot(ledger_root, slot)


def test_complete_fit_resume_only_restores(ledger_root, monkeypatch):
    slot = e.SLOTS[0]
    e.reserve(ledger_root, slot, "token")
    e.exclusive(ledger_root / "fit-complete" / (slot + ".json"), {"files": {}})
    commands = []

    def child(path, command, selected, token=None):
        commands.append(command)
        e.exclusive(path / "restored" / (selected + ".json"), {"passed": True})

    monkeypatch.setattr(e, "child", child)
    e.ensure_slot(ledger_root, slot)
    e.ensure_slot(ledger_root, slot)
    assert commands == ["_restore"]
    assert e.ledger(ledger_root) == 1


def test_worker_missing_permit_cannot_fit(ledger_root, monkeypatch):
    e.reserve(ledger_root, e.SLOTS[0], "permit")
    monkeypatch.setattr(e, "load_frozen", lambda p: {})
    monkeypatch.setenv("FUND_PEER_FIT_PERMIT", "wrong")
    monkeypatch.setattr(e.model, "fit", lambda *a: pytest.fail("must not fit"))
    with pytest.raises(ValueError, match="PERMIT"):
        e.fit_worker(ledger_root, e.SLOTS[0])


def test_worker_token_is_single_use(ledger_root, monkeypatch):
    slot = e.SLOTS[0]
    e.reserve(ledger_root, slot, "permit")
    e.exclusive(ledger_root / "started" / (slot + ".json"), {})
    monkeypatch.setattr(e, "load_frozen", lambda p: {})
    monkeypatch.setattr(e, "slot_data", lambda *a: ([], [], [], "L20"))
    monkeypatch.setenv("FUND_PEER_FIT_PERMIT", "permit")
    monkeypatch.setattr(e.model, "fit", lambda *a: pytest.fail("must not fit"))
    with pytest.raises(FileExistsError):
        e.fit_worker(ledger_root, slot)


def test_counts_failed_actual_call(ledger_root):
    e.reserve(ledger_root, e.SLOTS[0], "permit")
    e.exclusive(ledger_root / "classifier-started" / (e.SLOTS[0] + ".json"), {})
    result = e.counts(ledger_root)
    assert result["actual_new_fits"] == 1
    assert result["completed_classifier_calls"] == 0
    assert result["cumulative_actual_fits"] == 53


def test_lock_prevents_parallel_runner(tmp_path):
    with e.lock(tmp_path), pytest.raises(ValueError, match="ALREADY_RUNNING"), e.lock(tmp_path):
        pass


@pytest.fixture
def comparison_rows():
    rows = [
        {
            "fund_code": "002112",
            "target": f"2023-{month:02d}-{day:02d}",
            "actual_direction": c,
            "input_hash": f"{month}-{day}",
            "direction": c,
            "scores": [0.2, 0.3, 0.5],
        }
        for month in (4, 7, 10)
        for day, c in enumerate(e.CLASSES, 1)
    ]
    values = {v: copy.deepcopy(rows) for v in ("L20_RECOVERED", "L20_ORIGINAL", "N7_ORIGINAL")}
    for v in ("L20_ORIGINAL", "N7_ORIGINAL"):
        for row in values[v]:
            row["direction"] = "FLAT" if row["actual_direction"] == "UP" else "UP"
    return values


def test_all_gates_can_pass(comparison_rows):
    result = e.compare(comparison_rows, historical=True)
    assert result["numerical_passed"]
    assert result["pairs"]["L20_ORIGINAL"]["gained"] == 9
    assert result["models"]["L20_RECOVERED"]["class_correct"] == {c: 3 for c in e.CLASSES}


def test_equal_total_fails_strict_gate(comparison_rows):
    comparison_rows["L20_ORIGINAL"] = copy.deepcopy(comparison_rows["L20_RECOVERED"])
    assert not e.compare(comparison_rows, historical=True)["checks"]["total_strictly_above_controls"]


def test_constant_baseline_must_be_strictly_beaten(comparison_rows):
    for r in comparison_rows["L20_RECOVERED"]:
        r["direction"] = "DOWN"
    result = e.compare(comparison_rows, historical=True)
    assert not result["checks"]["total_strictly_above_constants"]


@pytest.mark.parametrize("category", e.CLASSES)
def test_each_class_gate_enforced(comparison_rows, category):
    for r in comparison_rows["L20_ORIGINAL"]:
        if r["actual_direction"] == category:
            r["direction"] = category
    for r in comparison_rows["L20_RECOVERED"]:
        if r["actual_direction"] == category:
            r["direction"] = next(c for c in e.CLASSES if c != category)
    assert not e.compare(comparison_rows, historical=True)["checks"]["class_not_worse_" + category]


def test_full_has_no_historical_quarter_gate(comparison_rows):
    assert "at_least_two_quarters_not_worse_L20" not in e.compare(comparison_rows, historical=False)["checks"]


def test_quarter_gate_needs_two(comparison_rows):
    for i, row in enumerate(comparison_rows["L20_RECOVERED"]):
        if i < 6:
            row["direction"] = "FLAT" if row["actual_direction"] == "UP" else "UP"
    comparison_rows["L20_ORIGINAL"] = copy.deepcopy(comparison_rows["N7_ORIGINAL"])
    for row in comparison_rows["L20_ORIGINAL"][:6]:
        row["direction"] = row["actual_direction"]
    assert not e.compare(comparison_rows, historical=True)["checks"]["at_least_two_quarters_not_worse_L20"]


@pytest.mark.parametrize("field", ["target", "input_hash", "actual_direction", "fund_code"])
def test_comparison_identity_mismatch(comparison_rows, field):
    comparison_rows["N7_ORIGINAL"][0][field] = "changed"
    with pytest.raises(ValueError, match="IDENTITY"):
        e.compare(comparison_rows, historical=True)


def test_paired_gained_lost_net(comparison_rows):
    comparison_rows["L20_ORIGINAL"][0]["direction"] = comparison_rows["L20_ORIGINAL"][0]["actual_direction"]
    comparison_rows["L20_RECOVERED"][0]["direction"] = "UP"
    result = e.compare(comparison_rows, historical=True)
    pair = result["pairs"]["L20_ORIGINAL"]
    assert (pair["gained"], pair["lost"], pair["net"]) == (8, 1, 7)
    assert sum(v["net"] for v in pair["quarters"].values()) == 7
    assert sum(v["net"] for v in pair["classes"].values()) == 7


@pytest.mark.parametrize(
    "training,target", [(True, "2024-01-01"), (True, "2025-01-01"), (False, "2025-01-01"), (False, "2026-01-01")]
)
def test_forbidden_label_years(training, target):
    with pytest.raises(ValueError, match="FORBIDDEN_LABEL"):
        e.validate_scope(
            [{"fund_code": "002112", "target": target, "actual_direction": "UP", "x": [1.0] * 20}], training=training
        )


@pytest.mark.parametrize("x", [[0.0] * 19, [float("nan")] * 20, [float("inf")] * 20])
def test_invalid_features_block(x):
    with pytest.raises(ValueError, match="FINITE_COMPLETE"):
        e.validate_scope(
            [{"fund_code": "002112", "target": "2023-01-01", "actual_direction": "UP", "x": x}], training=True
        )


def test_original_full_pool_is_separate():
    controls = e.read(e.PACKAGE / "frozen-controls.json")["controls"]
    old = Path(controls["2023Q2-L20-main"]["path"]).parents[1]
    values = {
        "inputs": e.read(e.PACKAGE / "inputs.json"),
        "folds": e.read(e.PACKAGE / "folds.json"),
        "original-inputs": e.read(old / "inputs.json"),
        "original-folds": e.read(old / "folds.json"),
    }
    recovered, exam, _, v = e.slot_data(values, "FULL-L20_RECOVERED-main")
    original, other_exam, _, ov = e.slot_data(values, "FULL-L20_ORIGINAL-main")
    n7, _, _, nv = e.slot_data(values, "FULL-N7_ORIGINAL-main")
    assert (len(recovered), len(original), len(n7)) == (5137, 4419, 4419)
    assert original == n7 == values["original-inputs"]["train"]
    assert exam == other_exam and len(exam) == 230
    assert (v, ov, nv) == ("L20", "L20", "N7")


def test_historical_failure_stops_before_full(ledger_root, monkeypatch):
    monkeypatch.setattr(e, "load_frozen", lambda p: {})
    monkeypatch.setattr(e, "checkpoint", lambda *a: {})
    completed = []

    def ensure(path, slot):
        e.reserve(path, slot, "token")
        completed.append(slot)

    monkeypatch.setattr(e, "ensure_slot", ensure)
    monkeypatch.setattr(e, "replay_check", lambda *a: {})
    monkeypatch.setattr(e, "stage_result", lambda *a: {"passed": False, "failed_checks": ["performance"]})
    decision = e.run(ledger_root)
    assert decision["status"] == "STOP_HISTORICAL_GATE_FAILED"
    assert completed == list(e.SLOTS[:6])
    assert not decision["full_executed"]
    assert e.run(ledger_root) == decision


def test_technical_failure_stops_and_consumes(ledger_root, monkeypatch):
    monkeypatch.setattr(e, "load_frozen", lambda p: {})

    def ensure(path, slot):
        e.reserve(path, slot, "token")
        raise ValueError("synthetic_fit_failure")

    monkeypatch.setattr(e, "ensure_slot", ensure)
    decision = e.run(ledger_root)
    assert decision["status"] == "STOP_ENGINEERING_FAILURE"
    assert decision["consumed_slots"] == 1
    assert e.run(ledger_root) == decision


def test_full_reached_only_after_pass(ledger_root, monkeypatch):
    monkeypatch.setattr(e, "load_frozen", lambda p: {})
    monkeypatch.setattr(e, "checkpoint", lambda *a: {})
    monkeypatch.setattr(e, "ensure_slot", lambda path, slot: e.reserve(path, slot, "token"))
    monkeypatch.setattr(e, "replay_check", lambda *a: {})
    monkeypatch.setattr(e, "stage_result", lambda *a: {"passed": True, "failed_checks": []})
    decision = e.run(ledger_root)
    assert decision["status"] == "PASSED_ALL_GATES_PENDING_ADOPTION"
    assert decision["consumed_slots"] == 12
    assert not decision["adopted"]


def test_safe_pause_then_resume(ledger_root, monkeypatch):
    monkeypatch.setattr(e, "load_frozen", lambda p: {})
    monkeypatch.setattr(e, "checkpoint", lambda *a: {})
    called = []

    def ensure(path, slot):
        if slot not in called:
            e.reserve(path, slot, "token")
            called.append(slot)

    monkeypatch.setattr(e, "ensure_slot", ensure)
    monkeypatch.setattr(e, "replay_check", lambda *a: {})
    monkeypatch.setattr(e, "stage_result", lambda *a: {"passed": False, "failed_checks": ["performance"]})
    assert e.run(ledger_root, interrupt_after=2)["status"] == "CHECKPOINT_PAUSE"
    assert e.run(ledger_root)["status"] == "STOP_HISTORICAL_GATE_FAILED"
    assert called == list(e.SLOTS[:6])


def test_no_old_runner_imports():
    text = Path(e.__file__).read_text(encoding="utf-8")
    assert "train_registered(" not in text
    assert "import fund_002112_repaired_experiment" not in text
    assert "import fund_002112_round3_run" not in text


@pytest.mark.parametrize("mutation", ["state", "nan", "score", "direction", "identity", "missing_row"])
def test_independent_replay_mismatch_is_rejected(tmp_path, monkeypatch, mutation):
    main, replay = e.SLOTS[:2]
    monkeypatch.setattr(e, "checkpoint", lambda *args: {})
    row = {
        "fund_code": "002112",
        "target": "2023-04-17",
        "actual_direction": "UP",
        "input_hash": "same",
        "direction": "UP",
        "scores": [0.2, 0.3, 0.5],
    }
    for slot in (main, replay):
        e.exclusive(
            tmp_path / "fit-complete" / (slot + ".json"),
            {"state_sha256": "changed" if slot == replay and mutation == "state" else "same"},
        )
        pred = {"train": [copy.deepcopy(row)], "exam": [copy.deepcopy(row)]}
        if slot == replay:
            if mutation == "nan":
                pred["train"][0]["scores"][0] = float("nan")
            elif mutation == "score":
                pred["train"][0]["scores"][0] += 1e-6
            elif mutation == "direction":
                pred["exam"][0]["direction"] = "DOWN"
            elif mutation == "identity":
                pred["exam"][0]["input_hash"] = "different"
            elif mutation == "missing_row":
                pred["train"] = []
        dest = tmp_path / "predictions" / (slot + ".json")
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(json.dumps(pred), encoding="utf-8")
    with pytest.raises(ValueError):
        e.replay_check(tmp_path, main)


def test_recipe_and_dependency_changes_block():
    plan = e.read(e.PACKAGE / "execution-plan.json")
    assert e.environment(plan)["threads"] == 1
    plan["recipe"]["C"] = 2.0
    with pytest.raises(ValueError, match="RECIPE_CHANGED"):
        e.environment(plan)
    plan = e.read(e.PACKAGE / "execution-plan.json")
    plan["inherited_environment"]["libraries"]["numpy"] = "other"
    with pytest.raises(ValueError, match="ENVIRONMENT_CHANGED"):
        e.environment(plan)
