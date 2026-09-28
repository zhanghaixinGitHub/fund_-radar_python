"""修复协议账本及流程验收；假账本只在 tmp_path，实际拟合仅 3 次合成数据。"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from app.services import fund_002112_repaired_experiment as run
from scripts.fund_002112_repaired_audit import recount_gate


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    path = root / ("002112-r3r-" + "a" * 24)
    path.mkdir(parents=True)
    monkeypatch.setattr(run, "ROOT", root)
    run.data.write_once(path / "protocol.json", {"bridge": {}})
    run.data.write_once(path / "bridge-complete.json", {"inherited_fits": 5})
    return path


def seed_attempts(path, count):
    for slot in run.NEW_SLOTS[:count]:
        run.exclusive(path / "attempts" / (slot + ".json"), {"slot": slot})


def test_fixed_inheritance_and_budget():
    assert len(run.INHERITED) == 5
    assert run.INHERITED[-1] == "2023Q2-T20-main"
    assert run.NEW_SLOTS[0] == "2023Q2-T20-replay"
    assert len(run.HISTORICAL_SLOTS) == 13
    assert len(run.NEW_SLOTS) == 19
    assert set(run.INHERITED).isdisjoint(run.NEW_SLOTS)
    assert 34 + len(run.INHERITED) + len(run.NEW_SLOTS) == 58


@pytest.mark.parametrize("count", [0, 1, 7, 13, 19])
def test_inherited_budget_never_resets(sandbox, count):
    seed_attempts(sandbox, count)
    assert run.validate_ledger(sandbox) == count


def test_more_than_nineteen_rejected(sandbox):
    seed_attempts(sandbox, 19)
    run.exclusive(sandbox / "attempts" / "unexpected.json", {})
    with pytest.raises(ValueError, match="BUDGET_EXHAUSTED"):
        run.validate_ledger(sandbox)


@pytest.mark.parametrize("slot", [run.INHERITED[0], run.NEW_SLOTS[1], "made-up-slot"])
def test_consumed_old_slot_and_skipped_order_rejected(sandbox, slot):
    run.exclusive(sandbox / "attempts" / (slot + ".json"), {})
    with pytest.raises(ValueError, match="ORDER_OR_INHERITANCE"):
        run.validate_ledger(sandbox)


def test_another_run_cannot_reopen_budget(sandbox):
    other = run.ROOT / ("002112-r3r-" + "b" * 24)
    run.exclusive(other / "attempts" / (run.NEW_SLOTS[0] + ".json"), {})
    with pytest.raises(ValueError, match="OTHER_RUN"):
        run.validate_ledger(sandbox)


@pytest.mark.parametrize("slot", run.INHERITED)
def test_reserve_rejects_all_five_inherited_slots(sandbox, slot):
    with pytest.raises(ValueError, match="INHERITED_OR_UNKNOWN"):
        run.reserve(sandbox, slot, "test")
    assert not run.attempts(sandbox)


def test_reservation_is_exclusive_and_nonrefundable(sandbox):
    slot = run.NEW_SLOTS[0]
    run.reserve(sandbox, slot, "test-permit")
    with pytest.raises(ValueError, match="NOT_NEXT"):
        run.reserve(sandbox, slot, "test-permit")
    assert len(run.attempts(sandbox)) == 1
    with pytest.raises(ValueError, match="INCOMPLETE_CONSUMED"):
        run.ensure_slot(sandbox, slot)
    assert len(run.attempts(sandbox)) == 1


@pytest.mark.parametrize("previous", [None, False])
def test_q3_cannot_precede_t20_reproduction(sandbox, previous):
    if previous is not None:
        run.data.write_once(sandbox / "reproduction/2023Q2.json", {"passed": previous})
    with pytest.raises((FileNotFoundError, ValueError)):
        run.ready_for_slot(sandbox, "2023Q3-N7-main")


@pytest.mark.parametrize("passed,comparison_passed", [(False, False), (False, True), (True, False)])
def test_full_requires_both_historical_decision_and_comparison(sandbox, passed, comparison_passed):
    run.data.write_once(sandbox / "reproduction/2023Q4.json", {"passed": True})
    comparison = {"numerical_passed": comparison_passed}
    run.data.write_once(sandbox / "historical-comparison.json", comparison)
    run.data.write_once(
        sandbox / "historical-decision.json",
        {
            "passed": passed,
            "comparison_hash": run.data.digest(comparison),
        },
    )
    with pytest.raises(ValueError, match="HISTORICAL_GATE"):
        run.ready_for_slot(sandbox, "FULL-N7-main")


def test_full_rejects_rewritten_comparison(sandbox):
    run.data.write_once(sandbox / "reproduction/2023Q4.json", {"passed": True})
    run.data.write_once(sandbox / "historical-comparison.json", {"numerical_passed": True})
    run.data.write_once(sandbox / "historical-decision.json", {"passed": True, "comparison_hash": "changed"})
    with pytest.raises(ValueError, match="HISTORICAL_GATE"):
        run.ready_for_slot(sandbox, "FULL-N7-main")


def test_full_gate_accepts_only_matching_passed_evidence(sandbox):
    run.data.write_once(sandbox / "reproduction/2023Q4.json", {"passed": True})
    comparison = {"numerical_passed": True}
    run.data.write_once(sandbox / "historical-comparison.json", comparison)
    run.data.write_once(
        sandbox / "historical-decision.json",
        {
            "passed": True,
            "comparison_hash": run.data.digest(comparison),
        },
    )
    run.ready_for_slot(sandbox, "FULL-N7-main")


def test_exclusive_event_does_not_reuse_identical_payload(sandbox):
    path = sandbox / "started/a.json"
    run.exclusive(path, {"slot": "a"})
    with pytest.raises(FileExistsError):
        run.exclusive(path, {"slot": "a"})


def test_incomplete_slot_stops_run_without_fitting(sandbox, monkeypatch):
    seed_attempts(sandbox, 1)
    monkeypatch.setattr(run, "load_frozen", lambda *args, **kwargs: {})
    monkeypatch.setattr(run, "ensure_slot", lambda *args: pytest.fail("INCOMPLETE_SLOT_MUST_NOT_RETRY"))
    with pytest.raises(ValueError, match="INCOMPLETE_CONSUMED"):
        run.run(sandbox)
    assert run.data.read(sandbox / "technical-stop.json")["new_reserved_fits"] == 1
    with pytest.raises(ValueError, match="TERMINAL_TECHNICAL"):
        run.run(sandbox)


def test_file_change_and_path_escape_rejected(sandbox):
    local = sandbox / "evidence.json"
    local.write_text("first")
    files = {"evidence.json": run.data.file_hash(local)}
    run.verify_files(files, sandbox)
    local.write_text("second")
    with pytest.raises(ValueError, match="EVIDENCE_CHANGED"):
        run.verify_files(files, sandbox)
    with pytest.raises(ValueError, match="PATH_ESCAPE"):
        run.verify_files({"../outside": "x"}, sandbox)


def test_checkpoint_reuse_has_no_child_or_reservation(sandbox, monkeypatch):
    slot = run.NEW_SLOTS[0]
    run.exclusive(sandbox / "checkpoints" / (slot + ".json"), {"test": True})
    monkeypatch.setattr(run, "checkpoint", lambda *args: {"reused": True})
    monkeypatch.setattr(run, "reserve", lambda *args: pytest.fail("REUSE_MUST_NOT_RESERVE"))
    monkeypatch.setattr(run, "child", lambda *args: pytest.fail("REUSE_MUST_NOT_FIT"))
    assert run.ensure_slot(sandbox, slot) == {"reused": True}


def test_failed_child_keeps_consumed_slot(sandbox, monkeypatch):
    def broken(*args):
        raise RuntimeError("SIMULATED_SAVE_FAILURE")

    monkeypatch.setattr(run, "child", broken)
    with pytest.raises(RuntimeError, match="SAVE_FAILURE"):
        run.ensure_slot(sandbox, run.NEW_SLOTS[0])
    assert len(run.attempts(sandbox)) == 1
    assert (sandbox / "failures" / (run.NEW_SLOTS[0] + ".json")).exists()
    with pytest.raises(ValueError, match="INCOMPLETE_CONSUMED"):
        run.ensure_slot(sandbox, run.NEW_SLOTS[0])


@pytest.mark.parametrize("passed,expected", [(False, 13), (True, 19)])
def test_whole_control_flow_and_repeat_use_only_remaining_slots(sandbox, monkeypatch, passed, expected):
    """用无拟合替身跑完整两分支，确认历史失败不会创建 FULL，重复执行不占预算。"""
    calls = []
    monkeypatch.setattr(run, "load_frozen", lambda *args, **kwargs: {})
    monkeypatch.setattr(run, "checkpoint", lambda *args: {})

    def ensure(path, slot):
        assert slot not in calls
        calls.append(slot)
        run.exclusive(path / "attempts" / (slot + ".json"), {})
        run.exclusive(path / "checkpoints" / (slot + ".json"), {})

    comparison = {"numerical_passed": passed, "checks": {"synthetic_gate": passed}}
    monkeypatch.setattr(run, "ensure_slot", ensure)
    monkeypatch.setattr(run, "stage_result", lambda *args: {})
    monkeypatch.setattr(run, "historical_result", lambda *args: comparison)
    monkeypatch.setattr(run.model, "comparison", lambda *args, **kwargs: comparison)
    run.run(sandbox)
    assert calls == list(run.NEW_SLOTS[:expected])
    assert run.data.read(sandbox / "historical-decision.json")["passed"] is passed
    assert run.data.read(sandbox / "completion.json")["new_real_fits"] == expected
    run.run(sandbox)
    assert len(calls) == expected


def test_concurrent_processes_cannot_consume_same_slot(tmp_path):
    script = """
import sys,time
from pathlib import Path
from app.services.fund_002112_repaired_experiment import lock,exclusive
root=Path(sys.argv[1])
try:
    with lock(root):
        exclusive(root/'one-slot.json',{'reserved':True})
        time.sleep(0.2)
    print('RESERVED')
except OSError:
    print('REJECTED')
"""
    command = [sys.executable, "-X", "utf8", "-B", "-c", script, str(tmp_path)]
    processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    results = [process.communicate(timeout=30) for process in processes]
    assert [process.returncode for process in processes] == [0, 0], results
    assert sorted(stdout.strip() for stdout, _ in results) == ["REJECTED", "RESERVED"]
    assert run.data.read(tmp_path / "one-slot.json") == {"reserved": True}


def _event(value):
    destination = os.environ.get("FUND_002112_SYNTHETIC_EVIDENCE")
    if destination:
        with Path(destination).open("a", encoding="utf-8") as output:
            output.write(json.dumps({"synthetic_only": True, **value}) + "\n")


@pytest.mark.parametrize("variant", ["N7", "L20", "T20"])
def test_real_worker_pipeline_uses_only_synthetic_rows(sandbox, monkeypatch, variant):
    """真实执行新 worker 的计数、保存和加载链路；每配置仅一次随机合成拟合。"""
    rng = np.random.default_rng(20260927)
    rows = [
        {
            "fund_code": "SYNTHETIC",
            "target": f"synthetic-{i:04d}",
            "actual_direction": run.data.CLASSES[i % 3],
            "x": rng.normal(size=20).tolist(),
        }
        for i in range(390)
    ]
    slot = f"2023Q3-{variant}-main"
    index = run.NEW_SLOTS.index(slot)
    monkeypatch.setattr(run, "load_frozen", lambda *args, **kwargs: {"eligibility": {"passed": True}})
    monkeypatch.setattr(run, "ready_for_slot", lambda *args: None)
    monkeypatch.setattr(run, "validate_ledger", lambda *args: index + 1)
    monkeypatch.setattr(run, "checkpoint", lambda *args: {})
    monkeypatch.setattr(run.original, "fold_data", lambda *args: (rows[:360], rows[360:], [1.0] * 360))
    token = "synthetic-test-only"
    monkeypatch.setenv("FUND_002112_FIT_PERMIT", token)
    run.exclusive(
        sandbox / "attempts" / (slot + ".json"),
        {
            "slot": slot,
            "permit_sha256": run.hashlib.sha256(token.encode()).hexdigest(),
            "protocol_hash": run.data.digest(run.data.read(sandbox / "protocol.json")),
        },
    )
    _event({"event": "before_classifier_fit", "variant": variant, "role": "synthetic_new_worker"})
    run.fit_worker(sandbox, slot)
    _event({"event": "completed_classifier_fit", "variant": variant, "role": "synthetic_new_worker"})
    run.restore_worker(sandbox, slot)
    evidence = run.data.read(sandbox / "restored" / (slot + ".json"))
    assert evidence["all_rows"] == {
        "train": {"rows": 360, "max_score_delta": 0.0},
        "exam": {"rows": 30, "max_score_delta": 0.0},
    }
    assert len(list((sandbox / "classifier-started").glob("*.json"))) == 1
    assert len(list((sandbox / "classifier-completed").glob("*.json"))) == 1
    with pytest.raises(FileExistsError):
        run.fit_worker(sandbox, slot)
    assert len(list((sandbox / "classifier-started").glob("*.json"))) == 1


def test_fit_worker_cannot_run_without_parent_permit(sandbox, monkeypatch):
    slot = run.NEW_SLOTS[0]
    run.reserve(sandbox, slot, "required")
    monkeypatch.setattr(run, "load_frozen", lambda *args: {})
    monkeypatch.delenv("FUND_002112_FIT_PERMIT", raising=False)
    with pytest.raises(ValueError, match="PERMIT_MISSING"):
        run.fit_worker(sandbox, slot)
    assert not (sandbox / "started").exists()


@pytest.mark.parametrize("changed", ["code", "environment", "input", "anchor"])
def test_frozen_identity_changes_are_rejected_before_fit(sandbox, monkeypatch, changed):
    """为冻结入口构造最小载荷，在读取任何真实包之前拒绝变更。"""
    old_spec = {"code": {"test": "original"}}
    spec = {
        "protocol": run.PROTOCOL,
        "schema": run.state.SCHEMA,
        "slots": list(run.NEW_SLOTS),
        "inherited_fits": 5,
        "maximum_new_fits": 19,
        "code": {"test": "original"},
        "environment": {"test": "original"},
        "anchors": {},
        "protected_files": {},
        "old_protocol_hash": run.data.digest(old_spec),
        "frozen_data": {},
        "bridge": {},
    }
    # 不覆盖 fixture 的 protocol；用独立测试目录表达不同冻结证据。
    path = sandbox.parent / ("002112-r3r-" + "c" * 24)
    if changed in ("code", "environment"):
        spec[changed] = {"test": "changed"}
    elif changed == "anchor":
        spec["anchors"] = {str(sandbox / "missing-anchor"): "x"}
    else:
        spec["frozen_data"] = {"inputs": run.data.digest({"test": "original"})}
    run.data.write_once(path / "protocol.json", spec)
    if changed == "input":
        run.data.write_once(path / "inputs.json", {"test": "changed"})
    run.data.write_once(run.ROOT / "run-binding.json", {"run_id": path.name, "protocol_hash": run.data.digest(spec)})
    monkeypatch.setattr(run, "code_manifest", lambda: {"test": "original"})
    monkeypatch.setattr(run.original, "environment", lambda: {"test": "original"})
    monkeypatch.setattr(run.original, "code_manifest", lambda: {"test": "original"})
    old = sandbox / "fake-old"
    monkeypatch.setattr(run, "OLD", old)
    run.data.write_once(old / "protocol.json", old_spec)
    expected = {
        "code": "CODE_ENVIRONMENT",
        "environment": "CODE_ENVIRONMENT",
        "input": "FROZEN_REPAIR_DATA_CHANGED",
        "anchor": "EVIDENCE_CHANGED",
    }
    with pytest.raises(ValueError, match=expected[changed]):
        run.load_frozen(path)
    assert not run.attempts(path)


def test_numeric_recount_does_not_trust_pass_flag():
    rows = [
        {
            "target": f"2023-0{quarter * 3:01d}-01",
            "actual_direction": "UP",
            "direction": "UP",
            "input_hash": "synthetic",
            "scores": [0.0, 0.0, 1.0],
        }
        for quarter in (1, 2, 3)
    ]
    comparison = run.model.comparison({v: rows for v in run.model.VARIANTS}, historical=True)
    assert not comparison["numerical_passed"]
    recount_gate(comparison, historical=True)
    comparison["numerical_passed"] = True
    with pytest.raises(AssertionError):
        recount_gate(comparison, historical=True)


@pytest.mark.parametrize("identifier", ["../old", "002112-r3-" + "a" * 24, "002112-r3r-invalid"])
def test_no_arbitrary_run_directory(identifier):
    with pytest.raises(ValueError, match="INVALID_REPAIR_RUN"):
        run.directory(identifier)


@pytest.mark.parametrize("variant", ["N7", "L20", "T20"])
def test_reproduction_mismatch_never_grants_next_stage(sandbox, monkeypatch, variant):
    monkeypatch.setattr(run, "checkpoint", lambda *args: {})
    for current in run.model.VARIANTS:
        for role in ("main", "replay"):
            slot = f"2023Q2-{current}-{role}"
            run.data.write_once(sandbox / "restored" / (slot + ".json"), {"separate_process": True})
            changed = current == variant and role == "replay"
            run.data.write_once(
                sandbox / "models" / (slot + ".manifest.json"),
                {
                    "state_sha256": "changed" if changed else "same",
                },
            )
            row = {
                "fund_code": "SYNTHETIC",
                "target": "2023-04-03",
                "actual_direction": "UP",
                "input_hash": "synthetic",
                "direction": "UP",
                "scores": [0.0, 0.0, 1.0],
            }
            run.data.write_once(sandbox / "predictions" / (slot + ".json"), {"train": [row], "exam": [row]})
    with pytest.raises(ValueError, match="INDEPENDENT_NUMERIC_STATE_MISMATCH"):
        run.stage_result(sandbox, "2023Q2")
    assert not (sandbox / "reproduction/2023Q2.json").exists()
    with pytest.raises(FileNotFoundError):
        run.ready_for_slot(sandbox, "2023Q3-N7-main")


def test_preflight_binds_once_and_inherits_five_without_fitting(tmp_path, monkeypatch):
    """模型为显式替身字节，只验证初始化、继承映射和幂等绑定，绝不反序列化。"""
    root, old, repair = (tmp_path / name for name in ("new", "old", "repair"))
    monkeypatch.setattr(run, "ROOT", root)
    monkeypatch.setattr(run, "OLD", old)
    monkeypatch.setattr(run, "REPAIR", repair)
    monkeypatch.setattr(run, "protection", lambda: {})
    values = {"inputs": {"synthetic": True}}
    monkeypatch.setattr(run, "preflight_inputs", lambda: values)
    monkeypatch.setattr(run, "load_frozen", lambda *args, **kwargs: values)
    monkeypatch.setattr(run, "code_manifest", lambda: {})
    monkeypatch.setattr(run.original, "environment", lambda: {})
    monkeypatch.setattr(run.original, "checkpoint", lambda *args: {})
    monkeypatch.setattr(run.original, "fold_data", lambda *args: ([], [], []))
    run.data.write_once(root / "authorization.json", {"max_new_real_fits": 19, "nonrefundable_inherited_fits": 5})
    run.data.write_once(root / "source-authorization.json", {"synthetic": True})
    run.data.write_once(root / "data-preflight.json", {"synthetic": True})
    run.data.write_once(root / "engineering-acceptance.json", {"passed": True, "new_real_fits": 0, "code": {}})
    run.data.write_once(repair / "new-plan.json", {"budget": {"new_total_maximum": 19}})
    run.data.write_once(old / "historical-decision.json", {"status": "STOPPED_TECHNICAL_FAILURE"})
    run.data.write_once(old / "protocol.json", {"selection": "synthetic-fixed-rule"})
    proofs = []
    for slot in run.INHERITED:
        for folder, suffix in (
            ("attempts", ".json"),
            ("started", ".json"),
            ("models", ".joblib"),
            ("model-manifests", ".json"),
            ("predictions", ".json"),
        ):
            run.data.write_once(old / folder / (slot + suffix), {"synthetic": True})
        if slot in run.INHERITED[:4]:
            run.data.write_once(old / "checkpoints" / (slot + ".json"), {"synthetic": True})
        for suffix in (".joblib", ".manifest.json"):
            run.data.write_once(repair / "verified-copies" / (slot + suffix), {"synthetic": True})
        proofs.append(
            {
                "slot": slot,
                "manifest_sha256": run.data.file_hash(repair / "verified-copies" / (slot + ".manifest.json")),
                "state_sha256": slot.split("-")[1],
            }
        )
    run.data.write_once(repair / "final-zero-fit-verification.json", {"results": proofs})
    monkeypatch.setattr(
        run.state,
        "restore_model",
        lambda directory, slot, sha: {
            "variant": slot.split("-")[1],
            "training_hash": run.data.digest([]),
            "weights_hash": run.data.digest([]),
        },
    )
    monkeypatch.setattr(run.state, "verify_all_predictions", lambda *args: {"synthetic_all_rows": True})
    monkeypatch.setattr(run.state, "model_state_digest", lambda model: model["variant"])
    children = []

    def child(path, command, slot):
        assert command == "_restore"
        binding = run.data.read(root / "run-binding.json")
        assert binding["protocol_hash"] == run.data.digest(run.data.read(path / "protocol.json"))
        children.append(slot)

    monkeypatch.setattr(run, "child", child)
    path = run.preflight()
    spec = run.data.read(path / "protocol.json")
    assert spec["inherited_fits"] == 5 and spec["maximum_new_fits"] == 19
    assert len(spec["bridge"]) == 5
    assert spec["bridge"][run.INHERITED[-1]]["state"] == "PENDING_REQUIRED_REPLAY"
    assert children == list(run.INHERITED)
    assert run.preflight() == path
    assert children == list(run.INHERITED)
    assert not run.attempts(path)
    assert run.data.read(old / "historical-decision.json")["status"] == "STOPPED_TECHNICAL_FAILURE"
