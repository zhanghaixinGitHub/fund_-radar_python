"""运行版本不得覆盖冻结训练；父模型未完成与运行绑定变化均拒绝接入。"""

from datetime import datetime

import pytest
from app.services import direction_1d_sprint_stock_activity_runtime as s


@pytest.fixture
def runtime_fixture(monkeypatch, tmp_path):
    source = tmp_path / "runtime.py"
    source.write_text("frozen runtime fixture", encoding="utf-8")
    design = tmp_path / "runtime-design.md"
    design.write_text("frozen design fixture", encoding="utf-8")
    train = {"plan": "train"}
    result = {"model": "trained"}
    baseline = {"source": "frozen twenty prior days"}
    s.base.save(tmp_path / "future-baseline.json", baseline)
    intent = {
        "baseline_hash": s.base.digest(baseline),
        "model_plan_hash": s.base.digest(train),
        "model_result_hash": s.base.digest(result),
        "parent_round": 114,
        "same_question_branches": 34,
        "new_request_budget": 0,
        "first_target": "2026-09-17",
        "code_hashes": {"runtime.py": s.core.sha(source)},
        "design_sha256": s.core.sha(design),
    }
    monkeypatch.setattr(s.model, "root", lambda: tmp_path)
    monkeypatch.setattr(s.base, "PROJECT", tmp_path)
    monkeypatch.setattr(s.base, "ROOT", tmp_path)
    for name, value in (("plan.json", train), ("result.json", result), ("runtime-intent.json", intent)):
        s.base.save(tmp_path / name, value)
    return tmp_path, intent


def test_frozen_runtime_files_verified_before_parent_is_ready(runtime_fixture):
    _, intent = runtime_fixture
    assert s.verify_intent() == intent


def test_changed_runtime_code_cannot_reuse_training_evidence(runtime_fixture):
    root, _ = runtime_fixture
    (root / "runtime.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="RUNTIME_CODE_CHANGED"):
        s.verify_intent()


def test_runtime_plan_is_not_created_before_parent_model_exists(runtime_fixture, monkeypatch):
    root, _ = runtime_fixture
    monkeypatch.setattr(s.model, "models", lambda: ({}, {}))

    def missing():
        raise FileNotFoundError("parent not trained")

    monkeypatch.setattr(s.parent, "models", missing)
    with pytest.raises(FileNotFoundError):
        s.plan()
    assert not (root / "runtime-plan.json").exists()


@pytest.mark.parametrize("change", ["plan", "time"])
def test_rehashed_answer_cannot_use_different_or_late_runtime(monkeypatch, change):
    frozen = {"at": "2026-09-16T17:00:00+08:00"}
    monkeypatch.setattr(s, "bindings", lambda: frozen)
    value = {"at": "2026-09-17T08:15:00+08:00", "runtime_plan_hash": s.base.digest(frozen)}
    s.validate_forecast_binding(value)
    if change == "plan":
        value["runtime_plan_hash"] = "changed"
    else:
        value["at"] = "2026-09-16T16:00:00+08:00"
    with pytest.raises(ValueError, match="RUNTIME_FORECAST_LATE_OR_CHANGED"):
        s.validate_forecast_binding(value)


def test_completed_parent_and_training_bind_once_without_changing_training(runtime_fixture, monkeypatch):
    root, _ = runtime_fixture
    trained = {"fingerprint": {"code": {}}, "plan_hash": "training"}
    prior = {"fingerprint": {"code": {}}, "plan_hash": "parent"}
    monkeypatch.setattr(s.model, "models", lambda: (trained, {}))
    monkeypatch.setattr(s.parent, "models", lambda: (prior, {}))
    monkeypatch.setattr(s.parent_runtime, "plan", lambda: {"code_hashes": {}, "at": "2026-09-16T16:00:00+08:00"})
    monkeypatch.setattr(s.model, "active", lambda: None)
    monkeypatch.setattr(s.base, "now", lambda: datetime.fromisoformat("2026-09-16T17:00:00+08:00"))
    # 测试替身只模拟已验证的上下游；真实入口额外逐一重算四份模型产物摘要。
    monkeypatch.setattr(s, "bindings", lambda: s.base.read(root / "runtime-plan.json"))
    before = (root / "plan.json").read_bytes(), (root / "result.json").read_bytes()
    frozen = s.plan()
    assert frozen["artifact_hashes"]["round-114/result.json"] == s.base.digest(prior)
    assert s.plan() == frozen
    assert before == ((root / "plan.json").read_bytes(), (root / "result.json").read_bytes())
    assert (root / "runtime-code/runtime.py").is_file()


def test_changed_parent_model_cannot_rebind_saved_runtime(runtime_fixture, monkeypatch):
    root, intent = runtime_fixture
    s.base.save(
        root / "runtime-plan.json",
        {"intent_hash": s.base.digest(intent), "code_hashes": {}, "artifact_hashes": {"result.json": "old"}},
    )
    with pytest.raises(ValueError, match="RUNTIME_ARTIFACT_CHANGED"):
        s.bindings()


def test_trained_parent_without_actual_runtime_binding_is_not_enough(runtime_fixture, monkeypatch):
    root, _ = runtime_fixture
    trained = {"fingerprint": {"code": {}}, "plan_hash": "training"}
    monkeypatch.setattr(s.model, "models", lambda: (trained, {}))
    monkeypatch.setattr(s.parent, "models", lambda: (trained, {}))

    def pending():
        raise FileNotFoundError("parent runtime not bound")

    monkeypatch.setattr(s.parent_runtime, "plan", pending)
    with pytest.raises(FileNotFoundError):
        s.plan()
    assert not (root / "runtime-plan.json").exists()
