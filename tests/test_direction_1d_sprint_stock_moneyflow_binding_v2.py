"""必须绑定已经完成的训练和当前实现，不允许借恢复流程替换模型。"""

import pytest
from app.services import direction_1d_sprint_stock_moneyflow_binding_v2 as m


@pytest.fixture
def saved(monkeypatch, tmp_path):
    monkeypatch.setattr(m.base, "ROOT", tmp_path)
    monkeypatch.setattr(m.base, "PROJECT", tmp_path)
    root = m.model.root()
    root.mkdir()
    (root / "runtime-design.md").write_text("synthetic", encoding="utf-8")
    (tmp_path / "entry.py").write_text("synthetic entry", encoding="utf-8")
    plan = {"synthetic": "plan"}
    result = {"plan_hash": m.base.digest(plan), "development_fits": 24, "current_fits": 3}
    completion = {"status": "TRAINED_ANALYZED_INDEPENDENT_RUNTIME_PENDING"}
    intent = {
        "model_plan_hash": m.base.digest(plan),
        "model_result_hash": m.base.digest(result),
        "training_completion_hash": m.base.digest(completion),
        "same_question_branches": 39,
        "first_target": "2026-09-17",
        "new_request_budget": 1,
        "code_hashes": {"entry.py": m.core.sha(tmp_path / "entry.py")},
        "design_sha256": m.core.sha(root / "runtime-design.md"),
    }
    for name, value in [
        ("plan", plan),
        ("result", result),
        ("training-completion", completion),
        ("runtime-intent", intent),
    ]:
        m.base.save(root / f"{name}.json", value)
    return root, tmp_path


def test_completed_model_can_bind(saved):
    assert m.verify_intent()["same_question_branches"] == 39


@pytest.mark.parametrize("artifact", ["plan", "result", "training-completion"])
def test_replaced_artifact_rejected(saved, artifact):
    root, _ = saved
    value = m.base.read(root / f"{artifact}.json") | {"changed": True}
    m.base.save(root / f"{artifact}.json", value, replace=True)
    with pytest.raises(ValueError, match="RUNTIME_INTENT_CHANGED"):
        m.verify_intent()


def test_changed_entry_rejected(saved):
    _, project = saved
    (project / "entry.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="RUNTIME_CODE_CHANGED"):
        m.verify_intent()
