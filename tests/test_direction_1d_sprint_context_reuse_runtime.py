"""独立运行入口的冻结代码/父运行计划/等价验收和分支边界不能被状态说明替代。"""

from types import SimpleNamespace

import pytest
from app.services import direction_1d_sprint_context_reuse_runtime as s


@pytest.fixture
def binding(monkeypatch, tmp_path):
    monkeypatch.setattr(s.base, "ROOT", tmp_path)
    monkeypatch.setattr(s.base, "PROJECT", tmp_path)
    root = s.root(116)
    root.mkdir()
    entry = tmp_path / "entry.py"
    entry.write_text("frozen entry", encoding="utf-8")
    (root / "design.md").write_text("frozen design", encoding="utf-8")
    native = tmp_path / "round-116"
    s.base.save(native / "runtime-plan.json", {"bound": "native"})
    s.base.save(tmp_path / "audit.json", {"all_layers_equal": True, "layers": 1})
    s.base.save(tmp_path / "artifact.json", {"proof": "original"})
    module = SimpleNamespace(model=SimpleNamespace(root=lambda: native))
    monkeypatch.setattr(s, "terminal", lambda number: module)
    plan = {
        "terminal_round": 116,
        "status": "MODEL_NOT_RELEASED",
        "new_source_request_budget": 0,
        "new_fits": 0,
        "same_question_branches": 35,
        "native_preflight_rounds": [116],
        "code_hashes": {"entry.py": s.core.sha(entry)},
        "artifact_hashes": {"artifact.json": s.base.digest({"proof": "original"})},
        "design_sha256": s.core.sha(root / "design.md"),
        "equivalence_result": "audit.json",
        "module_chain": ["one_layer"],
        "native_runtime_plan_hash": s.base.digest({"bound": "native"}),
    }
    s.base.save(root / "plan.json", plan)
    return root, plan, module


def test_valid_bindings_do_not_create_or_modify_native_model_plan(binding):
    root, plan, module = binding
    before = (module.model.root() / "runtime-plan.json").read_bytes()
    assert s.bindings(116) == (plan, module)
    assert (module.model.root() / "runtime-plan.json").read_bytes() == before


@pytest.mark.parametrize("field", ["code", "artifact", "audit", "native", "branches", "no_preflight"])
def test_changed_evidence_rejected_before_any_run(binding, field):
    root, plan, module = binding
    if field == "code":
        (s.base.PROJECT / "entry.py").write_text("changed", encoding="utf-8")
    elif field == "artifact":
        s.base.save(s.base.ROOT / "artifact.json", {"proof": "changed"}, replace=True)
    elif field == "audit":
        s.base.save(s.base.ROOT / "audit.json", {"all_layers_equal": False, "layers": 1}, replace=True)
    elif field == "native":
        s.base.save(module.model.root() / "runtime-plan.json", {"bound": "changed"}, replace=True)
    else:
        plan["same_question_branches" if field == "branches" else "native_preflight_rounds"] = (
            36 if field == "branches" else []
        )
        s.base.save(root / "plan.json", plan, replace=True)
    with pytest.raises(ValueError):
        s.bindings(116)


def test_unknown_terminal_cannot_select_arbitrary_module_or_path():
    with pytest.raises(ValueError, match="UNKNOWN_TERMINAL"):
        s.root(1000)
