"""验证原调用顺序、父层完成/失败边界、校验继续执行与原模块不被修改。"""

from types import ModuleType

import pytest
from app.services.direction_1d_sprint_forward_context_reuse import Executor


def layer(name, log, parent=None):
    module = ModuleType(name)
    module.log, module.label = log, name
    module.prior_forward = parent
    code = """
def validate(value):
    log.append((label, "validate"))
    if value["valid"] is not True:
        raise ValueError("actual_validator_rejected")
def context():
    prior = prior_forward.context() if prior_forward is not None else {"valid": True, "items": [1]}
    validate(prior)
    log.append((label, "context"))
    return {"valid": True, "items": list(prior["items"]) + [label]}
def report(loaded=None):
    value = loaded or context()
    validate(value)
    log.append((label, "report"))
    return {"valid": True, "items": value["items"]}
def run():
    log.append((label, "early_capture"))
    if prior_forward is not None:
        prior_forward.run()
    value = context()
    log.append((label, "forecast_and_readback"))
    return report(value)
"""
    exec(code, vars(module))
    if parent is None:
        # 根层没有prior_forward属性，与真实第一层模块相同；函数用独立常量None。
        del module.prior_forward
        exec(code.replace("prior_forward", "root_parent"), vars(module))
        module.root_parent = None
    return module


def test_same_answers_prefetch_and_write_order_with_fewer_rebuilt_contexts():
    original_log, reused_log = [], []
    a = layer("a", original_log)
    b = layer("b", original_log, a)
    c = layer("c", original_log, b)
    expected = c.run()
    x = layer("a", reused_log)
    y = layer("b", reused_log, x)
    z = layer("c", reused_log, y)
    executor = Executor(z)
    assert executor.run() == expected
    selected = {"early_capture", "forecast_and_readback", "report"}
    assert [v for v in original_log if v[1] in selected] == [v for v in reused_log if v[1] in selected]
    assert sum(v[1] == "context" for v in original_log) == 6
    assert sum(v[1] == "context" for v in reused_log) == 3
    assert sum(v[1] == "validate" for v in reused_log) == 6


def test_original_functions_globals_and_parent_module_are_untouched():
    log = []
    a, b = layer("a", log), None
    b = layer("b", log, a)
    original = b.run, b.context, b.report, dict(vars(b))
    Executor(b).run()
    assert (b.run, b.context, b.report) == original[:3]
    assert vars(b) == original[3] and b.prior_forward is a


def test_reused_parent_context_is_copied_and_cannot_change_parent_evidence():
    log = []
    a = layer("a", log)
    b = layer("b", log, a)
    executor = Executor(b)
    executor.run()
    proxy = executor.clones["b"]["run"].__globals__["prior_forward"]
    value = proxy.context()
    value["items"].append("tampered")
    assert "tampered" not in executor.states["a"]["context"]["items"]


def test_parent_validation_failure_propagates_and_no_child_forecast_is_written():
    log = []
    a = layer("a", log)
    exec('def validate(value):\n    raise ValueError("actual_hash_rejected")', vars(a))
    b = layer("b", log, a)
    executor = Executor(b)
    with pytest.raises(ValueError, match="actual_hash_rejected"):
        executor.run()
    assert ("b", "forecast_and_readback") not in log
    assert executor.states["a"]["phase"] == executor.states["b"]["phase"] == "FAILED"


def test_parent_must_complete_before_child_can_reuse_context():
    a = layer("a", [])
    b = layer("b", [], a)
    executor = Executor(b)
    proxy = executor.clones["b"]["run"].__globals__["prior_forward"]
    with pytest.raises(ValueError, match="PARENT_NOT_COMPLETED"):
        proxy.context()


def test_audit_never_runs_or_labels_parent_as_completed():
    log = []
    a = layer("a", log)
    b = layer("b", log, a)
    executor = Executor(b, audit_only=True)
    assert executor.audit_contexts()["b"]["items"] == [1, "a", "b"]
    assert all(state["phase"] == "CONTEXT_READY" for state in executor.states.values())
    assert not any(v[1] in {"early_capture", "forecast_and_readback"} for v in log)
    with pytest.raises(ValueError):
        executor.run()


def test_each_half_hour_requires_a_fresh_executor():
    executor = Executor(layer("a", []))
    executor.run()
    with pytest.raises(ValueError, match="ALREADY_USED"):
        executor.run()


def test_parent_cycle_is_rejected_before_any_work():
    a = layer("a", [])
    a.prior_forward = a
    with pytest.raises(ValueError, match="CYCLIC_PARENT_GRAPH"):
        Executor(a)


def test_core_helper_imported_modules_do_not_prevent_data_isolation():
    log = []
    a = layer("a", log)
    a.imported_helper = ModuleType("frozen_helper_dependency")
    exec(
        'def context():\n    return {"valid": True, "items": [1], "helper": {"module": imported_helper}}',
        vars(a),
    )
    b = layer("b", log, a)
    executor = Executor(b)
    executor.run()
    proxy = executor.clones["b"]["run"].__globals__["prior_forward"]
    copied = proxy.context()
    assert copied["helper"]["module"] is a.imported_helper
    copied["items"].append("changed")
    assert executor.states["a"]["context"]["items"] == [1]
