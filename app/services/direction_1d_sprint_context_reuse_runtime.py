"""共享父上下文版本的独立运行绑定；保留每轮原生预测产物、校验与采集预算。"""

import importlib
from time import perf_counter

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_core_forward as core
from app.services.direction_1d_sprint_forward_context_reuse import Executor

TERMINALS = {
    116: "stock_tail",
    117: "stock_moneyflow",
    118: "stock_exchange_spread",
    119: "stock_exchange_return",
}


def require(condition, reason):
    if not condition:
        raise ValueError("CONTEXT_REUSE_RUNTIME_" + reason)


def root(number):
    require(number in TERMINALS, "UNKNOWN_TERMINAL")
    return base.ROOT / f"context-reuse-runtime-r{number}-v1"


def terminal(number):
    require(number in TERMINALS, "UNKNOWN_TERMINAL")
    return importlib.import_module(f"app.services.direction_1d_sprint_{TERMINALS[number]}_forward")


def bindings(number):
    """每次操作前后都检查独立计划全部代码与不可变产物；不修改或补建原模型运行计划。"""
    plan = base.read(root(number) / "plan.json")
    require(plan["terminal_round"] == number and plan["status"] == "MODEL_NOT_RELEASED", "PLAN_CHANGED")
    require(plan["new_source_request_budget"] == 0 and plan["new_fits"] == 0, "SCOPE_CHANGED")
    require(plan["same_question_branches"] == number - 81, "BRANCH_COUNT_CHANGED")
    require(
        plan["native_preflight_rounds"]
        and plan["native_preflight_rounds"][-1] == number
        and len(set(plan["native_preflight_rounds"])) == len(plan["native_preflight_rounds"]),
        "PREFLIGHT_LIST_INVALID",
    )
    for name, expected in plan["code_hashes"].items():
        require(core.sha(base.PROJECT / name) == expected, "CODE_CHANGED")
    for name, expected in plan["artifact_hashes"].items():
        require(base.digest(base.read(base.ROOT / name)) == expected, "ARTIFACT_CHANGED")
    require(core.sha(root(number) / "design.md") == plan["design_sha256"], "DESIGN_CHANGED")
    audit = base.read(base.ROOT / plan["equivalence_result"])
    require(audit["all_layers_equal"] and audit["layers"] == len(plan["module_chain"]), "EQUIVALENCE_NOT_PASSED")
    module = terminal(number)
    require(module.model.root().name == f"round-{number}", "TERMINAL_MODEL_CHANGED")
    original = base.read(module.model.root() / "runtime-plan.json")
    require(base.digest(original) == plan["native_runtime_plan_hash"], "NATIVE_RUNTIME_PLAN_CHANGED")
    return plan, module


def execute(number, action):
    """原锁由CLI持有；每次使用全新Executor，原run中的实际时点与逐题回读完全保留。"""
    require(action in ("preflight", "run"), "UNKNOWN_ACTION")
    started = perf_counter()
    plan, module = bindings(number)
    executor = Executor(module, audit_only=action == "preflight")
    require([m.__name__ for m in executor.modules] == plan["module_chain"], "PARENT_GRAPH_CHANGED")
    checks = []
    if action == "preflight":
        # 按计划执行原生试推理；原脚本明确标记为dry-run，不冒充提前预测。
        for selected in plan["native_preflight_rounds"]:
            require(selected in TERMINALS and selected <= number, "PREFLIGHT_SCOPE_CHANGED")
            value = terminal(selected).preflight()
            require(value["kind"] == "DRY_RUN_ONLY_NOT_ACTUAL_FORWARD" and value["checks"] >= 30, "PREFLIGHT_FAILED")
            checks.append({"round": selected, "checks": value["checks"], "native_result_hash": base.digest(value)})
        contexts = executor.audit_contexts()
        result = {"context_layers": len(contexts), "native_preflights": checks, "new_forecasts": 0}
    else:
        # 父run由惰性代理按原顺序真实完成；没有把父run替换成未执行的空操作。
        result = executor.run()
        require(all(s["phase"] == "COMPLETED" for s in executor.states.values()), "LAYER_NOT_COMPLETED")
    checked_plan, _ = bindings(number)
    require(checked_plan == plan, "PLAN_CHANGED_DURING_RUN")
    receipt = {
        "at": base.now().isoformat(),
        "plan_hash": base.digest(plan),
        "action": action,
        "terminal_round": number,
        "elapsed_seconds": perf_counter() - started,
        "context_calls": {m.model.root().name: executor.states[m.__name__]["context_calls"] for m in executor.modules},
        "native_result_hash": base.digest(result),
        "native_preflights": checks,
        "same_question_branches": plan["same_question_branches"],
        "added_source_requests_by_orchestrator": 0,
        "new_fits": 0,
        "future_deadline_performance_proven": False,
        "status": "SUCCEEDED",
    }
    base.save(root(number) / ("preflight.json" if action == "preflight" else "last-run.json"), receipt, replace=True)
    return receipt
